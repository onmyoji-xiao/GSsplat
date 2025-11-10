from torch import nn, Tensor
# from transformers import BertTokenizer, BertModel, BertConfig
from einops import rearrange, einsum, repeat
import torch
import time
import torch.nn.functional as F

from .gaussian import sample_image_grid, get_world_rays, rotate_sh, build_covariance
from dataclasses import dataclass
from jaxtyping import Float
from models.backbone.backbone_multiview import BackboneMultiview, BackboneMultiview_test
from models.backbone.resnet2D import ResnetBlock2D
from models.depth_wrapper import CasMVSNet
import torch_scatter


@dataclass
class Gaussians:
    name: str
    means: Float[Tensor, "batch gaussian dim"]
    covariances: Float[Tensor, "batch gaussian dim dim"]
    colors: Float[Tensor, "batch gaussian d_color"]
    opacities: Float[Tensor, "batch gaussian"]


class SemUP(torch.nn.Module):
    def __init__(
            self,
            in_ch=128,
            feature_ch=128,
            num_blocks=3,
            upscale_factor=4.0,
            resnet_groups=32
    ):
        super(SemUP, self).__init__()
        self.feature_ch = feature_ch

        self.cnn_blocks = []
        for i in range(num_blocks):
            self.cnn_blocks.append(ResnetBlock2D(
                in_channels=self.feature_ch if i > 0 else in_ch,
                out_channels=self.feature_ch,
                groups=resnet_groups if i > 0 else 1,
                dropout=0.3,
                up=True if i < (upscale_factor / 2) else False
            ))
        self.cnn_blocks = nn.ModuleList(self.cnn_blocks)

    def forward(self, x):
        ''' images: (B, N_Views, C, H, W), range [0, 1] '''
        # resolution low to high
        for block in self.cnn_blocks:
            x = block(x)
        return x


class Encoder(nn.Module):
    def __init__(self, cfg, use_depth=False):
        super().__init__()
        self.cfg = cfg
        self.d_feature = self.cfg.out_channel
        self.use_depth = use_depth

        self.d_sh = (self.cfg.sh_degree + 1) ** 2
        self.register_buffer(
            "sh_mask",
            torch.ones((self.d_sh,), dtype=torch.float32),
            persistent=False,
        )
        for degree in range(1, self.cfg.sh_degree + 1):
            self.sh_mask[degree ** 2: (degree + 1) ** 2] = 0.1 * 0.25 ** degree

        self.backbone = BackboneMultiview(
            feature_channels=self.d_feature,
            resnet_groups=self.cfg.resnet_groups,
            cnn_num_blocks=5,
            att_num_blocks=3,
            num_attention_heads=self.cfg.attention_heads
        )

        if self.cfg.depth_em:
            self.depth_estimation = CasMVSNet()

        upscale_factor = self.cfg.upscale_factor
        self.cnn_up = nn.Sequential(
            nn.Conv2d(self.d_feature, self.d_feature, 3, 1, 1),
            nn.Upsample(
                scale_factor=upscale_factor / 2,
                mode="bilinear",
                align_corners=True,
            ),
            nn.GELU(),
            nn.Conv2d(self.d_feature, self.d_feature, 3, 1, 1),
        )

        self.semantic_up = SemUP(
            in_ch=self.d_feature + 1 if self.cfg.depth_add else self.d_feature,
            feature_ch=self.d_feature,
            num_blocks=self.cfg.num_blocks,
            upscale_factor=upscale_factor,
            resnet_groups=self.cfg.resnet_groups,
        )
        if self.cfg.sem_2d_loss:
            self.projection_sem = nn.Conv2d(self.d_feature, self.cfg.nb_class, 1)

        in_ch = self.d_feature
        gaussian_sem_channels = self.cfg.offset_dim + 3 + 4 + self.cfg.nb_class + 1
        self.to_sem_gaussians = nn.Sequential(
            nn.Conv2d(in_ch, in_ch * 2, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(in_ch * 2, gaussian_sem_channels, kernel_size=3, stride=1, padding=1),
        )
        if self.cfg.aggretate:
            self.sem_feat_map = nn.Sequential(
                nn.Conv2d(self.d_feature * 2, self.d_feature, kernel_size=3, stride=1, padding=1),
                nn.GELU()
            )
            self.rgb_feat_map = nn.Sequential(
                nn.Conv2d(self.d_feature * 2, self.d_feature, kernel_size=3, stride=1, padding=1),
                nn.GELU()
            )

        gaussian_rgb_channels = self.cfg.offset_dim + 3 + 4 + 75 + 1
        if self.cfg.rgb_add:
            in_ch = in_ch + 3
        self.to_rgb_gaussians = nn.Sequential(
            nn.Conv2d(in_ch, in_ch * 2, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(in_ch * 2, gaussian_rgb_channels, kernel_size=3, stride=1, padding=1),
        )

    def get_scale_multiplier(
            self,
            intrinsics,
            pixel_size,
            multiplier=0.1,
    ):
        xy_multipliers = multiplier * einsum(
            intrinsics[..., :2, :2].inverse(),
            pixel_size,
            "... i j, j -> ... i",
        )
        return xy_multipliers.sum(dim=-1)

    def create_gaussian(
            self,
            extrinsics: Float[Tensor, "*#batch 4 4"],
            intrinsics: Float[Tensor, "*#batch 3 3"],
            means: Float[Tensor, "*#batch 3"],
            depths: Float[Tensor, "*#batch"],
            opacities: Float[Tensor, "*#batch"],
            raw_gaussians: Float[Tensor, "*#batch _"],
            color_num: int,
            eps: float = 1e-8
    ):
        device = extrinsics.device
        if self.cfg.offset_dim > 0:
            offset, scales, rotations, colors = raw_gaussians.split((self.cfg.offset_dim, 3, 4, color_num), dim=-1)
            offset_sig = (offset[..., 0].sigmoid() > 0.5)
            re_offset = offset[..., 1:].sigmoid() * depths[..., None] / self.cfg.far * offset_sig[..., None].float()
            means = means + re_offset
        else:
            scales, rotations, colors = raw_gaussians.split((3, 4, color_num), dim=-1)
            offset_sig = None

        # Map scale features to valid scale range.
        scale_min = self.cfg.gaussian_scale_min
        scale_max = self.cfg.gaussian_scale_max
        scales = scale_min + (scale_max - scale_min) * scales.sigmoid()

        pixel_size = torch.tensor([1.0, 1.0], dtype=torch.float32, device=device)
        multiplier = self.get_scale_multiplier(intrinsics, pixel_size)
        scales = scales * depths[..., None] * multiplier[..., None]  # b v n 1 1 3

        # Normalize the quaternion features to yield a valid quaternion.
        rotations = rotations / (rotations.norm(dim=-1, keepdim=True) + eps)  # b v n 1 1 4

        # Create world-space covariance matrices.
        covariances = build_covariance(scales, rotations)
        c2w_rotations = extrinsics[..., :3, :3]  # b v 1 1 1 3 3
        covariances = c2w_rotations @ covariances @ c2w_rotations.transpose(-1, -2)

        if color_num == 3 * self.d_sh:
            sh = rearrange(colors, "... (xyz d_sh) -> ... xyz d_sh", xyz=3)
            sh = sh * self.sh_mask
            sh = rotate_sh(sh, c2w_rotations[..., None, :, :])
            colors = rearrange(sh, "... xyz d_sh -> ... (xyz d_sh)")

        return means, covariances, colors, opacities, re_offset

    def feat_3d(self, gs_points, gs_feats):
        unit = 0.1
        temp_points = (gs_points / unit).trunc()
        gs_feats_a = []
        for pts, fts, in_pts in zip(temp_points, gs_feats, gs_points):
            agg_points, mapping = torch.unique(pts.clone(), dim=0, return_inverse=True)
            agg_feats = torch_scatter.scatter(fts, mapping, dim=0, reduce='mean')
            avg_feats = agg_feats[mapping]
            distances = torch.norm((in_pts / unit - (pts + unit / 2)), p=2, dim=1)
            gs_feats_a.append(avg_feats * distances[..., None] + fts)

        gs_feats_a = torch.stack(gs_feats_a)
        return torch.cat((gs_feats, gs_feats_a), dim=2)
        # return gs_feats_a

    def forward(self, batch: dict):
        device = batch["imgs"].device
        ref_images = batch['imgs'][:, :-1]
        ref_intrinsics = batch['intrinsics'][:, :-1]
        ref_extrinsics = batch['poses'][:, :-1]

        b, v, c, h, w = ref_images.shape
        t0 = time.time()
        depth_map = None
        if self.cfg.depth_em:
            feats_vol, feats_fpn, depth_map, depth_values = self.depth_estimation(
                ref_images,
                affine_mats=batch["affine_mats"][:, :-1],
                affine_mats_inv=batch["affine_mats_inv"][:, :-1],
                near_far=batch["near_fars"][:, :-1],
                closest_idxs=batch["closest_idxs"],
                gt_depths=batch["depths"][:, :-1],
            )
            ref_depths = depth_map['level_0']
        else:
            ref_depths = batch['depths'][:, :-1]

        t1 = time.time()
        trans_features, cnn_features = self.backbone(ref_images)

        xy_ray, _ = sample_image_grid((h, w), device)
        xy_ray = rearrange(xy_ray, "h w xy -> (h w) () xy").float()
        xy_ray = repeat(xy_ray, "r srf xy -> b v r srf () xy", b=b, v=v)
        ref_extrinsics = rearrange(ref_extrinsics, "b v i j -> b v () () () i j")
        ref_intrinsics = rearrange(ref_intrinsics, "b v i j -> b v () () () i j")
        origins, directions = get_world_rays(xy_ray, ref_extrinsics, ref_intrinsics)

        _depths = rearrange(ref_depths, "b v h w -> b v (h w) () () ()", h=h, w=w)
        means_init = origins + directions * _depths

        # points = rearrange(means_init, "b v n 1 1 c -> b (v n) c")[0]
        # import matplotlib.pyplot as plt
        # import matplotlib
        # from mpl_toolkits.mplot3d import Axes3D
        # matplotlib.use('Agg')
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # points = points[::1000].cpu().numpy()
        # colors = ['red' for _ in range(len(points))]
        # ax.set_zlim(-1.5, 1.5)
        # ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, cmap='viridis', marker='o')
        # plt.savefig('ori_points.png')
        ref_images = rearrange(ref_images, "b v c h w -> (b v) c h w")
        if self.cfg.render_sem:
            # semantic render
            if self.cfg.depth_add:
                if self.use_depth and not self.cfg.depth_em:
                    cat_depths = rearrange(batch['depths_']['level_2'][:, :v], "b v h w -> (b v) () h w")
                else:
                    cat_depths = rearrange(depth_map['level_2'], "b v h w -> (b v) () h w") / self.cfg.far
                sem_2d_feats = self.semantic_up(torch.cat((trans_features, cat_depths), dim=1))
            else:
                sem_2d_feats = self.semantic_up(trans_features)

            gs_2d_feats = sem_2d_feats.clone()
            if self.cfg.aggretate:
                gs_2d_feats = self.feat_3d(rearrange(means_init, "b v n 1 1 c -> b (v n) c"),
                                           rearrange(gs_2d_feats, "(b v) c h w -> b (v h w) c", b=b, v=v))
                gs_2d_feats = rearrange(gs_2d_feats, "b (v h w) c -> (b v) c h w", b=b, v=v, h=h, w=w)
                gs_2d_feats = self.sem_feat_map(gs_2d_feats)

            sem_gaussian_feats = self.to_sem_gaussians(gs_2d_feats)  # (bv,c,h,w)
            sem_raw_gaussians = rearrange(sem_gaussian_feats, "(b v) c h w -> b v (h w) c", b=b, v=v)
            if self.cfg.sem_2d_loss:
                sem_2d_logits = rearrange(self.projection_sem(sem_2d_feats), "b c h w -> b h w c")
            else:
                sem_2d_logits = None
        else:
            sem_2d_feats = None
            sem_2d_logits = None

        if self.cfg.render_rgb:
            rgb_2d_feats = self.cnn_up(cnn_features)

            if self.cfg.aggretate:
                rgb_2d_feats = self.feat_3d(rearrange(means_init, "b v n 1 1 c -> b (v n) c"),
                                            rearrange(rgb_2d_feats, "(b v) c h w -> b (v h w) c", b=b, v=v))
                rgb_2d_feats = rearrange(rgb_2d_feats, "b (v h w) c -> (b v) c h w", b=b, v=v, h=h, w=w)
                rgb_2d_feats = self.rgb_feat_map(rgb_2d_feats)

            if self.cfg.rgb_add:
                rgb_2d_feats = torch.concat((rgb_2d_feats, ref_images), dim=1)

            rgb_raw_gaussians = self.to_rgb_gaussians(rgb_2d_feats)
            rgb_raw_gaussians = rearrange(rgb_raw_gaussians, "(b v) c h w -> b v (h w) c", b=b, v=v)
        t2 = time.time()

        offset_dict = {}
        if self.cfg.render_sem:
            # semantic
            sem_gaussians = rearrange(sem_raw_gaussians, "... (srf c) -> ... srf c", srf=1)
            opacity = torch.sigmoid(sem_gaussians[..., -1])
            means, covariances, colors, opacities, re_offset = self.create_gaussian(
                ref_extrinsics,
                ref_intrinsics,
                means_init,
                rearrange(ref_depths, "b v h w -> b v (h w) () ()"),
                rearrange(opacity, "b v r c -> b v r () c"),  # b v h*w 1 1
                rearrange(sem_gaussians[..., :-1], "b v r srf c -> b v r srf () c"),
                self.cfg.nb_class
            )
            sem_gs = Gaussians(
                "semantic",
                rearrange(means, "b v r srf spp xyz -> b (v r srf spp) xyz", ),
                rearrange(covariances, "b v r srf spp i j -> b (v r srf spp) i j", ),
                rearrange(colors, "b v r srf spp d_c -> b (v r srf spp) d_c", ),
                rearrange(opacities, "b v r srf spp -> b (v r srf spp)", ))
            if re_offset is not None:
                offset_dict['sem'] = rearrange(re_offset, "b v r srf spp c -> b (v r srf spp) c", )
        else:
            sem_gs = None
        t3 = time.time()
        # rgb
        if self.cfg.render_rgb:
            rgb_gaussians = rearrange(rgb_raw_gaussians, "... (srf c) -> ... srf c", srf=1)
            opacity = torch.sigmoid(rgb_gaussians[..., -1])
            means, covariances, sh, opacities, re_offset = self.create_gaussian(
                ref_extrinsics,
                ref_intrinsics,
                means_init,
                rearrange(ref_depths, "b v h w -> b v (h w) () ()"),
                rearrange(opacity, "b v r c -> b v r () c"),  # b v h*w 1 1
                rearrange(rgb_gaussians[..., :-1], "b v r srf c -> b v r srf () c"),
                3 * self.d_sh
            )

            rgb_gs = Gaussians(
                "rgb",
                rearrange(means, "b v r srf spp xyz -> b (v r srf spp) xyz", ),
                rearrange(covariances, "b v r srf spp i j -> b (v r srf spp) i j", ),
                rearrange(sh, "b v r srf spp d_c -> b (v r srf spp) d_c", ),
                rearrange(opacities, "b v r srf spp -> b (v r srf spp)", ))
            if re_offset is not None:
                offset_dict['rgb'] = rearrange(re_offset, "b v r srf spp c-> b (v r srf spp) c", )
        else:
            rgb_gs = None
        t4 = time.time()
        # print('depth: ', t1 - t0)
        # print('multi-network: ', t2 - t1)
        # print('sem-gaussian: ', t3 - t2)
        # print('rgb-gaussian: ', t4 - t3)
        return sem_gs, rgb_gs, sem_2d_feats, sem_2d_logits, depth_map, offset_dict
