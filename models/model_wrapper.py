import os

import torch
from einops import rearrange, repeat
from pytorch_lightning import LightningModule
from torch import Tensor, nn, optim
import imageio
from .encoder import Encoder
from .cuda_splatting import render_cuda, render_depth_cuda
from .metrics import IoU, mse2psnr
from utils.utils import lable_color_map
import lpips
from skimage.metrics import structural_similarity as SSIM
from utils.image_io import save_image
from models.depth_wrapper import threeD_to_depthimage, depthimage_to_threeD
import torch_scatter
import torch.nn.functional as F
import time
import open3d as o3d
from .midas_loss import ScaleAndShiftInvariantLoss
import random
import numpy as np
import cv2
from pytorch_lightning.utilities import rank_zero_only


def convert_to_buffer(module: nn.Module, persistent: bool = True):
    # Recurse over child modules.
    for name, child in list(module.named_children()):
        convert_to_buffer(child, persistent)

    # Also re-save buffers to change persistence.
    for name, parameter_or_buffer in (
            *module.named_parameters(recurse=False),
            *module.named_buffers(recurse=False),
    ):
        value = parameter_or_buffer.detach().clone()
        delattr(module, name)
        module.register_buffer(name, value, persistent=persistent)


def image_to_pointcloud(imgs, depths, C2W, intrinsics):
    """
    将批次图像+深度图投影为带颜色的世界坐标系点云

    Args:
        imgs: (B, 3, H, W)     —— RGB图像
        depths: (B, H, W)      —— 深度图
        C2W: (B, 4, 4)         —— 相机到世界坐标变换矩阵
        intrinsics: (B, 3, 3)  —— 相机内参矩阵 K

    Returns:
        points_world: (B, H*W, 3) —— 世界坐标系下的3D点
        rgbs: (B, H*W, 3)         —— 对应RGB颜色
    """
    B, _, H, W = imgs.shape
    device = imgs.device

    # Step 1: 生成像素坐标网格 (x, y)
    y, x = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=device),
        torch.arange(W, dtype=torch.float32, device=device),
        indexing='ij'
    )  # (H, W)
    x = x.reshape(-1)  # (H*W,)
    y = y.reshape(-1)  # (H*W,)

    # 扩展到 batch 维度
    x = x.unsqueeze(0).expand(B, -1)  # (B, N)
    y = y.unsqueeze(0).expand(B, -1)  # (B, N)
    z = depths.reshape(B, -1)  # (B, N) —— 真实深度值！

    # Step 2: 构建齐次像素坐标 (x, y, 1) → 但这里我们用深度 z 缩放
    pixel_coords = torch.stack([x, y, torch.ones_like(z)], dim=1)  # (B, 3, N)

    # Step 3: 用内参逆矩阵投影到相机坐标系
    intrinsics_inv = torch.inverse(intrinsics)  # (B, 3, 3)
    cam_coords = torch.bmm(intrinsics_inv, pixel_coords)  # (B, 3, N)

    # 用深度缩放，得到真实的相机坐标系3D点（不是单位射线！）
    cam_coords = cam_coords * z.unsqueeze(1)  # (B, 3, N) —— 关键！深度缩放

    # Step 4: 转换到世界坐标系
    # 添加齐次坐标 1 → (B, 4, N)
    cam_coords_homogeneous = torch.cat([
        cam_coords,
        torch.ones(B, 1, cam_coords.shape[-1], device=device)
    ], dim=1)  # (B, 4, N)

    # 应用 C2W 变换
    world_coords_homogeneous = torch.bmm(C2W, cam_coords_homogeneous)  # (B, 4, N)

    # 去除齐次坐标（除以 w，通常为1，但安全起见）
    points_world = world_coords_homogeneous[:, :3, :] / world_coords_homogeneous[:, 3:4, :]  # (B, 3, N)

    # 转置为 (B, N, 3)
    points_world = points_world.transpose(1, 2)  # (B, N, 3)

    # Step 5: 提取对应 RGB 颜色
    rgbs = imgs.permute(0, 2, 3, 1).reshape(B * H * W, 3)  # (B, H, W, 3) -> (B, N, 3)
    points_world = points_world.reshape(B * H * W, 3)
    return points_world, rgbs


class ModelWrapper(LightningModule):
    def __init__(
            self,
            cfg,
            args,
            writer=None

    ):
        super().__init__()
        self.train_cfg = cfg.trainer
        self.optimizer_cfg = cfg.optimizer
        self.model_cfg = cfg.model
        self.data_cfg = cfg.data
        self.output_dir = args.output_dir
        self.dataname = args.dataset_name
        self.mode = args.mode
        self.writer = writer

        self.near = torch.tensor([self.model_cfg.near])
        self.far = torch.tensor([self.model_cfg.far])
        self.render_h = self.model_cfg.render_h
        self.render_w = self.model_cfg.render_w

        self.background_color = torch.FloatTensor([0])
        # Set up the model.
        self.encoder = Encoder(self.model_cfg, self.data_cfg.use_depth)
        if self.model_cfg.depth_em:
            self.encoder.depth_estimation.load_state_dict(
                torch.load(f'./save/{self.dataname}_depth_estimation_v{self.data_cfg.view_num}.pth',
                           map_location='cpu'), strict=True)

        self.val_step_outputs = {}
        self.wr_cntr = 0
        self.wr_cntr_s = 0

        self.lpips = lpips.LPIPS(net="vgg")
        convert_to_buffer(self.lpips, persistent=False)

        self.lable_color_map = torch.from_numpy(lable_color_map) / 255.0

        weights = [1.0] * self.model_cfg.nb_class
        for c in self.model_cfg.background_class:
            weights[c] = self.optimizer_cfg.background_weight

        class_weights = torch.FloatTensor(weights)
        if self.dataname == 'replica':
            self.semantic_loss = nn.CrossEntropyLoss(ignore_index=self.model_cfg.ignore_label, weight=class_weights)
        else:
            self.semantic_loss = nn.CrossEntropyLoss(weight=class_weights)
        # self.semantic_3d_loss = nn.CosineEmbeddingLoss(reduction='mean')
        self.miou = IoU(num_classes=self.model_cfg.nb_class - 1, ignore_label=self.model_cfg.ignore_label)

    # def get_extra_loss(self, sem_2d_logits, ref_labels, semfeat_2d, gs_points):
    #     b, v, _, _ = ref_labels.shape
    #     stride = ref_labels.shape[-1] // semfeat_2d.shape[-1]
    #     if stride > 1:
    #         ref_labels = ref_labels[:, :, ::int(stride), ::int(stride)]
    #         sem_2d_logits = sem_2d_logits[:, ::int(stride), ::int(stride)]
    #
    #     ref_labels = rearrange(ref_labels, "b v h w -> b (v h w)")
    #     semantic_pred = rearrange(torch.argmax(sem_2d_logits, dim=-1), "(b v) h w -> b (v h w)", b=b, v=v)
    #     semfeat_2d = rearrange(semfeat_2d, "(b v) c h w -> b (v h w) c", b=b, v=v)
    #     acc_mask = (ref_labels == semantic_pred)
    #     acc_pred = ref_labels.clone()
    #
    #     # semfeat_2d = F.max_pool1d(semfeat_2d, kernel_size=2, stride=2)
    #     sem_3d_loss = 0.0
    #     for semfeat_2d_i, acc_pred_i, acc_mask_i, ref_label_i, points_i in zip(semfeat_2d, acc_pred, acc_mask,
    #                                                                            ref_labels, gs_points):
    #         inter_loss, outer_loss = 0.0, 0.0
    #         block_points, block_feats = [], []
    #         c_ids, mapping = torch.unique(acc_pred_i, return_inverse=True)
    #         for c in c_ids:
    #             c_feat_list = semfeat_2d_i[ref_label_i == c]
    #             c_points = points_i[ref_label_i == c]
    #             if len(c_feat_list) < 2:
    #                 continue
    #             c_points = c_points // ((self.model_cfg.far - self.model_cfg.near) / 100)
    #             c_block_points, block_id_mapping = torch.unique(c_points.clone(), dim=0, return_inverse=True)
    #             c_block_feats = torch_scatter.scatter(c_feat_list, block_id_mapping, dim=0, reduce='mean')
    #
    #             cfeats_a = c_block_feats[block_id_mapping]
    #             cfeats_b = c_feat_list
    #             sim = F.cosine_similarity(cfeats_a, cfeats_b, dim=1)
    #             inter_loss = inter_loss + torch.mean(1 - sim)
    #
    #             block_points.append(c_block_points)
    #             block_feats.append(c_block_feats)
    #
    #         block_points = torch.concat(block_points)
    #         block_feats = torch.concat(block_feats)
    #
    #         _, mapping = torch.unique(block_points.detach(), dim=0, return_inverse=True)
    #         m_n = 0
    #         for i in torch.unique(mapping):
    #             n_feats = block_feats[mapping == i]
    #             cnum = len(n_feats)
    #             if cnum < 2:
    #                 continue
    #             m_n += 1
    #             target = (torch.eye(cnum) * 2 - 1).ravel().to(n_feats.device)
    #             cfeats_a = repeat(n_feats, "n c -> (n x) c", x=cnum)
    #             cfeats_b = repeat(n_feats, "n c -> (x n) c", x=cnum)
    #             outer_loss = outer_loss + self.semantic_3d_loss(cfeats_a, cfeats_b, target)
    #         outer_loss = outer_loss / m_n if m_n > 0 else 0.0
    #
    #         sem_3d_loss = sem_3d_loss + inter_loss + outer_loss
    #     return sem_3d_loss / b

    def target_depth_loss(self, gsmeans, ori_depth, intrinsics, extrinsics, offset_mask):
        B, V, H, W = ori_depth.shape

        loss = torch.zeros(1, dtype=torch.float).to(ori_depth.device)
        for bi in range(B):
            points = gsmeans[bi][~offset_mask[bi]]
            if len(points) == 0:
                continue
            target_intrinsics = intrinsics[bi, :V]
            target_extrinsics = extrinsics[bi, :V]
            target_depths = ori_depth[bi, :]
            pred_depths = threeD_to_depthimage(points, target_intrinsics, target_extrinsics, H, W)
            mask = target_depths > 0
            target_loss = F.smooth_l1_loss(target_depths[mask], pred_depths[mask], reduction="mean")
            loss = loss + target_loss

        return loss / B

    def depth_loss(self, depths, gt_depths):
        l1_loss = 0.0
        smooth_loss = 0.0
        if isinstance(depths, dict):
            for l in range(3):
                depth_pred_l = depths[f"level_{l}"]
                V = depth_pred_l.shape[1]

                depth_gt_l = gt_depths[f"level_{l}"]
                depth_gt_l = depth_gt_l[:, :V]
                mask_l = depth_gt_l > 0

                if not self.data_cfg.use_depth:
                    depth_pred_l = F.normalize(depth_pred_l, dim=1, p=1)
                    depth_gt_l = F.normalize(depth_gt_l, dim=1, p=1)
                l1_loss = l1_loss + F.smooth_l1_loss(depth_pred_l[mask_l], depth_gt_l[mask_l]) * 2 ** (1 - l)
                if l == 0:
                    depth_dx = depth_pred_l.diff(dim=-1)
                    depth_dy = depth_pred_l.diff(dim=-2)
                    smooth_loss = smooth_loss + depth_dx.abs().mean() + depth_dy.abs().mean()
        else:
            l1_loss = F.smooth_l1_loss(depths, gt_depths, reduction="mean")
            depth_dx = depths.diff(dim=-1)
            depth_dy = depths.diff(dim=-2)
            smooth_loss = depth_dx.abs().mean() + depth_dy.abs().mean()

        return 0.5 * smooth_loss + l1_loss
        # return l1_loss

    def training_step(self, batch, batch_idx):
        # Run the model.
        device = batch['imgs'].device
        sem_gs, rgb_gs, semfeat_2d, sem_2d_logits, depth_maps, offset_dict = self.encoder(batch)
        tg_intrinsics = batch['intrinsics'][:, -2:]
        tg_extrinsics = batch['poses'][:, -2:]
        tg_labels = batch['labels'][:, -2:]
        tg_images = batch['imgs'][:, -2:]
        b, v, _, _ = tg_intrinsics.shape

        results = {}
        for gs in (sem_gs, rgb_gs):
            if gs is not None:
                render_images = render_cuda(
                    extrinsics=rearrange(tg_extrinsics, "b v i j -> (b v) i j"),
                    intrinsics=rearrange(tg_intrinsics, "b v i j -> (b v) i j"),
                    near=repeat(self.near.to(device), "c -> (n c)", n=b * v),
                    far=repeat(self.far.to(device), "c -> (n c)", n=b * v),
                    image_shape=(self.render_h, self.render_w),
                    background_color=repeat(self.background_color.to(device), "c -> (b v) (n c)", b=b, v=v,
                                            n=self.model_cfg.nb_class),
                    # gaussian_means=repeat(gs.means, "b g xyz -> (b v) g xyz", v=v),
                    # gaussian_covariances=repeat(gs.covariances, "b g i j -> (b v) g i j", v=v),
                    # gaussian_opacities=repeat(gs.opacities, "b g -> (b v) g", v=v),
                    # gaussian_colors=repeat(gs.colors, "b g d_c -> (b v) g d_c", v=v),
                    gaussian_means=gs.means,
                    gaussian_covariances=gs.covariances,
                    gaussian_opacities=gs.opacities,
                    gaussian_colors=gs.colors,
                    view_num=v,
                    use_sh=True if gs.name == "rgb" else False,
                )
                results[gs.name] = render_images
        loss = 0.0
        if sem_gs is not None:
            rendered_sem = rearrange(results["semantic"], "b c h w -> (b h w) c")
            render_sem_loss = self.semantic_loss(rendered_sem, tg_labels.ravel())
            if not torch.isnan(render_sem_loss):
                loss = loss + render_sem_loss
                self.log("train/render_sem_loss", render_sem_loss.item(), on_step=True, prog_bar=False)
            if self.model_cfg.sem_2d_loss:
                sem_2d_logits = rearrange(sem_2d_logits, "b h w c -> (b h w) c")
                sem_2d_loss = self.semantic_loss(sem_2d_logits, batch['labels'][:, :-1].ravel())
                if not torch.isnan(sem_2d_loss):
                    loss = loss + sem_2d_loss
                    self.log("train/sem_2d_loss", sem_2d_loss.item(), on_step=True, prog_bar=False)

        if rgb_gs is not None:
            rgb_images = rearrange(results["rgb"], "(b v) c h w -> b v c h w", b=b, v=v)
            mse_loss = ((rgb_images - tg_images) ** 2).mean()
            lpips_loss = self.lpips.forward(
                rearrange(rgb_images, "b v c h w -> (b v) c h w"),
                rearrange(tg_images, "b v c h w -> (b v) c h w"),
                normalize=True,
            )
            rgb_loss = mse_loss * 10.0 + lpips_loss.mean()
            if not torch.isnan(rgb_loss):
                loss = loss + rgb_loss
                self.log("train/rgb_loss", rgb_loss.item(), on_step=True, prog_bar=False)

        # depth loss
        # rendered_depth = render_depth_cuda(
        #     rearrange(tg_extrinsics, "b v i j -> (b v) i j"),
        #     rearrange(tg_intrinsics, "b v i j -> (b v) i j"),
        #     repeat(self.near.to(device), "c -> (n c)", n=b * v),
        #     repeat(self.far.to(device), "c -> (n c)", n=b * v),
        #     (self.render_h, self.render_w),
        #     sem_gs.means,
        #     sem_gs.covariances,
        #     sem_gs.opacities,
        #     view_num=v,
        # )
        # rendered_depth = rearrange(rendered_depth, "(b v) c h w -> b v h w c", b=b, v=v).squeeze(-1)
        if self.model_cfg.offset_sup:
            off_loss = 0.0
            if self.model_cfg.depth_em:
                ori_depths = depth_maps['level_0']
            else:
                ori_depths = batch['depths_']['level_0'][:, :-1]
            if rgb_gs is not None and offset_dict['rgb'] is not None:
                rgb_t_loss = self.target_depth_loss(rgb_gs.means, ori_depths, batch['intrinsics'],
                                                    batch['poses'],
                                                    offset_dict['rgb'])
                self.log("train/rgb_t_loss", rgb_t_loss.item(), on_step=True, prog_bar=False)

                off_loss = off_loss + rgb_t_loss
            if sem_gs is not None and offset_dict['sem'] is not None:
                sem_t_loss = self.target_depth_loss(sem_gs.means, ori_depths, batch['intrinsics'],
                                                    batch['poses'],
                                                    offset_dict['sem'])
                self.log("train/sem_t_loss", sem_t_loss.item(), on_step=True, prog_bar=False)
                off_loss = off_loss + sem_t_loss
            if not torch.isnan(off_loss):
                loss = loss + 0.2 * off_loss

        if self.model_cfg.depth_em:
            depth_estimation_loss = self.depth_loss(depth_maps, batch['depths_'])
            if not torch.isnan(depth_estimation_loss):
                loss = loss + depth_estimation_loss
                self.log("train/depth_loss", depth_estimation_loss.item(), on_step=True, prog_bar=False)

        # losses = self.all_gather(loss)
        # if any(torch.isnan(loss) for loss in losses):
        #     print("skip nan loss")
        #     return None

        if torch.isnan(loss):
            print("Nan loss encountered, skipping batch...")
            return None
        else:
            self.log("train/loss", loss.item(), on_step=True, prog_bar=True, sync_dist=True)
            self.log("train/lr", self.optimizers().param_groups[0]["lr"], on_step=True, prog_bar=False)

        return loss

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            begin = time.time()
            device = batch['imgs'].device
            sem_gs, rgb_gs, semfeat_2d, sem_2d_logits, depth_maps, offset_dict = self.encoder(batch)

            # for i in range(ori_depths[0].shape[0]):
            #     r_dep = (ori_depths[0, i] / 6.0).clip(0, 1).cpu().numpy()
            #     r_dep = (r_dep * 255.0).astype(np.uint8)
            #     cv2.imwrite(f'./example_{i}.png', r_dep)

            scene_name = batch['scene_name'][0]
            if scene_name == 'scene0067_00':
                x = 1
                rimgs = batch['imgs'][0, :8]
                rc2ws = batch['poses'][0, :8]
                rintrinsics = batch['intrinsics'][0, :8]
                rdepths = batch['depths'][0, :8]
                points_world, rgbs = image_to_pointcloud(rimgs, rdepths, rc2ws, rintrinsics)

                pcd_o3d = o3d.geometry.PointCloud()
                pcd_o3d.points = o3d.utility.Vector3dVector(points_world.cpu().numpy())
                pcd_o3d.colors = o3d.utility.Vector3dVector(rgbs.cpu().numpy())
                o3d.io.write_point_cloud(f"{scene_name}.ply", pcd_o3d)
                #
                # gs_points = rgb_gs.means[0]
                # rgb_offset = offset_dict['rgb'][0]
                # offset_norm = torch.norm(rgb_offset, dim=1)  # (N,)
                # rgb_offset_mask = offset_norm > 0.005  # (N,) 布尔张量
                # # non-offset
                # points1 = gs_points[~rgb_offset_mask]
                # pcd_o3d1 = o3d.geometry.PointCloud()
                # pcd_o3d1.points = o3d.utility.Vector3dVector(points1.cpu().numpy())
                # colors1 = rgbs[~rgb_offset_mask]
                # pcd_o3d1.colors = o3d.utility.Vector3dVector(colors1.cpu().numpy())
                # o3d.io.write_point_cloud(f"{scene_name}_non.ply", pcd_o3d1)
                #
                # points2 = gs_points[rgb_offset_mask]
                # pcd_o3d2 = o3d.geometry.PointCloud()
                # pcd_o3d2.points = o3d.utility.Vector3dVector(points2.cpu().numpy())
                # o3d.io.write_point_cloud(f"{scene_name}_off.ply", pcd_o3d2)

            # rgb_offset_mask = offset_dict['rgb']
            # points = rgb_gs.means[0]
            # import matplotlib.pyplot as plt
            # import matplotlib
            # from mpl_toolkits.mplot3d import Axes3D
            # matplotlib.use('Agg')
            # fig = plt.figure()
            # ax = fig.add_subplot(111, projection='3d')
            # points = points[::1000].cpu().numpy()
            # rgb_offset_mask = rgb_offset_mask[0][::1000].cpu().numpy()
            # inds = np.nonzero(rgb_offset_mask)[0]
            # colors = ['blue' for _ in range(len(points))]
            # for k in inds:
            #     colors[k] = 'green'
            # ax.set_zlim(-1.5, 1.5)
            # ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, cmap='viridis', marker='o')
            # plt.savefig('offset_points.png')

            tg_intrinsics = batch['intrinsics'][:, -1:]
            tg_extrinsics = batch['poses'][:, -1:]
            tg_labels = batch['labels'][:, -1:]
            tg_images = batch['imgs'][:, -1:]
            b, v, _, _ = tg_intrinsics.shape

            results = {}
            for gs in (sem_gs, rgb_gs):
                if gs is not None:
                    render_images = render_cuda(
                        extrinsics=rearrange(tg_extrinsics, "b v i j -> (b v) i j"),
                        intrinsics=rearrange(tg_intrinsics, "b v i j -> (b v) i j"),
                        near=repeat(self.near.to(device), "c -> (n c)", n=b * v),
                        far=repeat(self.far.to(device), "c -> (n c)", n=b * v),
                        image_shape=(self.render_h, self.render_w),
                        background_color=repeat(self.background_color.to(device), "c -> (b v) (n c)", b=b, v=v,
                                                n=self.model_cfg.nb_class),
                        gaussian_means=gs.means,
                        gaussian_covariances=gs.covariances,
                        gaussian_opacities=gs.opacities,
                        gaussian_colors=gs.colors,
                        view_num=v,
                        use_sh=True if gs.name == "rgb" else False
                    )
                    results[gs.name] = render_images
            if self.mode == 'test':
                if f"time" not in self.val_step_outputs:
                    self.val_step_outputs[f"time"] = []
                self.val_step_outputs[f"time"].append(time.time() - begin)
            if sem_gs is not None:
                if f"miou" not in self.val_step_outputs:
                    self.val_step_outputs[f"miou"] = []
                if f"acc" not in self.val_step_outputs:
                    self.val_step_outputs[f"acc"] = []
                if f"class_acc" not in self.val_step_outputs:
                    self.val_step_outputs[f"class_acc"] = []

                rendered_sem_image = rearrange(results["semantic"], "b c h w  -> b h w c")
                semantic_pred = torch.argmax(rendered_sem_image, dim=-1)

                semantic_pred = rearrange(self.all_gather(semantic_pred), "g b h w  -> (g b) h w")
                tg_labels = rearrange(self.all_gather(tg_labels), "g b v h w -> (g b v) h w")

                for i in range(semantic_pred.shape[0]):
                    iou_score = self.miou(
                        true_labels=tg_labels[i].ravel(),
                        predicted_labels=semantic_pred[i].ravel(),
                    )
                    miou = iou_score["miou"]
                    acc = iou_score["total_accuracy"]
                    class_acc = iou_score["class_average_accuracy"]

                    self.val_step_outputs[f"miou"].append(miou.item())
                    self.val_step_outputs[f"acc"].append(acc.item())
                    self.val_step_outputs[f"class_acc"].append(class_acc.item())

                if self.mode == 'test':
                    # visual
                    semantic_logits_img = self.lable_color_map[semantic_pred.cpu()]
                    semantic_gt_img = self.lable_color_map[tg_labels.cpu()]

                    os.makedirs(os.path.join(f'{self.output_dir}/{self.global_step:08d}'), exist_ok=True)
                    for bi in range(semantic_logits_img.shape[0]):
                        save_image(semantic_gt_img[bi].permute(2, 0, 1),
                                   f"{self.output_dir}/{self.global_step:08d}/{self.wr_cntr:02d}_semantic_gt.png", )
                        save_image(semantic_logits_img[bi].permute(2, 0, 1),
                                   f"{self.output_dir}/{self.global_step:08d}/{self.wr_cntr:02d}_semantic.png", )
                        self.wr_cntr_s += 1

            if rgb_gs is not None:
                if f"psnr" not in self.val_step_outputs:
                    self.val_step_outputs[f"psnr"] = []
                if f"ssim" not in self.val_step_outputs:
                    self.val_step_outputs[f"ssim"] = []
                if f"lpips" not in self.val_step_outputs:
                    self.val_step_outputs[f"lpips"] = []

                depth_target = batch["depths"][:, -1:]
                mask = depth_target > 0
                mask = rearrange(mask, "b v h w -> (b v) h w")
                tg_images = rearrange(tg_images, "b v c h w -> (b v) c h w")

                rendered_images = rearrange(self.all_gather(results["rgb"]), "g b c h w -> (g b) c h w")
                mask = rearrange(self.all_gather(mask), "g b h w -> (g b) h w")
                tg_images = rearrange(self.all_gather(tg_images), "g b c h w -> (g b) c h w")

                for i in range(tg_images.shape[0]):
                    img_gt_masked = tg_images[i] * mask[i][None]
                    rendered_rgb_masked = rendered_images[i] * mask[i][None]

                    lpips = self.lpips.forward(rendered_rgb_masked[None], img_gt_masked[None],
                                               normalize=True).mean().item()

                    img_err_abs = (rendered_rgb_masked - img_gt_masked).abs().cpu()
                    psnr = mse2psnr(torch.mean(img_err_abs[:, mask[i].cpu()] ** 2)).item()
                    ssim = SSIM(
                        rendered_rgb_masked.permute(1, 2, 0).cpu().numpy(),
                        img_gt_masked.permute(1, 2, 0).cpu().numpy(),
                        data_range=1,
                        channel_axis=2,
                    )

                    self.val_step_outputs[f"psnr"].append(psnr)
                    self.val_step_outputs[f"ssim"].append(ssim)
                    self.val_step_outputs[f"lpips"].append(lpips)

                if self.mode == 'test':
                    os.makedirs(os.path.join(f'{self.output_dir}/{self.global_step:08d}'), exist_ok=True)
                    for bi in range(tg_images.shape[0]):
                        save_image(tg_images[bi],
                                   f"{self.output_dir}/{self.global_step:08d}/{self.wr_cntr:02d}_rgb_gt.png", )
                        save_image(rendered_images[bi],
                                   f"{self.output_dir}/{self.global_step:08d}/{self.wr_cntr:02d}_rgb.png", )
                        self.wr_cntr += 1

    def on_validation_epoch_end(self):
        saved_scores = {}
        if self.mode == 'train' and self.global_step > 1000:
            self.trainer.save_checkpoint(f'{self.output_dir}/checkpoints/latest.ckpt')
        if self.global_rank == 0:
            print(f"validation results : ")
        for metric_name, metric_scores in self.val_step_outputs.items():
            if metric_name == 'time':
                metric_scores = metric_scores[10:]
            avg_score = sum(metric_scores) / len(metric_scores)
            saved_scores[metric_name] = avg_score
            if self.global_rank == 0:
                print(f"val/{metric_name} = {avg_score}")
            if self.mode == 'train':
                self.log(f"val/{metric_name}", avg_score, prog_bar=False, sync_dist=True)

        self.val_step_outputs.clear()
        self.wr_cntr = 0

    def configure_optimizers(self):

        optimizer = optim.Adam(self.parameters(), lr=self.optimizer_cfg.lr)
        if self.optimizer_cfg.cosine_lr:
            warm_up = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, self.optimizer_cfg.lr,
                self.trainer.max_steps,
                pct_start=0.02,
                cycle_momentum=False,
                anneal_strategy='cos',
                final_div_factor=20
            )
        else:
            warm_up_steps = self.optimizer_cfg.warm_up_steps
            warm_up = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                1 / warm_up_steps,
                1,
                total_iters=warm_up_steps,
            )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": warm_up,
                "interval": "step",
                "frequency": 1,
            },
        }
