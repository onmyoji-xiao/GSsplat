import torch
from einops import rearrange
from pytorch_lightning import LightningModule
from torch import nn, optim
from utils.utils import lable_color_map
from skimage.metrics import structural_similarity as SSIM

import torch.nn.functional as F
from models.backbone.UNet import UNet
from models.backbone.casmvnet import InPlaceABN, CostRegNet, homo_warp, get_depth_values, checkpoint
import torch_scatter
from .midas_loss import ScaleAndShiftInvariantLoss


class CasMVSNet(nn.Module):
    def __init__(self, num_groups=8, norm_act=InPlaceABN, levels=3, use_depth=False):
        super(CasMVSNet, self).__init__()
        self.levels = levels  # 3 depth levels
        self.n_depths = [16, 32, 48]
        self.interval_ratios = [1, 2, 4]
        self.use_depth = use_depth

        self.G = num_groups  # number of groups in groupwise correlation
        self.feature = UNet(3)

        for l in range(self.levels):
            if l == self.levels - 1 and self.use_depth:
                cost_reg_l = CostRegNet(self.G + 1, norm_act)
            else:
                cost_reg_l = CostRegNet(self.G, norm_act)

            setattr(self, f"cost_reg_{l}", cost_reg_l)

    def build_cost_volumes(self, feats, affine_mats, affine_mats_inv, depth_values, idx, spikes):
        B, V, C, H, W = feats.shape
        D = depth_values.shape[1]

        ref_feats, src_feats = feats[:, idx[0]], feats[:, idx[1:]]
        src_feats = src_feats.permute(1, 0, 2, 3, 4)  # (V-1, B, C, h, w)

        affine_mats_inv = affine_mats_inv[:, idx[0]]
        affine_mats = affine_mats[:, idx[1:]]
        affine_mats = affine_mats.permute(1, 0, 2, 3)  # (V-1, B, 3, 4)

        ref_volume = ref_feats.unsqueeze(2).repeat(1, 1, D, 1, 1)  # (B, C, D, h, w)

        ref_volume = ref_volume.view(B, self.G, C // self.G, *ref_volume.shape[-3:])
        volume_sum = 0

        for i in range(len(idx) - 1):
            proj_mat = (affine_mats[i].double() @ affine_mats_inv.double()).float()[:, :3]  # shape (1,3,4)
            warped_volume, grid = homo_warp(src_feats[i], proj_mat, depth_values)

            warped_volume = warped_volume.view_as(ref_volume)
            volume_sum = volume_sum + warped_volume  # (B, G, C//G, D, h, w)
            if torch.isnan(volume_sum).sum() > 0:
                print("nan in volume_sum")

        volume = (volume_sum * ref_volume).mean(dim=2) / (V - 1)

        if spikes is None:
            output = volume
        else:
            output = torch.cat([volume, spikes], dim=1)

        return output

    def create_neural_volume(
            self,
            feats,
            affine_mats,
            affine_mats_inv,
            idx,
            init_depth_min,
            depth_interval,
            gt_depths,
    ):
        if feats["level_0"].shape[-1] >= 800:
            hres_input = True
        else:
            hres_input = False

        B, V = affine_mats.shape[:2]

        v_feat = {}
        depth_maps = {}
        depth_values = {}
        for l in reversed(range(self.levels)):  # (2, 1, 0)
            feats_l = feats[f"level_{l}"]  # (B*V, C, h, w)
            feats_l = feats_l.view(B, V, *feats_l.shape[1:])  # (B, V, C, h, w)
            h, w = feats_l.shape[-2:]
            depth_interval_l = depth_interval * self.interval_ratios[l]
            D = self.n_depths[l]
            if l == self.levels - 1:  # coarsest level
                depth_values_l = init_depth_min + depth_interval_l * torch.arange(
                    0, D, device=feats_l.device, dtype=feats_l.dtype
                )  # (D)
                depth_values_l = depth_values_l[None, :, None, None].expand(
                    B, -1, h, w
                )
                if self.use_depth:
                    gt_mask = gt_depths > 0
                    sp_idx_float = (gt_mask * (gt_depths - init_depth_min) / (depth_interval_l))[:, :, None]
                    spikes = (torch.arange(D).view(1, 1, -1, 1, 1).to(gt_mask.device) == sp_idx_float.floor().long()
                              ) * (1 - sp_idx_float.frac())
                    spikes = spikes + (
                            torch.arange(D).view(1, 1, -1, 1, 1).to(gt_mask.device)
                            == sp_idx_float.ceil().long()
                    ) * (sp_idx_float.frac())
                    spikes = (spikes * gt_mask[:, :, None]).float()
            else:
                depth_lm1 = depth_l.detach()  # the depth of previous level
                depth_lm1 = F.interpolate(
                    depth_lm1, scale_factor=2, mode="bilinear", align_corners=True
                )  # (B, 1, h, w)
                depth_values_l = get_depth_values(depth_lm1, D, depth_interval_l)

            affine_mats_l = affine_mats[..., l]
            affine_mats_inv_l = affine_mats_inv[..., l]

            if l == self.levels - 1 and self.use_depth:
                spikes_ = spikes
            else:
                spikes_ = None

            if hres_input:
                v_feat_l = checkpoint(
                    self.build_cost_volumes,
                    feats_l,
                    affine_mats_l,
                    affine_mats_inv_l,
                    depth_values_l,
                    idx,
                    spikes_,
                    preserve_rng_state=False,
                )
            else:
                v_feat_l = self.build_cost_volumes(
                    feats_l,
                    affine_mats_l,
                    affine_mats_inv_l,
                    depth_values_l,
                    idx,
                    spikes_,
                )

            cost_reg_l = getattr(self, f"cost_reg_{l}")
            v_feat_l_, depth_prob = cost_reg_l(v_feat_l)  # (B, 1, D, h, w)

            depth_l = (F.softmax(depth_prob, dim=2) * depth_values_l[:, None]).sum(
                dim=2
            )
            # v_feat_l have 8 nan values, go debug build_cost_volumes
            if torch.isnan(v_feat_l_).sum() > 0:
                print("nan in v_feat_l_")
            v_feat[f"level_{l}"] = v_feat_l_
            depth_maps[f"level_{l}"] = depth_l
            depth_values[f"level_{l}"] = depth_values_l

        return v_feat, depth_maps, depth_values

    def forward(
            self, imgs, affine_mats, affine_mats_inv, near_far, closest_idxs, gt_depths=None
    ):
        B, V, _, H, W = imgs.shape

        ## Feature Pyramid
        feats = self.feature(
            imgs.reshape(B * V, 3, H, W))  # (B*V, 8, H, W), (B*V, 16, H//2, W//2), (B*V, 32, H//4, W//4)
        feats_fpn = feats[f"feature"]

        feats_vol = {"level_0": [], "level_1": [], "level_2": []}
        depth_map = {"level_0": [], "level_1": [], "level_2": []}
        depth_values = {"level_0": [], "level_1": [], "level_2": []}
        ## Create cost volumes for each view
        for i in range(0, V):
            permuted_idx = closest_idxs[0, i].clone().detach().to(feats['level_0'].device)

            init_depth_min = near_far[0, i, 0]
            depth_interval = (
                    (near_far[0, i, 1] - near_far[0, i, 0])
                    / self.n_depths[-1]
                    / self.interval_ratios[-1]
            )

            v_feat, d_map, d_values = self.create_neural_volume(
                feats,
                affine_mats,
                affine_mats_inv,
                idx=permuted_idx,
                init_depth_min=init_depth_min,
                depth_interval=depth_interval,
                gt_depths=gt_depths[:, i: i + 1],
            )

            for l in range(3):
                feats_vol[f"level_{l}"].append(v_feat[f"level_{l}"])
                depth_map[f"level_{l}"].append(d_map[f"level_{l}"])
                depth_values[f"level_{l}"].append(d_values[f"level_{l}"])

        for l in range(3):
            feats_vol[f"level_{l}"] = torch.stack(feats_vol[f"level_{l}"], dim=1)
            depth_map[f"level_{l}"] = torch.cat(depth_map[f"level_{l}"], dim=1)
            depth_values[f"level_{l}"] = torch.stack(depth_values[f"level_{l}"], dim=1)

        return feats_vol, feats_fpn, depth_map, depth_values


def depthimage_to_threeD(depth, intrinsic, pose, mask=None):
    device = depth.device
    h, w = depth.shape
    v, u = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing='ij')

    Z = depth
    X = (u - intrinsic[0, 2]) * Z / intrinsic[0, 0]
    Y = (v - intrinsic[1, 2]) * Z / intrinsic[1, 1]

    X = torch.ravel(X)
    Y = torch.ravel(Y)
    Z = torch.ravel(Z)
    if mask is not None:
        mask = mask.view(-1).to(bool)
        X = X[mask]
        Y = Y[mask]
        Z = Z[mask]
    position = torch.vstack((X, Y, Z, torch.ones(len(X), device=device)))
    position = torch.mm(pose, position)
    points = position[:3, :].transpose(0, 1)
    return points


def threeD_to_depthimage(points, intrinsics, poses, h, w):
    device = points.device
    images = []

    intrinsic = intrinsics[0]
    coords = poses.new(4, len(points))
    coords[:3, :] = torch.t(points)
    coords[3, :].fill_(1)

    # project world (coords) to camera
    world_to_cameras = torch.stack([torch.inverse(pose) for pose in poses])
    camera = torch.bmm(world_to_cameras, coords.repeat(len(world_to_cameras), 1, 1))

    # # project camera to image
    xys = torch.zeros_like(camera)
    xys[:, 0] = (camera[:, 0] * intrinsic[0, 0]) / camera[:, 2] + intrinsic[0, 2]
    xys[:, 1] = (camera[:, 1] * intrinsic[1, 1]) / camera[:, 2] + intrinsic[1, 2]
    xys = torch.round(xys[:, :2]).long()
    depth_vals = camera[:, 2]

    valid_masks = torch.ge(xys[:, 0], 0) * torch.ge(xys[:, 1], 0) * torch.lt(xys[:, 0], w) * torch.lt(xys[:, 1], h)

    for xy, dv, mask in zip(xys, depth_vals, valid_masks):
        wrap_depth_image = torch.zeros(h * w, device=device)
        valid_xy = xy[:, mask]
        valid_inds = valid_xy[1] * w + valid_xy[0]
        valid_image_z = dv[mask]
        depmask = valid_image_z > 0
        new_inds, mapping = torch.unique(valid_inds[depmask], return_inverse=True)
        new_values = torch_scatter.scatter(valid_image_z[depmask], mapping, dim=0, reduce='min')
        wrap_depth_image[new_inds] = new_values
        images.append(wrap_depth_image.view((h, w)))
    return torch.stack(images)


class ModelWrapper(LightningModule):
    def __init__(
            self,
            cfg,
            args,

    ):
        super().__init__()
        self.train_cfg = cfg.trainer
        self.optimizer_cfg = cfg.optimizer
        self.model_cfg = cfg.model
        self.data_cfg = cfg.data
        self.output_dir = args.output_dir
        self.output_path = f"./save/{args.dataset_name}_depth_estimation.pth"

        # Set up the model.
        self.depth_estimation = CasMVSNet()
        # self.depth_estimation.load_state_dict(
        #     torch.load('./save/replica_depth_estimation.pth', map_location='cpu'), strict=True)
        self.save_model = args.save_model
        self.mode = args.mode
        self.eval_cnt = 0
        self.val_step_outputs = {}
        self.wr_cntr = 0
        self.lable_color_map = torch.from_numpy(lable_color_map) / 255.0

    def target_depth_loss(self, depths, gt_depth, intrinsics, extrinsics):
        B, V, H, W = depths.shape

        self_loss = 0.0
        for bi in range(B):
            batch_pred_depths = depths[bi]
            batch_intrinsics = intrinsics[bi]
            batch_extrinsics = extrinsics[bi]

            all_points = []
            for k in range(V):
                points = depthimage_to_threeD(batch_pred_depths[k], batch_intrinsics[k], batch_extrinsics[k])
                all_points.append(points)
            all_points = torch.cat(all_points)
            if len(all_points) == 0:
                continue

            target_intrinsics = intrinsics[bi, -1:]
            target_extrinsics = extrinsics[bi, -1:]
            target_gt_depths = gt_depth[bi, -1:]
            target_pred_depths = threeD_to_depthimage(all_points, target_intrinsics, target_extrinsics, H, W)
            target_loss = F.smooth_l1_loss(target_gt_depths, target_pred_depths,
                                           reduction="mean")
            self_loss = self_loss + target_loss

        return self_loss / B

    def consistent_depth_loss(self, depths, intrinsics, extrinsics):
        B, V, H, W = depths.shape

        self_loss = 0.0
        for bi in range(B):
            batch_pred_depths = depths[bi]
            batch_intrinsics = intrinsics[bi]
            batch_extrinsics = extrinsics[bi]

            all_points = []
            for k in range(V):
                points = depthimage_to_threeD(batch_pred_depths[k], batch_intrinsics[k], batch_extrinsics[k])
                all_points.append(points)
            all_points = torch.cat(all_points)
            if len(all_points) == 0:
                continue
            intrinsics_i = batch_intrinsics[:V]
            extrinsics_i = batch_extrinsics[:V]
            pred_depths_i = batch_pred_depths[:]
            wrap_depths = threeD_to_depthimage(all_points, intrinsics_i, extrinsics_i, H, W)
            mask = wrap_depths > 0
            consist_loss = F.smooth_l1_loss(pred_depths_i[mask], wrap_depths[mask], reduction="mean")
            self_loss = self_loss + consist_loss

        return self_loss / B

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

                # if not self.data_cfg.use_depth:
                #     depth_pred_l = F.normalize(depth_pred_l, dim=1, p=1)
                #     depth_gt_l = F.normalize(depth_gt_l, dim=1, p=1)
                l1_loss = l1_loss + F.smooth_l1_loss(depth_pred_l[mask_l], depth_gt_l[mask_l]) * 2 ** (1 - l)
                if l == 0:
                    depth_dx = depth_pred_l.diff(dim=-1)
                    depth_dy = depth_pred_l.diff(dim=-2)
                    smooth_loss = smooth_loss + depth_dx.abs().mean() + depth_dy.abs().mean()
        else:
            mask_l = gt_depths > 0
            l1_loss = F.smooth_l1_loss(depths[mask_l], gt_depths[mask_l], reduction="mean")
            depth_dx = depths.diff(dim=-1)
            depth_dy = depths.diff(dim=-2)
            smooth_loss = depth_dx.abs().mean() + depth_dy.abs().mean()

        return 0.5 * smooth_loss + l1_loss

    def training_step(self, batch, batch_idx):
        # Run the model.
        ref_images = batch['imgs'][:, :-1]
        feats_vol, feats_fpn, depth_maps, depth_values = self.depth_estimation(
            ref_images,
            affine_mats=batch["affine_mats"][:, :-1],
            affine_mats_inv=batch["affine_mats_inv"][:, :-1],
            near_far=batch["near_fars"][:, :-1],
            closest_idxs=batch["closest_idxs"],
            gt_depths=batch["depths_aug"][:, :-1],
        )
        if isinstance(depth_maps, dict):
            gt_depths = batch['depths_']
        else:
            gt_depths = batch['depths'][:, :-1]

        # mask_fg = batch["fmasks"][:, :-1]
        # scale_loss = self.scale_loss(
        #     rearrange(depth_maps['level_0'], "b v h w -> (b v) h w"),
        #     rearrange(batch['depths'][:, :-1], "b v h w -> (b v) h w"),
        #     rearrange(mask_fg, "b v h w -> (b v) h w")
        # )
        depth_loss = self.depth_loss(depth_maps, gt_depths)
        loss = depth_loss
        # depth_maps_ = depth_maps['level_0']
        # consistent_loss = self.consistent_depth_loss(depth_maps_, batch['intrinsics'], batch['poses'])
        # if self.data_cfg.use_depth:
        #     loss = depth_loss

        losses = self.all_gather(loss)
        if any(torch.isnan(loss) for loss in losses):
            print("skip nan loss")
            return None  # skip training step

        if torch.isnan(loss):
            if self.global_rank == 0:
                print("Nan semantic loss encountered, skipping batch...")
            return torch.nan_to_num(loss, nan=0.0)
        else:
            self.log("train/loss", loss.item(), prog_bar=True, logger=True)
            self.log("train/lr", self.optimizers().param_groups[0]["lr"], on_step=True, prog_bar=False)

        return loss

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            ref_images = batch['imgs'][:, :-1]
            feats_vol, feats_fpn, depth_maps, depth_values = self.depth_estimation(
                ref_images,
                affine_mats=batch["affine_mats"][:, :-1],
                affine_mats_inv=batch["affine_mats_inv"][:, :-1],
                near_far=batch["near_fars"][:, :-1],
                closest_idxs=batch["closest_idxs"],
                gt_depths=batch["depths_aug"][:, :-1],
            )

            depth_target = batch['depths'][:, :-1]
            ref_depths = depth_maps['level_0']

            if f"closs" not in self.val_step_outputs:
                self.val_step_outputs[f"closs"] = []
            consistent_loss = self.consistent_depth_loss(ref_depths, batch['intrinsics'], batch['poses'])
            self.val_step_outputs[f"closs"].append(consistent_loss)

            if f"ssim" not in self.val_step_outputs:
                self.val_step_outputs[f"ssim"] = []
            depth_target = rearrange(depth_target, "b v h w -> (b v) h w")
            depth_target = depth_target.unsqueeze(1).repeat(1, 3, 1, 1)
            ref_depths = rearrange(ref_depths, "b v h w -> (b v) h w")
            ref_depths = ref_depths.unsqueeze(1).repeat(1, 3, 1, 1)

            depth_target = rearrange(self.all_gather(depth_target), "g b c h w -> (g b) c h w")
            ref_depths = rearrange(self.all_gather(ref_depths), "g b c h w -> (g b) c h w")

            for i in range(depth_target.shape[0]):
                ssim = SSIM(
                    depth_target[i].permute(1, 2, 0).cpu().numpy(),
                    ref_depths[i].permute(1, 2, 0).cpu().numpy(),
                    data_range=1,
                    channel_axis=2,
                )
                self.val_step_outputs[f"ssim"].append(ssim)

    def on_validation_epoch_end(self):
        if self.save_model:
            print(f'save to {self.output_path}')
            with torch.no_grad():
                torch.save(self.depth_estimation.state_dict(), self.output_path)
        saved_scores = {}
        if self.mode == 'train' and self.global_step > 1000:
            self.trainer.save_checkpoint(f'{self.output_dir}/checkpoints/latest.ckpt')
        if self.global_rank == 0:
            print(f"validation results : ")
        for metric_name, metric_scores in self.val_step_outputs.items():
            avg_score = sum(metric_scores) / len(metric_scores)
            saved_scores[metric_name] = avg_score
            if self.global_rank == 0:
                print(f"val/{metric_name} = {avg_score}")
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
