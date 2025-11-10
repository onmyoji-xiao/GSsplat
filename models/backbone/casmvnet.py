import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from inplace_abn import InPlaceABN
from kornia.utils import create_meshgrid


def homo_warp(src_feat, proj_mat, depth_values, src_grid=None, pad=0):
    if src_grid == None:
        B, C, H, W = src_feat.shape
        device = src_feat.device

        if pad > 0:
            H_pad, W_pad = H + pad * 2, W + pad * 2
        else:
            H_pad, W_pad = H, W

        if depth_values.dim() != 4:
            depth_values = depth_values[..., None, None].repeat(1, 1, H_pad, W_pad)
        D = depth_values.shape[1]

        R = proj_mat[:, :, :3]  # (B, 3, 3)
        T = proj_mat[:, :, 3:]  # (B, 3, 1)
        # create grid from the ref frame
        ref_grid = create_meshgrid(
            H_pad, W_pad, normalized_coordinates=False, device=device
        )  # (1, H, W, 2)
        if pad > 0:
            ref_grid -= pad

        ref_grid = ref_grid.permute(0, 3, 1, 2)  # (1, 2, H, W)
        ref_grid = ref_grid.reshape(1, 2, W_pad * H_pad)  # (1, 2, H*W)
        ref_grid = ref_grid.expand(B, -1, -1)  # (B, 2, H*W)
        ref_grid = torch.cat(
            (ref_grid, torch.ones_like(ref_grid[:, :1])), 1
        )  # (B, 3, H*W)
        ref_grid_d = ref_grid.repeat(1, 1, D)  # (B, 3, D*H*W)
        src_grid_d = R @ ref_grid_d + T / depth_values.reshape(B, 1, D * W_pad * H_pad)
        # del ref_grid_d, ref_grid, proj_mat, R, T, depth_values  # release (GPU) memory

        src_grid = (
                src_grid_d[:, :2] / src_grid_d[:, 2:]
        )  # divide by depth (B, 2, D*H*W)
        # del src_grid_d
        src_grid[:, 0] = src_grid[:, 0] / ((W - 1) / 2) - 1  # scale to -1~1
        src_grid[:, 1] = src_grid[:, 1] / ((H - 1) / 2) - 1  # scale to -1~1
        src_grid = src_grid.permute(0, 2, 1)  # (B, D*H*W, 2)
        src_grid = src_grid.view(B, D, W_pad, H_pad, 2)
        # Important, if the grid value too small or too large cause NAN in grid_sample
        margin = 1e-1
        src_grid = torch.clamp(src_grid, -1.0 - margin, 1.0 + margin)

    B, D, W_pad, H_pad = src_grid.shape[:4]
    warped_src_feat = F.grid_sample(
        src_feat,
        src_grid.view(B, D, W_pad * H_pad, 2),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )  # (B, C, D, H*W)
    warped_src_feat = warped_src_feat.view(B, -1, D, H_pad, W_pad)
    # src_grid = src_grid.view(B, 1, D, H_pad, W_pad, 2)
    if torch.isnan(warped_src_feat).any():
        print("nan in warped_src_feat")
    return warped_src_feat, src_grid


def get_depth_values(current_depth, n_depths, depth_interval):
    depth_min = torch.clamp_min(current_depth - n_depths / 2 * depth_interval, 1e-3)
    depth_values = (
            depth_min
            + depth_interval
            * torch.arange(
        0, n_depths, device=current_depth.device, dtype=current_depth.dtype
    )[None, :, None, None]
    )
    return depth_values


class ConvBnReLU(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            pad=1,
            norm_act=InPlaceABN,
    ):
        super(ConvBnReLU, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=pad,
            bias=False,
        )
        self.bn = norm_act(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))


class ConvBnReLU3D(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            pad=1,
            norm_act=InPlaceABN,
    ):
        super(ConvBnReLU3D, self).__init__()
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=pad,
            bias=False,
        )
        self.bn = norm_act(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))


class FeatureNet(nn.Module):
    def __init__(self):
        super(FeatureNet, self).__init__()

        self.toplayer = nn.Conv2d(128, 64, 1)
        self.lat1 = nn.Conv2d(64, 64, 1)
        self.lat0 = nn.Conv2d(32, 64, 1)

        # to reduce channel size of the outputs from FPN
        self.smooth1 = nn.Conv2d(64, 32, 3, padding=1)
        self.smooth0 = nn.Conv2d(64, 16, 3, padding=1)

    def _upsample_add(self, x, y):
        return F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True) + y

    def forward(self, x):
        # x: (B, 3, H, W)
        x0, x1, x2 = x[0], x[1], x[2]
        feat2 = self.toplayer(x2)  # (B, 32, H//4, W//4)
        feat1 = self._upsample_add(feat2, self.lat1(x1))  # (B, 32, H//2, W//2)
        feat0 = self._upsample_add(feat1, self.lat0(x0))  # (B, 32, H, W)

        # reduce output channels
        feat1 = self.smooth1(feat1)  # (B, 16, H//2, W//2)
        feat0 = self.smooth0(feat0)  # (B, 8, H, W)

        feats = {"level_0": feat0, "level_1": feat1, "level_2": feat2}

        return feats


class CostRegNet(nn.Module):
    def __init__(self, in_channels, norm_act=InPlaceABN):
        super(CostRegNet, self).__init__()
        self.conv0 = ConvBnReLU3D(in_channels, 8, norm_act=norm_act)

        self.conv1 = ConvBnReLU3D(8, 16, stride=2, norm_act=norm_act)
        self.conv2 = ConvBnReLU3D(16, 16, norm_act=norm_act)

        self.conv3 = ConvBnReLU3D(16, 32, stride=2, norm_act=norm_act)
        self.conv4 = ConvBnReLU3D(32, 32, norm_act=norm_act)

        self.conv5 = ConvBnReLU3D(32, 64, stride=2, norm_act=norm_act)
        self.conv6 = ConvBnReLU3D(64, 64, norm_act=norm_act)

        self.conv7 = nn.Sequential(
            nn.ConvTranspose3d(
                64, 32, 3, padding=1, output_padding=1, stride=2, bias=False
            ),
            norm_act(32),
        )

        self.conv9 = nn.Sequential(
            nn.ConvTranspose3d(
                32, 16, 3, padding=1, output_padding=1, stride=2, bias=False
            ),
            norm_act(16),
        )

        self.conv11 = nn.Sequential(
            nn.ConvTranspose3d(
                16, 8, 3, padding=1, output_padding=1, stride=2, bias=False
            ),
            norm_act(8),
        )

        self.br1 = ConvBnReLU3D(8, 8, norm_act=norm_act)
        self.br2 = ConvBnReLU3D(8, 8, norm_act=norm_act)

        self.prob = nn.Conv3d(8, 1, 3, stride=1, padding=1)

    def forward(self, x):
        if x.shape[-2] % 8 != 0 or x.shape[-1] % 8 != 0:
            pad_h = 8 * (x.shape[-2] // 8 + 1) - x.shape[-2]
            pad_w = 8 * (x.shape[-1] // 8 + 1) - x.shape[-1]
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="constant", value=0)
        else:
            pad_h = 0
            pad_w = 0

        conv0 = self.conv0(x)
        conv2 = self.conv2(self.conv1(conv0))
        conv4 = self.conv4(self.conv3(conv2))

        x = self.conv6(self.conv5(conv4))
        x = conv4 + self.conv7(x)
        del conv4
        x = conv2 + self.conv9(x)
        del conv2
        x = conv0 + self.conv11(x)
        del conv0
        ####################
        # x1 = self.br1(x)
        # with torch.enable_grad():
        #     x2 = self.br2(x)
        x1 = self.br1(x)
        x2 = self.br2(x)
        ####################
        p = self.prob(x1)

        if pad_h > 0 or pad_w > 0:
            x2 = x2[..., :-pad_h, :-pad_w]
            p = p[..., :-pad_h, :-pad_w]

        return x2, p

