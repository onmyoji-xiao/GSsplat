import time

import torch
from torch import nn, Tensor
from einops import rearrange
from .resnet2D import ResnetBlock2D
from .attnblock2D import AttnDownBlock2D


class BackboneMultiview(torch.nn.Module):
    """docstring for BackboneMultiview."""

    def __init__(
            self,
            feature_channels=128,
            cnn_num_blocks=5,
            att_num_blocks=3,
            resnet_groups=32,
            num_attention_heads=8,
    ):
        super(BackboneMultiview, self).__init__()
        self.feature_ch = feature_channels

        hidden_chs = [32, 64, 96, 64, 128, 64]

        self.conv_in = nn.Sequential(
            nn.Conv2d(3, hidden_chs[0], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(hidden_chs[0]),
            nn.ReLU(inplace=True)
        )

        self.cnn_blocks = []
        for i in range(cnn_num_blocks):
            self.cnn_blocks.append(ResnetBlock2D(
                in_channels=hidden_chs[i],
                out_channels=hidden_chs[i + 1],
                groups=resnet_groups,
                dropout=0.3,
                down=True if (i == 2 or i == 3) else False
            ))
        self.cnn_blocks = nn.ModuleList(self.cnn_blocks)

        self.attn_blocks = []
        for i in range(att_num_blocks):
            self.attn_blocks.append(AttnDownBlock2D(
                in_channels=hidden_chs[-1] if i == 0 else self.feature_ch,
                out_channels=self.feature_ch,
                num_layers=2,
                add_downsample=False,
                resnet_act_fn="silu",
                resnet_groups=resnet_groups,
                dropout=0.3,
                attention_head_dim=self.feature_ch // num_attention_heads
            ))
        self.attn_blocks = nn.ModuleList(self.attn_blocks)

    def normalize_images(self, images):
        '''Normalize image to match the pretrained GMFlow backbone.
            images: (B, N_Views, C, H, W)
        '''
        shape = [*[1] * (images.dim() - 3), 3, 1, 1]
        mean = torch.tensor([0.485, 0.456, 0.406]).reshape(*shape).to(images.device)
        std = torch.tensor([0.229, 0.224, 0.225]).reshape(*shape).to(images.device)

        return (images - mean) / std

    def forward(
            self,
            images,
    ):
        ''' images: (B, N_Views, C, H, W), range [0, 1] '''
        # resolution low to high
        images = self.normalize_images(images)
        x = rearrange(images, "b v c h w -> (b v) c h w")
        x = self.conv_in(x)
        cnn_features = None
        for i, block in enumerate(self.cnn_blocks):
            x = block(x)
            if i == 2:
                cnn_features = x
        for block in self.attn_blocks:
            x = block(x)

        return x, cnn_features


class BackboneMultiview_test(torch.nn.Module):
    """docstring for BackboneMultiview."""

    def __init__(
            self,
            feature_channels=128,
            cnn_num_blocks=5,
            att_num_blocks=3,
            resnet_groups=32,
            num_attention_heads=8,
    ):
        super(BackboneMultiview_test, self).__init__()
        self.feature_ch = feature_channels

        hidden_chs = [32, 64, 96, 64, 128, 64]

        self.conv_in = nn.Sequential(
            nn.Conv2d(3, hidden_chs[0], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(hidden_chs[0]),
            nn.ReLU(inplace=True)
        )

        self.cnn_blocks = []
        for i in range(cnn_num_blocks):
            self.cnn_blocks.append(ResnetBlock2D(
                in_channels=hidden_chs[i],
                out_channels=hidden_chs[i + 1],
                groups=resnet_groups,
                down=True if (i == 0 or i == 3) else False
            ))
        self.cnn_blocks = nn.ModuleList(self.cnn_blocks)

        self.attn_blocks = []
        for i in range(att_num_blocks):
            self.attn_blocks.append(AttnDownBlock2D(
                in_channels=hidden_chs[-1] if i == 0 else self.feature_ch,
                out_channels=self.feature_ch,
                num_layers=2,
                add_downsample=False,
                resnet_act_fn="silu",
                resnet_groups=resnet_groups,
                attention_head_dim=self.feature_ch // num_attention_heads
            ))
        self.attn_blocks = nn.ModuleList(self.attn_blocks)

    def normalize_images(self, images):
        '''Normalize image to match the pretrained GMFlow backbone.
            images: (B, N_Views, C, H, W)
        '''
        shape = [*[1] * (images.dim() - 3), 3, 1, 1]
        mean = torch.tensor([0.485, 0.456, 0.406]).reshape(*shape).to(images.device)
        std = torch.tensor([0.229, 0.224, 0.225]).reshape(*shape).to(images.device)

        return (images - mean) / std

    def forward(
            self,
            images,
    ):
        ''' images: (B, N_Views, C, H, W), range [0, 1] '''
        # resolution low to high
        images = self.normalize_images(images)
        x = rearrange(images, "b v c h w -> (b v) c h w")
        x = self.conv_in(x)
        for i, block in enumerate(self.cnn_blocks):
            x = block(x)
            if i == 0:
                cnn_features = x
        # cnn_features = x
        for block in self.attn_blocks:
            x = block(x)

        return x, cnn_features
