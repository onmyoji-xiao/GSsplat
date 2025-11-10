import torch.nn as nn
import torch
from typing import Any, Dict, Optional
from models.backbone.activations import get_activation
from models.backbone.downsampling import Downsample2D
from models.backbone.upsampling import Upsample2D


class ResnetBlock2D(nn.Module):
    r"""
    A Resnet block.

    Parameters:
        in_channels (`int`): The number of channels in the input.
        out_channels (`int`, *optional*, default to be `None`):
            The number of output channels for the first conv2d layer. If None, same as `in_channels`.
        dropout (`float`, *optional*, defaults to `0.0`): The dropout probability to use.
        groups (`int`, *optional*, default to `32`): The number of groups to use for the first normalization layer.
        groups_out (`int`, *optional*, default to None):
            The number of groups to use for the second normalization layer. if set to None, same as `groups`.
        eps (`float`, *optional*, defaults to `1e-6`): The epsilon to use for the normalization.
        non_linearity (`str`, *optional*, default to `"swish"`): the activation function to use.
        kernel (`torch.Tensor`, optional, default to None): FIR filter, see
            [`~models.resnet.FirUpsample2D`] and [`~models.resnet.FirDownsample2D`].
        output_scale_factor (`float`, *optional*, default to be `1.0`): the scale factor to use for the output.
        use_in_shortcut (`bool`, *optional*, default to `True`):
            If `True`, add a 1x1 nn.conv2d layer for skip-connection.
        conv_shortcut_bias (`bool`, *optional*, default to `True`):  If `True`, adds a learnable bias to the
            `conv_shortcut` output.
        conv_2d_out_channels (`int`, *optional*, default to `None`): the number of channels in the output.
            If None, same as `out_channels`.
    """

    def __init__(
            self,
            *,
            in_channels: int,
            out_channels: Optional[int] = None,
            conv_shortcut: bool = False,
            dropout: float = 0.0,
            groups: int = 32,
            groups_out: Optional[int] = None,
            pre_norm: bool = True,
            eps: float = 1e-6,
            non_linearity: str = "swish",
            output_scale_factor: float = 1.0,
            use_in_shortcut: Optional[bool] = None,
            conv_shortcut_bias: bool = True,
            conv_2d_out_channels: Optional[int] = None,
            up: bool = False,
            down: bool = False,
    ):
        super().__init__()

        self.pre_norm = pre_norm
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.output_scale_factor = output_scale_factor
        self.up = up
        self.down = down

        if groups_out is None:
            groups_out = groups

        self.norm1 = torch.nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)
        # self.norm1 = nn.InstanceNorm2d(in_channels)

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.norm2 = torch.nn.GroupNorm(num_groups=groups_out, num_channels=out_channels, eps=eps, affine=True)
        # self.norm2 = nn.InstanceNorm2d(out_channels)

        self.dropout = torch.nn.Dropout(dropout)
        conv_2d_out_channels = conv_2d_out_channels or out_channels
        self.conv2 = nn.Conv2d(out_channels, conv_2d_out_channels, kernel_size=3, stride=1, padding=1)

        self.nonlinearity = get_activation(non_linearity)

        self.upsample = self.downsample = None

        if self.up:
            self.upsample = Upsample2D(in_channels, use_conv=True)
        elif self.down:
            self.downsample = Downsample2D(in_channels, use_conv=True, padding=1, name="op")

        self.use_in_shortcut = self.in_channels != conv_2d_out_channels if use_in_shortcut is None else use_in_shortcut

        self.conv_shortcut = None
        if self.use_in_shortcut:
            self.conv_shortcut = nn.Conv2d(
                in_channels,
                conv_2d_out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=conv_shortcut_bias,
            )

    def forward(self, input_tensor: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        hidden_states = input_tensor

        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)

        if self.upsample is not None:
            # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
            if hidden_states.shape[0] >= 64:
                input_tensor = input_tensor.contiguous()
                hidden_states = hidden_states.contiguous()
            input_tensor = self.upsample(input_tensor)
            hidden_states = self.upsample(hidden_states)
        elif self.downsample is not None:
            input_tensor = self.downsample(input_tensor)
            hidden_states = self.downsample(hidden_states)

        hidden_states = self.conv1(hidden_states)

        hidden_states = self.norm2(hidden_states)

        hidden_states = self.nonlinearity(hidden_states)

        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor)

        output_tensor = (input_tensor + hidden_states) / self.output_scale_factor

        return output_tensor


if __name__ == "__main__":
    model = ResnetBlock2D(
        in_channels=128,
        out_channels=128,
        eps=1e-5,
        groups=32,
        dropout=0,
        non_linearity="silu",
        output_scale_factor=1.0,
        pre_norm=True,
        down=False)

    # x = torch.rand((2, 128, 60, 80))
    # y = torch.rand((2, 128, 60, 80))
    # device = torch.device("cuda:0")
    # x = x.to(device)
    # y = y.to(device)
    # model.to(device)
    # from einops import rearrange
    # x1 = model(x)
    # y1 = model(y)
    # x = rearrange(x, "b c h w -> b c (h w) ")
    # y = rearrange(y, "b c h w -> b c (h w) ")
    # # x1 = x1.view((2, 128, 4800))
    # # y1 = y1.view((2, 128, 4800))
    # h = x1 + y1
