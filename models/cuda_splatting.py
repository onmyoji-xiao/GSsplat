from math import isqrt
from typing import Literal

import torch
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
from einops import einsum, rearrange, repeat
from jaxtyping import Float
from torch import Tensor


def get_fov(intrinsics: Float[Tensor, "batch 3 3"]) -> Float[Tensor, "batch 2"]:
    intrinsics_inv = intrinsics.inverse()

    def process_vector(vector):
        vector = torch.tensor(vector, dtype=torch.float32, device=intrinsics.device)
        vector = einsum(intrinsics_inv, vector, "b i j, j -> b i")  # 复制了n个
        return vector / vector.norm(dim=-1, keepdim=True)

    left = process_vector([0, 0.5, 1])
    right = process_vector([1, 0.5, 1])
    top = process_vector([0.5, 0, 1])
    bottom = process_vector([0.5, 1, 1])
    fov_x = (left * right).sum(dim=-1).acos()
    fov_y = (top * bottom).sum(dim=-1).acos()
    return torch.stack((fov_x, fov_y), dim=-1)


def get_projection_matrix(
        near: Float[Tensor, " "],
        far: Float[Tensor, " "],
        fov_x: Float[Tensor, " batch"],
        fov_y: Float[Tensor, " batch"],
) -> Float[Tensor, "batch 4 4"]:
    """Maps points in the viewing frustum to (-1, 1) on the X/Y axes and (0, 1) on the Z
    axis. Differs from the OpenGL version in that Z doesn't have range (-1, 1) after
    transformation and that Z is flipped.
    """
    tan_fov_x = (0.5 * fov_x).tan()
    tan_fov_y = (0.5 * fov_y).tan()

    top = tan_fov_y * near
    bottom = -top
    right = tan_fov_x * near
    left = -right

    (b,) = fov_x.shape
    result = torch.zeros((b, 4, 4), dtype=torch.float32, device=near.device)
    result[:, 0, 0] = 2 * near / (right - left)
    result[:, 1, 1] = 2 * near / (top - bottom)
    result[:, 0, 2] = (right + left) / (right - left)
    result[:, 1, 2] = (top + bottom) / (top - bottom)
    result[:, 3, 2] = 1
    result[:, 2, 2] = far / (far - near)
    result[:, 2, 3] = -(far * near) / (far - near)
    return result


def render_cuda(
        extrinsics: Float[Tensor, "batch 4 4"],
        intrinsics: Float[Tensor, "batch 3 3"],
        near: Float[Tensor, " batch"],
        far: Float[Tensor, " batch"],
        image_shape: tuple[int, int],
        background_color: Float[Tensor, "batch 3"],
        gaussian_means: Float[Tensor, "batch gaussian 3"],
        gaussian_covariances: Float[Tensor, "batch gaussian 3 3"],
        gaussian_opacities: Float[Tensor, "batch gaussian"],
        gaussian_colors: Float[Tensor, "batch gaussian d_sh"],
        view_num: int,
        backdepths: Float[Tensor, "batch gaussian"] = None,
        scale_invariant: bool = True,
        use_sh: bool = True,

) -> Float[Tensor, "batch 3 height width"]:
    # Make sure everything is in a range where numerical issues don't appear.
    if scale_invariant:
        scale = 1 / near
        extrinsics = extrinsics.clone()
        extrinsics[..., :3, 3] = extrinsics[..., :3, 3] * scale[:, None]
        gaussian_covariances = gaussian_covariances * (scale[::view_num, None, None, None] ** 2)
        gaussian_means = gaussian_means * scale[::view_num, None, None]
        near = near * scale
        far = far * scale

    if use_sh:
        gaussian_sh_coefficients = rearrange(gaussian_colors, "b g (c d) -> b g c d", c=3)
        _, _, _, n = gaussian_sh_coefficients.shape
        degree = isqrt(n) - 1
        shs = rearrange(gaussian_sh_coefficients, "b g xyz n -> b g n xyz").contiguous()

    b, _, _ = extrinsics.shape
    h, w = image_shape

    # fov_x, fov_y = get_fov(intrinsics).unbind(dim=-1)
    focal_length_x, focal_length_y = intrinsics[:, 0, 0], intrinsics[:, 1, 1]
    fov_y = 2 * (h / (2 * focal_length_y)).atan()
    fov_x = 2 * (w / (2 * focal_length_x)).atan()

    tan_fov_x = (0.5 * fov_x).tan()
    tan_fov_y = (0.5 * fov_y).tan()

    projection_matrix = get_projection_matrix(near, far, fov_x, fov_y)
    projection_matrix = rearrange(projection_matrix, "b i j -> b j i")
    view_matrix = rearrange(extrinsics.inverse(), "b i j -> b j i")
    full_projection = view_matrix @ projection_matrix

    all_images = []
    all_radii = []
    for i in range(b):
        gs_i = i // view_num
        mean_gradients = torch.zeros_like(gaussian_means[gs_i], requires_grad=True)
        try:
            mean_gradients.retain_grad()
        except Exception:
            pass

        settings = GaussianRasterizationSettings(
            image_height=h,  # int
            image_width=w,  #
            tanfovx=tan_fov_x[i].item(),  # (1)
            tanfovy=tan_fov_y[i].item(),  # (1)
            bg=background_color[i],  # (3)
            scale_modifier=1.0,
            viewmatrix=view_matrix[i],  # (4,4)
            projmatrix=full_projection[i],  # (4,4)
            sh_degree=degree if use_sh else 0,
            campos=extrinsics[i, :3, 3],  # (3)
            prefiltered=False,  # This matches the original usage.
            debug=False,
            num_channels=3 if use_sh else gaussian_colors.shape[-1]
        )
        rasterizer = GaussianRasterizer(settings)

        row, col = torch.triu_indices(3, 3)

        if backdepths is not None:
            mask = backdepths[gs_i] > 0
            image, radii = rasterizer(
                means3D=gaussian_means[gs_i][mask],
                means2D=mean_gradients[mask],
                shs=shs[gs_i][mask] if use_sh else None,
                colors_precomp=None if use_sh else gaussian_colors[gs_i][mask],
                opacities=gaussian_opacities[gs_i, ..., None][mask],
                cov3D_precomp=gaussian_covariances[gs_i, :, row, col][mask],
            )
        else:
            image, radii = rasterizer(
                means3D=gaussian_means[gs_i],
                means2D=mean_gradients,
                shs=shs[gs_i] if use_sh else None,
                colors_precomp=None if use_sh else gaussian_colors[gs_i],
                opacities=gaussian_opacities[gs_i, ..., None],
                cov3D_precomp=gaussian_covariances[gs_i, :, row, col],
            )
        all_images.append(image)
        all_radii.append(radii)
    return torch.stack(all_images)


def homogenize_points(
        points: Float[Tensor, "*batch dim"],
) -> Float[Tensor, "*batch dim+1"]:
    """Convert batched points (xyz) to (xyz1)."""
    return torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)


def render_depth_cuda(
        extrinsics: Float[Tensor, "batch 4 4"],
        intrinsics: Float[Tensor, "batch 3 3"],
        near: Float[Tensor, " batch"],
        far: Float[Tensor, " batch"],
        image_shape: tuple[int, int],
        gaussian_means: Float[Tensor, "batch gaussian 3"],
        gaussian_covariances: Float[Tensor, "batch gaussian 3 3"],
        gaussian_opacities: Float[Tensor, "batch gaussian"],
        view_num: int,
        scale_invariant: bool = True,
        off_mask: Float[Tensor, "batch gaussian"] = None
) -> Float[Tensor, "batch height width"]:
    # Specify colors according to Gaussian depths.

    camera_space_gaussians = einsum(
        extrinsics.inverse(), homogenize_points(repeat(gaussian_means, "b g j-> (b v) g j", v=view_num)),
        "b i j, b g j -> b g i"
    )
    fake_color = camera_space_gaussians[..., 2]

    # Render using depth as color.
    b, _ = fake_color.shape

    background_color = torch.zeros((b, 3), dtype=fake_color.dtype, device=fake_color.device)
    fake_color = repeat(fake_color, "b g -> b g ()")
    if scale_invariant:
        scale = 1 / near
        extrinsics = extrinsics.clone()
        extrinsics[..., :3, 3] = extrinsics[..., :3, 3] * scale[:, None]
        gaussian_covariances = gaussian_covariances * (scale[::view_num, None, None, None] ** 2)
        gaussian_means = gaussian_means * scale[::view_num, None, None]
        near = near * scale
        far = far * scale

    b, _, _ = extrinsics.shape
    h, w = image_shape

    # fov_x, fov_y = get_fov(intrinsics).unbind(dim=-1)
    focal_length_x, focal_length_y = intrinsics[:, 0, 0], intrinsics[:, 1, 1]
    fov_y = 2 * (h / (2 * focal_length_y)).atan()
    fov_x = 2 * (w / (2 * focal_length_x)).atan()

    tan_fov_x = (0.5 * fov_x).tan()
    tan_fov_y = (0.5 * fov_y).tan()

    projection_matrix = get_projection_matrix(near, far, fov_x, fov_y)
    projection_matrix = rearrange(projection_matrix, "b i j -> b j i")
    view_matrix = rearrange(extrinsics.inverse(), "b i j -> b j i")
    full_projection = view_matrix @ projection_matrix

    all_images = []
    all_radii = []
    for i in range(b):
        gs_i = i // view_num

        settings = GaussianRasterizationSettings(
            image_height=h,  # int
            image_width=w,  #
            tanfovx=tan_fov_x[i].item(),  # (1)
            tanfovy=tan_fov_y[i].item(),  # (1)
            bg=background_color[i],  # (3)
            scale_modifier=1.0,
            viewmatrix=view_matrix[i],  # (4,4)
            projmatrix=full_projection[i],  # (4,4)
            sh_degree=0,
            campos=extrinsics[i, :3, 3],  # (3)
            prefiltered=False,  # This matches the original usage.
            debug=False,
            num_channels=fake_color.shape[-1]
        )
        rasterizer = GaussianRasterizer(settings)

        row, col = torch.triu_indices(3, 3)
        if off_mask is not None:
            re_mask = ~off_mask[gs_i]
            mean_gradients = torch.zeros_like(gaussian_means[gs_i][re_mask], requires_grad=True)
            try:
                mean_gradients.retain_grad()
            except Exception:
                pass
            image, radii = rasterizer(
                means3D=gaussian_means[gs_i][re_mask],
                means2D=mean_gradients,
                shs=None,
                colors_precomp=fake_color[i][re_mask],
                opacities=gaussian_opacities[gs_i, ..., None][re_mask],
                cov3D_precomp=gaussian_covariances[gs_i, :, row, col][re_mask],
            )
        else:
            mean_gradients = torch.zeros_like(gaussian_means[gs_i], requires_grad=True)
            try:
                mean_gradients.retain_grad()
            except Exception:
                pass
            image, radii = rasterizer(
                means3D=gaussian_means[gs_i],
                means2D=mean_gradients,
                shs=None,
                colors_precomp=fake_color[i],
                opacities=gaussian_opacities[gs_i, ..., None],
                cov3D_precomp=gaussian_covariances[gs_i, :, row, col],
            )
        all_images.append(image)
        all_radii.append(radii)

    return torch.stack(all_images)
