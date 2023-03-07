""" Visualization tools for mutliMAE"""
import math
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.transforms import ToTensor
from einops import rearrange
import matplotlib.pyplot as plt
from moviepy.editor import VideoClip, ImageSequenceClip
from moviepy.video.io.bindings import mplfig_to_npimage
from mpl_toolkits.axes_grid1 import ImageGrid
from .tMultiMAE.utils.data_constants import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
)


def get_masked_video(
    video: torch.Tensor,
    mask: torch.Tensor,
    num_frames: int = 8,
    image_size: int = 224,
    temporal_patch_size: int = 4,
    spatial_patch_size: int = 16,
    mask_value: float = 0.0,
):
    video_token = rearrange(
        video,
        "b c (nt pt) (nh ph) (nw pw) -> b (nt nh nw) (c pt ph pw)",
        nt=num_frames // temporal_patch_size,
        pt=temporal_patch_size,
        ph=spatial_patch_size,
        pw=spatial_patch_size,
        nh=image_size // spatial_patch_size,
        nw=image_size // spatial_patch_size,
    )
    video_token[mask != 0] = mask_value
    video = rearrange(
        video_token,
        "b (nt nh nw) (c pt ph pw) -> b c (nt pt) (nh ph) (nw pw)",
        nt=num_frames // temporal_patch_size,
        pt=temporal_patch_size,
        ph=spatial_patch_size,
        pw=spatial_patch_size,
        nh=image_size // spatial_patch_size,
        nw=image_size // spatial_patch_size,
    )
    return video


### not dealing with semantic images for now
# def plot_semseg_gt(input_dict, ax=None, image_size=224):
#     """Simplied plotting of semseg images
#
#     Args:
#         input_dict (_type_): _description_
#         ax (_type_, optional): _description_. Defaults to None.
#         image_size (int, optional): _description_. Defaults to 224.
#     """
#     semseg = F.interpolate(
#         input_dict["semseg"].cpu().detach().float().unsqueeze(1),
#         size=(image_size, image_size),
#         mode="nearest",
#     ).long()[0, 0]
#
#     ax.imshow(semseg)
#
#
# def plot_semseg_gt_masked(input_dict, mask, ax=None, mask_value=1.0, image_size=224):
#     semseg = F.interpolate(
#         input_dict["semseg"].cpu().detach().float().unsqueeze(1),
#         size=(image_size, image_size),
#         mode="nearest",
#     ).long()
#     masked_img = get_masked_image(
#         semseg.float() / 255.0,
#         mask,
#         image_size=image_size,
#         patch_size=16,
#         mask_value=mask_value,
#     )
#     masked_img = masked_img[0].permute(1, 2, 0)
#     ax.imshow(masked_img)
#
#
# def plot_semseg_pred_masked(semseg_preds, semseg_gt, mask, ax=None, image_size=224):
#     semseg = get_pred_with_input(
#         semseg_gt.unsqueeze(1),
#         semseg_preds.argmax(1).unsqueeze(1),
#         mask,
#         image_size=image_size // 4,
#         patch_size=4,
#     )
#     semseg = (
#         F.interpolate(semseg.float(), size=(image_size, image_size), mode="nearest")[
#             0, 0
#         ]
#         .long()
#         .cpu()
#     )
#     ax.imshow(semseg)


def denormalize(
    video: torch.Tensor, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD
):
    # b c t h w -> b t c h w
    video = video.transpose(1, 2).contiguous()
    video = TF.normalize(
        video.clone(),
        mean=[-m / s for m, s in zip(mean, std)],
        std=[1 / s for s in std],
    )
    # b t c h w -> b c t h w
    video = video.transpose(1, 2)
    return video


def get_pred_with_input(
    gt: torch.Tensor,
    pred: torch.Tensor,
    mask: torch.Tensor,
    num_frames: int = 8,
    image_size: int = 224,
    temporal_patch_size: int = 4,
    spatial_patch_size: int = 16,
):
    gt_token = rearrange(
        gt,
        "b c (nt pt) (nh ph) (nw pw) -> b (nt nh nw) (c pt ph pw)",
        pt=temporal_patch_size,
        ph=spatial_patch_size,
        pw=spatial_patch_size,
        nt=num_frames // temporal_patch_size,
        nh=image_size // spatial_patch_size,
        nw=image_size // spatial_patch_size,
    )
    pred_token = rearrange(
        pred,
        "b c (nt pt) (nh ph) (nw pw) -> b (nt nh nw) (c pt ph pw)",
        pt=temporal_patch_size,
        ph=spatial_patch_size,
        pw=spatial_patch_size,
        nt=num_frames // temporal_patch_size,
        nh=image_size // spatial_patch_size,
        nw=image_size // spatial_patch_size,
    )
    pred_token[mask == 0] = gt_token[mask == 0]
    video = rearrange(
        pred_token,
        "b (nt nh nw) (c pt ph pw) -> b c (nt pt) (nh ph) (nw pw)",
        pt=temporal_patch_size,
        ph=spatial_patch_size,
        pw=spatial_patch_size,
        nt=num_frames // temporal_patch_size,
        nh=image_size // spatial_patch_size,
        nw=image_size // spatial_patch_size,
    )
    return video


def visualize_predictions(
    input_dict,
    preds,
    masks,
    image_size: int = 224,
    num_frames: int = 8,
    temporal_patch_size: int = 4,
    spatial_patch_size: int = 16,
    fps: int = 24,
):
    masked_rgb_video = (
        get_masked_video(
            denormalize(input_dict["rgb"]),
            masks["rgb"],
            num_frames=num_frames,
            image_size=image_size,
            temporal_patch_size=temporal_patch_size,
            spatial_patch_size=spatial_patch_size,
            mask_value=1.0,
        )[0]
        .permute(1, 2, 3, 0)
        .detach()
        .cpu()
    )
    # masked_rgb_video of size t x h x w x c
    masked_depth_video = (
        get_masked_video(
            input_dict["depth"],
            masks["depth"],
            num_frames=num_frames,
            image_size=image_size,
            temporal_patch_size=temporal_patch_size,
            spatial_patch_size=spatial_patch_size,
            mask_value=np.nan,
        )[0, 0]
        .detach()
        .cpu()
    )
    # masked_depth_video of size t x h x w
    pred_rgb2_video = (
        get_pred_with_input(
            denormalize(input_dict["rgb"]),
            denormalize(preds["rgb"]).clamp(0, 1),
            masks["rgb"],
            num_frames=num_frames,
            image_size=image_size,
            temporal_patch_size=temporal_patch_size,
            spatial_patch_size=spatial_patch_size,
        )[0]
        .permute(1, 2, 3, 0)
        .detach()
        .cpu()
    )
    pred_depth2_video = (
        get_pred_with_input(
            input_dict["depth"],
            preds["depth"],
            masks["depth"],
            num_frames=num_frames,
            image_size=image_size,
            temporal_patch_size=temporal_patch_size,
            spatial_patch_size=spatial_patch_size,
        )[0, 0]
        .detach()
        .cpu()
    )

    fig = plt.figure(figsize=(10, 5))
    grid = ImageGrid(fig, 111, nrows_ncols=(2, 3), axes_pad=0)

    def make_frame(t):
        nth_frame = math.floor(t * fps)
        grid[0].imshow(masked_rgb_video[nth_frame])
        grid[1].imshow(pred_rgb2_video[nth_frame])
        grid[2].imshow(
            denormalize(input_dict["rgb"])[0]
            .permute(1, 2, 3, 0)[nth_frame]
            .detach()
            .cpu()
        )

        grid[3].imshow(masked_depth_video[nth_frame])
        grid[4].imshow(pred_depth2_video[nth_frame])
        grid[5].imshow(input_dict["depth"][0, 0][nth_frame].detach().cpu())

        for ax in grid:
            ax.set_xticks([])
            ax.set_yticks([])

        fontsize = 16
        grid[0].set_title("Masked inputs", fontsize=fontsize)
        grid[1].set_title("tMultiMAE predictions", fontsize=fontsize)
        grid[2].set_title("Original Reference", fontsize=fontsize)
        grid[0].set_ylabel("RGB", fontsize=fontsize)
        grid[3].set_ylabel("Depth", fontsize=fontsize)

        return mplfig_to_npimage(fig)

    animation = VideoClip(make_frame, duration=(1 / fps) * num_frames)
    return animation, fps
