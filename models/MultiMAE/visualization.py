""" Visualization tools for mutliMAE"""
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.transforms import ToTensor
from einops import rearrange
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from .MultiMAE.utils.data_constants import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
)


def get_masked_image(img, mask, image_size=224, patch_size=16, mask_value=0.0):
    img_token = rearrange(
        img,
        "b c (nh ph) (nw pw) -> b (nh nw) (c ph pw)",
        ph=patch_size,
        pw=patch_size,
        nh=image_size // patch_size,
        nw=image_size // patch_size,
    )
    img_token[mask != 0] = mask_value
    img = rearrange(
        img_token,
        "b (nh nw) (c ph pw) -> b c (nh ph) (nw pw)",
        ph=patch_size,
        pw=patch_size,
        nh=image_size // patch_size,
        nw=image_size // patch_size,
    )
    return img


def plot_semseg_gt(input_dict, ax=None, image_size=224):
    """Simplied plotting of semseg images

    Args:
        input_dict (_type_): _description_
        ax (_type_, optional): _description_. Defaults to None.
        image_size (int, optional): _description_. Defaults to 224.
    """
    semseg = F.interpolate(
        input_dict["semseg"].cpu().detach().float().unsqueeze(1),
        size=(image_size, image_size),
        mode="nearest",
    ).long()[0, 0]

    ax.imshow(semseg)


def plot_semseg_gt_masked(input_dict, mask, ax=None, mask_value=1.0, image_size=224):
    semseg = F.interpolate(
        input_dict["semseg"].cpu().detach().float().unsqueeze(1),
        size=(image_size, image_size),
        mode="nearest",
    ).long()
    masked_img = get_masked_image(
        semseg.float() / 255.0,
        mask,
        image_size=image_size,
        patch_size=16,
        mask_value=mask_value,
    )
    masked_img = masked_img[0].permute(1, 2, 0)
    ax.imshow(masked_img)


def plot_semseg_pred_masked(semseg_preds, semseg_gt, mask, ax=None, image_size=224):
    semseg = get_pred_with_input(
        semseg_gt.unsqueeze(1),
        semseg_preds.argmax(1).unsqueeze(1),
        mask,
        image_size=image_size // 4,
        patch_size=4,
    )
    semseg = (
        F.interpolate(semseg.float(), size=(image_size, image_size), mode="nearest")[
            0, 0
        ]
        .long()
        .cpu()
    )
    ax.imshow(semseg)


def denormalize(img, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD):
    return TF.normalize(
        img.clone(), mean=[-m / s for m, s in zip(mean, std)], std=[1 / s for s in std]
    )


def get_pred_with_input(gt, pred, mask, image_size=224, patch_size=16):
    gt_token = rearrange(
        gt,
        "b c (nh ph) (nw pw) -> b (nh nw) (c ph pw)",
        ph=patch_size,
        pw=patch_size,
        nh=image_size // patch_size,
        nw=image_size // patch_size,
    )
    pred_token = rearrange(
        pred,
        "b c (nh ph) (nw pw) -> b (nh nw) (c ph pw)",
        ph=patch_size,
        pw=patch_size,
        nh=image_size // patch_size,
        nw=image_size // patch_size,
    )
    pred_token[mask == 0] = gt_token[mask == 0]
    img = rearrange(
        pred_token,
        "b (nh nw) (c ph pw) -> b c (nh ph) (nw pw)",
        ph=patch_size,
        pw=patch_size,
        nh=image_size // patch_size,
        nw=image_size // patch_size,
    )
    return img


def plot_predictions(input_dict, preds, masks, image_size=224):
    masked_rgb = (
        get_masked_image(
            denormalize(input_dict["rgb"]),
            masks["rgb"],
            image_size=image_size,
            mask_value=1.0,
        )[0]
        .permute(1, 2, 0)
        .detach()
        .cpu()
    )
    masked_depth = (
        get_masked_image(
            input_dict["depth"],
            masks["depth"],
            image_size=image_size,
            mask_value=np.nan,
        )[0, 0]
        .detach()
        .cpu()
    )

    # todo: remove this?
    pred_rgb = denormalize(preds["rgb"])[0].permute(1, 2, 0).clamp(0, 1)
    pred_depth = preds["depth"][0, 0].detach().cpu()

    pred_rgb2 = (
        get_pred_with_input(
            denormalize(input_dict["rgb"]),
            denormalize(preds["rgb"]).clamp(0, 1),
            masks["rgb"],
            image_size=image_size,
        )[0]
        .permute(1, 2, 0)
        .detach()
        .cpu()
    )
    pred_depth2 = (
        get_pred_with_input(
            input_dict["depth"], preds["depth"], masks["depth"], image_size=image_size
        )[0, 0]
        .detach()
        .cpu()
    )

    fig = plt.figure(figsize=(10, 10))
    grid = ImageGrid(fig, 111, nrows_ncols=(3, 3), axes_pad=0)

    grid[0].imshow(masked_rgb)
    grid[1].imshow(pred_rgb2)
    grid[2].imshow(denormalize(input_dict["rgb"])[0].permute(1, 2, 0).detach().cpu())

    grid[3].imshow(masked_depth)
    grid[4].imshow(pred_depth2)
    grid[5].imshow(input_dict["depth"][0, 0].detach().cpu())

    # semseg plotting
    # plot_semseg_gt_masked(
    #     input_dict,
    #     masks["semseg"],
    #     grid[6],
    #     mask_value=1.0,
    #     image_size=image_size,
    # )
    # plot_semseg_pred_masked(
    #     preds["semseg"],
    #     input_dict["semseg"],
    #     masks["semseg"],
    #     grid[7],
    #     image_size=image_size,
    # )
    # plot_semseg_gt(input_dict, grid[8], image_size=image_size)

    for ax in grid:
        ax.set_xticks([])
        ax.set_yticks([])

    fontsize = 16
    grid[0].set_title("Masked inputs", fontsize=fontsize)
    grid[1].set_title("MultiMAE predictions", fontsize=fontsize)
    grid[2].set_title("Original Reference", fontsize=fontsize)
    grid[0].set_ylabel("RGB", fontsize=fontsize)
    grid[3].set_ylabel("Depth", fontsize=fontsize)

    # semseg plotting
    grid[6].set_ylabel("Semantic", fontsize=fontsize)
    # todo: more efficient ways to plot images in tensor board
    # buf = io.BytesIO()
    # plt.savefig(buf, format="png")
    # plt.close(fig)
    # buf.seek(0)

    # # decode the array into an image
    # # todo: use pillow-simd in future
    # image_grid = PIL.Image.open(buf)
    # image_grid = ToTensor()(image_grid)
    return fig
