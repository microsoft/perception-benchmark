# Copyright (c) EPFL VILAB.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# Based on timm, DeiT, DINO, MoCo-v3, BEiT, MAE-priv and MAE code bases
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# https://github.com/facebookresearch/moco-v3
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/BUPT-PRIV/MAE-priv
# https://github.com/facebookresearch/mae
# --------------------------------------------------------

# extended to spatial temporal
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class MaskedCrossEntropyLoss(nn.Module):
    """Cross-entropy loss with masking
    :param patch_size: Patch size
    :param stride: Stride of task / modality
    :param label_smoothing: Amount of smoothing in the loss (default is 0.0)
    """

    def __init__(
        self,
        spatial_patch_size: int = 16,
        temporal_patch_size: int = 4,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.spatial_patch_size = spatial_patch_size
        self.temporal_patch_size = temporal_patch_size
        self.label_smoothing = label_smoothing

    def forward(self, input, target, mask=None):

        loss = F.cross_entropy(
            input, target, reduction="none", label_smoothing=self.label_smoothing
        )

        if mask is not None:
            if mask.sum() == 0:
                return torch.tensor(0).to(loss.device)

            T, H, W = input.shape[-3:]
            nt, nh, nw = (
                T // self.temporal_patch_size,
                H // self.spatial_patch_size,
                W // self.spatial_patch_size,
            )
            # Resize mask and upsample
            mask = rearrange(mask, "b (nt nh nw) -> b nt nh nw", nt=nt, nh=nh, nw=nw)
            mask = F.interpolate(
                mask.unsqueeze(1).float(), size=(T, H, W), mode="nearest"
            ).squeeze(1)
            loss = loss * mask  # b, nt, nh, nw
            # Compute mean per sample
            loss = loss.flatten(start_dim=1).sum(dim=1) / mask.flatten(start_dim=1).sum(
                dim=1
            )
            loss = loss.nanmean()  # Account for zero masks
        else:
            loss = loss.mean()  # If this is ever nan, we want it to stop training

        return loss


class MaskedMSELoss(nn.Module):
    """L1 loss with masking
    :param patch_size: Patch size
    :param norm_pix: Normalized pixel loss
    """

    def __init__(
        self, temporal_patch_size: int = 4, spatial_patch_size: int = 16, norm_pix=False
    ):
        super().__init__()
        self.temporal_patch_size = temporal_patch_size
        self.spatial_patch_size = spatial_patch_size
        self.norm_pix = norm_pix

    def patchify(self, imgs, nt, nh, nw):
        x = rearrange(
            imgs,
            "b c (nt tp) (nh sp) (nw sp) -> b (nt nh nw) (tp sp sp c)",
            nt=nt,
            nh=nh,
            nw=nw,
            tp=self.temporal_patch_size,
            sp=self.spatial_patch_size,
        )
        return x

    def unpatchify(self, x, nt, nh, nw):
        imgs = rearrange(
            x,
            "b (nt nh nw) (tp sp sp c) -> b c (nt tp) (nh sp) (nw sp)",
            nt=nt,
            nh=nh,
            nw=nw,
            tp=self.temporal_patch_size,
            sp=self.spatial_patch_size,
        )
        return imgs

    def forward(self, input, target, mask=None):

        T, H, W = input.shape[-3:]
        nt, nh, nw = (
            T // self.temporal_patch_size,
            H // self.spatial_patch_size,
            W // self.spatial_patch_size,
        )

        if self.norm_pix:
            target = self.patchify(target, nt, nh, nw)
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            eps = 1e-6
            target = (target - mean) / torch.sqrt(var + eps)
            target = self.unpatchify(target, nt, nh, nw)

        loss = F.mse_loss(input, target, reduction="none")

        if mask is not None:
            if mask.sum() == 0:
                return torch.tensor(0).to(loss.device)

            # Resize mask and upsample
            mask = rearrange(mask, "b (nt nh nw) -> b nt nh nw", nh=nh, nw=nw)
            mask = F.interpolate(
                mask.unsqueeze(1).float(), size=(T, H, W), mode="nearest"
            ).squeeze(1)
            loss = loss.mean(dim=1)  # B, C, T, H, W -> B, T, H, W
            loss = loss * mask
            # Compute mean per sample
            loss = loss.flatten(start_dim=1).sum(dim=1) / mask.flatten(start_dim=1).sum(
                dim=1
            )
            loss = loss.nanmean()  # Account for zero masks
        else:
            loss = loss.mean()  # If this is ever nan, we want it to stop training

        return loss


class MaskedL1Loss(nn.Module):
    """L1 loss with masking
    :param patch_size: Patch size
    :param norm_pix: Normalized pixel loss
    """

    def __init__(
        self,
        temporal_patch_size: int = 4,
        spatial_patch_size: int = 16,
        norm_pix=False,
    ):
        super().__init__()
        self.temporal_patch_size = temporal_patch_size
        self.spatial_patch_size = spatial_patch_size
        self.norm_pix = norm_pix

    def patchify(self, imgs, nt, nh, nw):
        x = rearrange(
            imgs,
            "b c (nt tp) (nh sp) (nw sp) -> b (nt nh nw) (tp sp sp c)",
            nt=nt,
            nh=nh,
            nw=nw,
            tp=self.temporal_patch_size,
            sp=self.spatial_patch_size,
        )
        return x

    def unpatchify(self, x, nt, nh, nw):
        imgs = rearrange(
            x,
            "b (nt nh nw) (tp sp sp c) -> b c (nt tp) (nh sp) (nw sp)",
            nt=nt,
            nh=nh,
            nw=nw,
            ts=self.temporal_patch_size,
            p2=self.spatial_patch_size,
        )
        return imgs

    def forward(self, input, target, mask=None):

        T, H, W = input.shape[-3:]
        nt, nh, nw = (
            T // self.temporal_patch_size,
            H // self.spatial_patch_size,
            W // self.spatial_patch_size,
        )

        if self.norm_pix:
            target = self.patchify(target, nt, nh, nw)
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            eps = 1e-6
            target = (target - mean) / torch.sqrt(var + eps)
            target = self.unpatchify(target, nt, nh, nw)

        loss = F.l1_loss(input, target, reduction="none")

        if mask is not None:
            if mask.sum() == 0:
                return torch.tensor(0).to(loss.device)

            # Resize mask and upsample
            mask = rearrange(mask, "b (nt nh nw) -> b nt nh nw", nt=nt, nh=nh, nw=nw)
            mask = F.interpolate(
                mask.unsqueeze(1).float(), size=(T, H, W), mode="nearest"
            ).squeeze(1)
            loss = loss.mean(dim=1)  # B, C, T, H, W -> B, T, H, W
            loss = loss * mask
            # Compute mean per sample
            loss = loss.flatten(start_dim=1).sum(dim=1) / mask.flatten(start_dim=1).sum(
                dim=1
            )
            loss = loss.nanmean()  # Account for zero masks
        else:
            loss = loss.mean()  # If this is ever nan, we want it to stop training

        return loss
