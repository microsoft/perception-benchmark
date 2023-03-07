import sys
from typing import Dict
import math
from functools import partial
import torch
from torchvision.utils import save_image
from einops import rearrange
import pytorch_lightning as pl
from torchlightning_utils.lr_scheduler import LinearWarmupCosineAnnealingLR
from registry.registry import registry
from .MultiMAE.multimae import multimae
from .MultiMAE.multimae.criterion import (
    MaskedCrossEntropyLoss,
    MaskedL1Loss,
    MaskedMSELoss,
)
from .MultiMAE.utils.task_balancing import (
    NoWeightingStrategy,
    UncertaintyWeightingStrategy,
)
from .MultiMAE.multimae.input_adapters import PatchedInputAdapter, SemSegInputAdapter
from .MultiMAE.multimae.output_adapters import SpatialOutputAdapter
from .MultiMAE.utils import create_model
from .MultiMAE.utils.data_constants import COCO_SEMSEG_NUM_CLASSES

# todo: make plotting a callback
# todo: HIGH PRIORITY run large scale experiment on aml (determine batch size and etc)
# todo: val transform for input transform
# todo: support float16

# low priority
# todo: weight_decay scheduler? check paper
# todo: monitor gradient scale
# todo: go over the training code of multiMAE again

# default configuration for input and output
DOMAIN_CONF = {
    "rgb": {
        "channels": 3,
        "stride_level": 1,
        "input_adapter": partial(PatchedInputAdapter, num_channels=3),
        "output_adapter": partial(SpatialOutputAdapter, num_channels=3),
        "loss": MaskedMSELoss,
    },
    "depth": {
        "channels": 1,
        "stride_level": 1,
        "input_adapter": partial(PatchedInputAdapter, num_channels=1),
        "output_adapter": partial(SpatialOutputAdapter, num_channels=1),
        "loss": MaskedL1Loss,
    },
    "semseg": {
        "num_classes": 133,
        "stride_level": 4,
        "input_adapter": partial(
            SemSegInputAdapter,
            num_classes=COCO_SEMSEG_NUM_CLASSES,
            dim_class_emb=64,
            interpolate_class_emb=False,
        ),
        "output_adapter": partial(
            SpatialOutputAdapter, num_channels=COCO_SEMSEG_NUM_CLASSES
        ),
        "loss": partial(MaskedCrossEntropyLoss, label_smoothing=0.0),
    },
}


def make_model(model_config: dict):
    """Make a mutliMAE model based on model config and default settings in DOMAIN_CONF

    Args:
        model_config (dict): _description_

    Returns:
        MultiMAE model
    """
    input_adapters = {
        domain: DOMAIN_CONF[domain]["input_adapter"](
            stride_level=DOMAIN_CONF[domain]["stride_level"],
            patch_size_full=model_config["patch_size"],
        )
        for domain in model_config["in_domains"]
    }

    output_adapters = {
        domain: DOMAIN_CONF[domain]["output_adapter"](
            stride_level=DOMAIN_CONF[domain]["stride_level"],
            patch_size_full=model_config["patch_size"],
            dim_tokens=model_config["decoder_dim"],
            depth=model_config["decoder_depth"],
            num_heads=model_config["decoder_num_heads"],
            use_task_queries=model_config["decoder_use_task_queries"],
            task=domain,
            context_tasks=list(model_config["in_domains"]),
            use_xattn=model_config["decoder_use_xattn"],
        )
        for domain in model_config["out_domains"]
    }

    # Add normalized pixel output adapter if specified
    if model_config["extra_norm_pix_loss"]:
        output_adapters["norm_rgb"] = DOMAIN_CONF["rgb"]["output_adapter"](
            stride_level=DOMAIN_CONF["rgb"]["stride_level"],
            patch_size_full=model_config["patch_size"],
            dim_tokens=model_config["decoder_dim"],
            depth=model_config["decoder_depth"],
            num_heads=model_config["decoder_num_heads"],
            use_task_queries=model_config["decoder_use_task_queries"],
            task="rgb",
            context_tasks=list(model_config["in_domains"]),
            use_xattn=model_config["decoder_use_xattn"],
        )

    model = create_model(
        model_config["model_type"],
        input_adapters=input_adapters,
        output_adapters=output_adapters,
        num_global_tokens=model_config["num_global_tokens"],
        drop_path_rate=model_config["drop_path"],
    )
    return model


@registry.register_lightningmodule(name="MultiMAE")
class MultiMAEPretrain(pl.LightningModule):
    def __init__(
        self,
        model_config: dict,
        optimizer_config: dict,
        scheduler_config: dict,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(
            "model_config", "optimizer_config", "scheduler_config"
        )

        # create input/ output adapters
        self.model = make_model(model_config)
        # transform image to patches

        # define loss
        if model_config.get("task_balancer", "") == "uncertainty":
            self.loss_balancer = UncertaintyWeightingStrategy(
                tasks=model_config["out_domains"]
            )
        else:
            self.loss_balancer = NoWeightingStrategy()

        self.tasks_loss_fn = {
            domain: DOMAIN_CONF[domain]["loss"](
                patch_size=model_config["patch_size"],
                stride=DOMAIN_CONF[domain]["stride_level"],
            )
            for domain in model_config["out_domains"]
        }

        # Add normalized pixel loss if specified
        if model_config["extra_norm_pix_loss"]:
            self.tasks_loss_fn["norm_rgb"] = DOMAIN_CONF["rgb"]["loss"](
                patch_size=model_config["patch_size"],
                stride=DOMAIN_CONF["rgb"]["stride_level"],
                norm_pix=True,
            )

        self.sample_train_batch = None
        self.sample_val_batch = None


    def on_train_start(self):
        pass

    def prepare_batch(self, batch: Dict[str, torch.Tensor]) -> None:
        # batch = self.transform(batch)
        # Truncated depth standardization
        if self.hparams.model_config["standardize_depth"] and "depth" in batch:
            # Flatten depth and remove bottom and top 10% of values
            trunc_depth = torch.sort(
                rearrange(batch["depth"], "b c h w -> b (c h w)"), dim=1
            )[0]
            trunc_depth = trunc_depth[
                :, int(0.1 * trunc_depth.shape[1]) : int(0.9 * trunc_depth.shape[1])
            ]
            batch["depth"] = (
                batch["depth"] - trunc_depth.mean(dim=1)[:, None, None, None]
            ) / torch.sqrt(trunc_depth.var(dim=1)[:, None, None, None] + 1e-6)

        input_dict = {
            task: tensor
            for task, tensor in batch.items()
            if task in self.hparams.model_config["in_domains"]
        }
        return batch, input_dict

    def step(self, batch):
        batch, input_dict = self.prepare_batch(batch)
        # forwarding batch to model
        preds, masks = self.model(
            input_dict,
            num_encoded_tokens=self.hparams.model_config["num_encoded_tokens"],
            alphas=self.hparams.model_config["alphas"],
            sample_tasks_uniformly=self.hparams.model_config["sample_tasks_uniformly"],
            fp32_output_adapters=self.hparams.model_config["fp32_output_adapters"],
        )
        if self.hparams.model_config["extra_norm_pix_loss"]:
            batch["norm_rgb"] = batch["rgb"]
            masks["norm_rgb"] = masks.get("rgb", None)

        task_losses = {}
        for task in preds:
            target = batch[task]

            if self.hparams.model_config["loss_on_unmasked"]:
                task_losses[task] = self.tasks_loss_fn[task](
                    preds[task].float(), target
                )
            else:
                task_losses[task] = self.tasks_loss_fn[task](
                    preds[task].float(), target, mask=masks.get(task, None)
                )

        weighted_task_losses = self.loss_balancer(task_losses)
        loss = sum(weighted_task_losses.values())

        # values for logging
        loss_value = sum(task_losses.values()).item()
        task_loss_values = {f"{task}_loss": l.item() for task, l in task_losses.items()}
        weighted_task_loss_values = {
            f"{task}_loss_weighted": l.item()
            for task, l in weighted_task_losses.items()
        }

        values4logging = {
            "loss_total": loss_value,
            **task_loss_values,
            **weighted_task_loss_values,
        }

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)

        return loss, values4logging, input_dict, preds, masks

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        if self.sample_train_batch is None and self.global_rank == 0:
            self.sample_train_batch = batch
            # todo: check if we need deep copy of batch dict and tensor
        loss, values4logging, _, _, _ = self.step(batch)

        # logging for training
        for k, v in values4logging.items():
            self.log(
                f"train/{k}",
                v,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True
            )

        return {"loss": loss}

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        if self.sample_val_batch is None and self.global_rank == 0:
            self.sample_val_batch = batch
            # todo: check if we need deepcopy here
        loss, values4logging, _, _, _ = self.step(batch)
        for k, v in values4logging.items():
            self.log(
                f"val/{k}",
                v,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )
        return {"loss": loss}

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.
        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """

        optimizer = torch.optim.AdamW(
            params=self.parameters(),
            lr=self.hparams.optimizer_config["lr"],
            betas=self.hparams.optimizer_config["betas"],
            weight_decay=self.hparams.optimizer_config["weight_decay"],
        )
        # weight decay scheduler and parameter group between: model and balancer

        lr_scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            self.hparams.scheduler_config["warmup_epochs"],
            self.hparams.scheduler_config["max_epochs"],
            self.hparams.scheduler_config["start_lr"],
            self.hparams.scheduler_config["min_lr"],
        )

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
