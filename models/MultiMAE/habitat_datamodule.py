from typing import Dict, Any
import math
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
from torchvision import transforms
import torchvision.transforms.functional as TF
from seqrecorder.datapipes import ItemDatapipeFromSeqRecord, build_datapipes
from seqrecorder.seqrecord import SeqRecord


from registry.registry import registry

# for input transform
from .MultiMAE.utils.data_constants import (
    IMAGE_TASKS,
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    IMAGENET_INCEPTION_MEAN,
    IMAGENET_INCEPTION_STD,
    DEFAULT_CROP_PCT,
)


@registry.register_datamodule(name="habitat")
class HabitatDataModule(LightningDataModule):
    def __init__(
        self, dataset_config: Dict[str, Any], dataloader_config: Dict[str, Any]
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.dataset_config = dataset_config
        self.dataloader_config = dataloader_config
        self.transform = registry.get_torchnnmodule(
            dataset_config["transform_config"]["name"]
        )(**dataset_config["transform_config"]["kwargs"])

        # for parameters in self.transform.parameters():
        #     parameters.requires_grad = False

    def setup(self, stage=None):
        train_record = SeqRecord.load_record_from_dict(
            self.dataset_config["train_recorddir"]
        )
        train_item_datapipe = ItemDatapipeFromSeqRecord(
            train_record,
            features_rename=self.dataset_config["features_rename"],
            shuffle_recordfiles=True,
        )
        self.train_datapipe = build_datapipes(
            train_item_datapipe,
            self.dataloader_config["shuffle_buffer_size"],
            self.dataloader_config["batch_size"],
            mappings=[],
        )

        val_record = SeqRecord.load_record_from_dict(
            self.dataset_config["val_recorddir"]
        )
        val_item_datapipe = ItemDatapipeFromSeqRecord(
            val_record,
            features_rename=self.dataset_config["features_rename"],
            shuffle_recordfiles=True,
        )
        self.val_datapipe = build_datapipes(
            val_item_datapipe,
            None,
            self.dataloader_config["batch_size"],
            mappings=[],
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_datapipe,
            shuffle=True,
            batch_size=None,
            num_workers=self.dataloader_config["num_workers"],
            prefetch_factor=self.dataloader_config["prefetch_factor"],
            pin_memory=self.dataloader_config["pin_memory"],
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_datapipe,
            shuffle=False,
            batch_size=None,
            num_workers=self.dataloader_config["val_num_workers"],
            prefetch_factor=self.dataloader_config["prefetch_factor"],
            pin_memory=self.dataloader_config["pin_memory"],
        )

    def test_dataloader(self):
        return DataLoader(
            self.val_datapipe,
            shuffle=False,
            batch_size=None,
            num_workers=self.dataloader_config["val_num_workers"],
            prefetch_factor=self.dataloader_config["prefetch_factor"],
            pin_memory=self.dataloader_config["pin_memory"],
        )

    def on_after_batch_transfer(self, batch, dataloader_idx):
        batch = self.transform(batch)
        return batch


# adapted from: https://github.com/EPFL-VILAB/MultiMAE/blob/main/utils/datasets.py#L66
@registry.register_torchnnmodule(name="transform_habitat")
class HabitatTransformGPU(torch.nn.Module):
    """Apply (batch) transform (preprocessing to inputs) where transforms are performed on GPU"""

    def __init__(self, input_size: int, hflip: float) -> None:
        super().__init__()
        self.rgb_mean = IMAGENET_DEFAULT_MEAN  # or IMAGENET_INCEPTION_MEAN
        self.rgb_std = IMAGENET_DEFAULT_STD  # IMAGENET_INCEPTION_STD
        self.input_size = input_size
        self.hflip = hflip

    @torch.no_grad()
    def forward(self, task_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        flip = random.random() < self.hflip  # Stores whether to flip all images or not
        ijhw = None  # Stores crop coordinates used for all tasks

        # normalization
        for task in task_dict:
            if task in ["depth"]:
                img = task_dict[task] / 2**16
                # img = img.permute(0, 3, 1, 2)  # b x 1 x H x W. This causes shape warnings!
                img = img.transpose(1, 3).contiguous().transpose(2, 3).contiguous()
            elif task in ["rgb"]:
                # to_tensor()
                img = task_dict[task].permute(0, 3, 1, 2).contiguous()  # b x 3 x H x W
                img = img.to(torch.float32).div(255)
                img = TF.normalize(img, mean=self.rgb_mean, std=self.rgb_std)
            elif task in ["semseg", "semseg_coco"]:
                # not used by us (shc)
                # TODO: add this to a config instead
                # todo: blow transform not adapted to batch version
                # Rescale to 0.25x size (stride 4)
                scale_factor = 0.25
                img = task_dict[task].resize(
                    (
                        int(self.input_size * scale_factor),
                        int(self.input_size * scale_factor),
                    )
                )
                # Using pil_to_tensor keeps it in uint8, to_tensor converts it to float (rescaled to [0, 1])
                img = TF.pil_to_tensor(img).to(torch.long).squeeze(0)

            task_dict[task] = img

        # Crop and flip all tasks randomly, but consistently for all tasks
        for task in task_dict:
            if task not in IMAGE_TASKS:
                continue
            if ijhw is None:
                # Official MAE code uses (0.2, 1.0) for scale and (0.75, 1.3333) for ratio
                ijhw = transforms.RandomResizedCrop.get_params(
                    task_dict[task], scale=(0.2, 1.0), ratio=(0.75, 1.3333)
                )
            i, j, h, w = ijhw

            # code fix by shc. The codes below does work when cropped size is not the shape of the original data
            task_dict[task] = TF.resized_crop(
                task_dict[task], i, j, h, w, [self.input_size, self.input_size]
            )
            # task_dict[task] = TF.crop(task_dict[task], i, j, h, w)
            # task_dict[task] = task_dict[task].resize((self.input_size, self.input_size))
            if flip:
                task_dict[task] = TF.hflip(task_dict[task])

        return task_dict
