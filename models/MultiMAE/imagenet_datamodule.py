from typing import Dict, Any
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule

from registry.registry import registry


from .MultiMAE.utils.dataset_folder import MultiTaskImageFolder
from .MultiMAE.utils.datasets import DataAugmentationForMultiMAE


@registry.register_datamodule(name="imagenet")
class ImagenetDataModule(LightningDataModule):
    def __init__(
        self, dataset_config: Dict[str, Any], dataloader_config: Dict[str, Any]
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.dataset_config = dataset_config
        self.dataloader_config = dataloader_config

        self.transform = DataAugmentationForMultiMAE(
            dataset_config["transform_config"]["kwargs"]["input_size"],
            dataset_config["transform_config"]["kwargs"]["hflip"],
        )

        # for parameters in self.transform.parameters():
        #     parameters.requires_grad = False

    def setup(self, stage=None):
        self.train_dataset = MultiTaskImageFolder(
            self.hparams["dataset_config"]["train_data_path"],
            self.hparams["dataset_config"]["inputs"],
            transform=self.transform,
        )
        self.val_dataset = MultiTaskImageFolder(
            self.hparams["dataset_config"]["val_data_path"],
            self.hparams["dataset_config"]["inputs"],
            transform=self.transform,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.dataloader_config["batch_size"],
            num_workers=self.dataloader_config["num_workers"],
            prefetch_factor=self.dataloader_config["prefetch_factor"],
            pin_memory=self.dataloader_config["pin_memory"],
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            shuffle=True,
            batch_size=self.dataloader_config["batch_size"],
            num_workers=self.dataloader_config["val_num_workers"],
            prefetch_factor=self.dataloader_config["prefetch_factor"],
            pin_memory=self.dataloader_config["pin_memory"],
        )

    # def on_after_batch_transfer(self, batch, dataloader_idx):
    #     batch = self.transform(batch)
    #     return batch


if __name__ == "__main__":
    import models

    config_path = "/home/azureuser/AutonomousSystemsResearch/perception-benchmark/configs/multiMAE_imagenet.yaml"
    with open(config_path, mode="r") as f:
        import yaml

        config = yaml.safe_load(f.read())["data"]

    datamodule = ImagenetDataModule(
        config["dataset_config"], config["dataloader_config"]
    )
    datamodule.setup()
    for batch in datamodule.train_dataloader():
        # imsave print on element from batch
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1 import ImageGrid

        fig = plt.figure(figsize=(10, 10))
        for key in batch:
            if key == "rgb":
                img = batch[key][0].permute(1, 2, 0).detach().cpu()
                plt.imshow(img)
                plt.savefig("image_net_rgb.png")
            elif key == "depth":
                img = batch[key][0].permute(1, 2, 0).detach().cpu()
                plt.imshow(img)
                plt.savefig("image_net_depth.png")
            else:
                img = batch[key][0].detach().cpu()
                plt.imshow(img)
                plt.savefig("image_net_semseg.png")
        break
    print("reading example imagenet done")
