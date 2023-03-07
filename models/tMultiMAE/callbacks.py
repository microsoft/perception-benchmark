from typing import Dict, List
import torch
import numpy as np
from pytorch_lightning import Callback
from pytorch_lightning.utilities.cli import CALLBACK_REGISTRY
from pytorch_lightning.utilities import rank_zero_only
from .visualization import visualize_predictions
from azureml.core import Run


@CALLBACK_REGISTRY
class PlotTMultiMAEInference(Callback):
    def __init__(self, inputs: list, mask_on_inputs_during_inference: bool) -> None:
        super().__init__()
        self.inputs = inputs
        self.mask_on_inputs_during_inference = mask_on_inputs_during_inference
        self.datamode = "val"
        self.run = Run.get_context()

    def tMultiMAE_inference(
        self, batch: dict, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ):
        # pick a random element from batch for visualization
        ele_idx = torch.randint(low=0, high=batch.size(0), size=(1,))
        for key in batch:
            # only take the first element in batch
            batch[key] = batch[key][ele_idx].unsqueeze(0)
        # batch = trainer.datamodule.transform(batch)
        _, _, input_dict, preds, masks = pl_module.step(batch)
        # output under amp (autocast) can be float16, which needs to be cast to 32 for plotting
        for key in preds:
            if key in input_dict:
                preds[key] = preds[key].to(input_dict[key].dtype)

        animation, fps = visualize_predictions(
            input_dict,
            preds,
            masks,
            image_size=pl_module.hparams.model_config["image_size"],
            num_frames=pl_module.hparams.model_config["num_frames"],
            temporal_patch_size=pl_module.hparams.model_config["temporal_patch_size"],
            spatial_patch_size=pl_module.hparams.model_config["spatial_patch_size"],
        )
        tag_name = f"{self.datamode}/recon_{trainer.current_epoch:03d}"

        # video tensor of size T H W C: https://moviepy.readthedocs.io/en/latest/ref/VideoClip/VideoClip.html?highlight=Numpy#moviepy.video.VideoClip.ImageClip.get_frame
        vid_tensor = torch.from_numpy(
            np.stack([frame for frame in animation.iter_frames(fps=fps)], axis=0)
        ).unsqueeze(0)
        # 1 T H W C -> 1 T C H W required by tensorboard: https://pytorch.org/docs/stable/tensorboard.html
        vid_tensor = vid_tensor.permute(0, 1, 4, 2, 3)

        # log image grids to tensorboard
        pl_module.logger.experiment.add_video(
            tag=tag_name,
            vid_tensor=vid_tensor,
            global_step=trainer.current_epoch,
            fps=fps,
        )

        # log image grids to ml-flow
        if "OfflineRun" not in self.run.id:
            animation.write_gif(f"{tag_name}.gif", fps=fps, verbose=False, logger=None)
            animation.write_videofile(
                f"{tag_name}.mp4", fps=fps, verbose=False, logger=None
            )
            self.run.log_image(
                name=f"{self.datamode}/recon_{trainer.current_epoch:03d}",
                path=f"{tag_name}.gif",
                description="tMultiMAE reconstruction",
            )

        # cross-modal prediction
        for inputs in self.inputs:
            self.cross_modal_inference(inputs, input_dict, masks, pl_module, trainer)

    def cross_modal_inference(
        self,
        inputs: List[str],
        input_dict: Dict[str, torch.Tensor],
        masks: Dict[str, torch.Tensor],
        pl_module: "pl.LightningModule",
        trainer: "pl.Trainer",
    ):
        """cross-modal prediction
        Notes:
            Call me only during inference! Otherwise the model grads will be messed up!
        Args:
            inputs (List[str]): a list of inputs feed into multiMAE (without masking?)
            input_dict (_type_): input batch
            mask (Dict[str, torch.Tensor]): mask here is only provided as shape-reference
        """
        # mask == 1 -> the patch is masked out!
        task_masks = {}
        for key, mask in masks.items():
            if key in inputs:
                task_masks[key] = (
                    mask
                    if self.mask_on_inputs_during_inference
                    else torch.full_like(mask, 0)
                )
            else:
                task_masks[key] = torch.full_like(mask, 1)

        preds, masks = pl_module.model.forward(
            input_dict, mask_inputs=True, task_masks=task_masks
        )

        preds = {domain: pred.detach() for domain, pred in preds.items()}
        masks = {domain: mask.detach() for domain, mask in masks.items()}

        # cast preds data type in case float16 is used
        for key in preds:
            if key in input_dict:
                preds[key] = preds[key].to(input_dict[key].dtype)

        animation, fps = visualize_predictions(
            input_dict,
            preds,
            masks,
            image_size=pl_module.hparams.model_config["image_size"],
            num_frames=pl_module.hparams.model_config["num_frames"],
            temporal_patch_size=pl_module.hparams.model_config["temporal_patch_size"],
            spatial_patch_size=pl_module.hparams.model_config["spatial_patch_size"],
        )

        tag_name = (
            f"{self.datamode}/recon_{trainer.current_epoch:03d}_from_{'_'.join(inputs)}"
        )

        vid_tensor = torch.from_numpy(
            np.stack([frame for frame in animation.iter_frames(fps=fps)], axis=0)
        ).unsqueeze(0)
        # 1 T H W C -> 1 T C H W required by tensorboard: https://pytorch.org/docs/stable/tensorboard.html
        vid_tensor = vid_tensor.permute(0, 1, 4, 2, 3)

        # log image to tensorboard
        pl_module.logger.experiment.add_video(
            tag=tag_name,
            vid_tensor=vid_tensor,
            global_step=trainer.current_epoch,
            fps=fps,
        )

        # log image to mlflow
        if "OfflineRun" not in self.run.id:
            animation.write_gif(f"{tag_name}.gif", fps=fps, verbose=False, logger=None)
            animation.write_videofile(
                f"{tag_name}.mp4", fps=fps, verbose=False, logger=None
            )
            self.run.log_image(
                name=f"{self.datamode}/recon_{trainer.current_epoch:03d}",
                path=f"{tag_name}.gif",
                description="tMultiMAE reconstruction",
            )

    @rank_zero_only
    def on_validation_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        # todo: random item from validation dataloader
        self.datamode = "val"
        batch = pl_module.sample_val_batch
        self.tMultiMAE_inference(batch, trainer, pl_module)
        self.datamode = "train"
        batch = pl_module.sample_train_batch 
        self.tMultiMAE_inference(batch, trainer, pl_module)
        return
