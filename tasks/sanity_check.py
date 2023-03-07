import sys
import numpy as np
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from torchlightning_utils import cli, pl_instantiate
from registry.registry import registry
import models
from models.tMultiMAE.visualization import visualize_predictions


def check_datamodule(cfg):
    datamodule = registry.get_datamodule(cfg["data_name"])(**cfg["data"])
    datamodule.setup()
    for batch in datamodule.train_dataloader():
        for key, value in batch.items():
            print(key, type(value), value.size())
    # todo add visualization of batches


def check_module(cfg):
    module = registry.get_lightningmodule(cfg["model_name"])(**cfg["model"])
    datamodule = registry.get_datamodule(cfg["data_name"])(**cfg["data"])
    datamodule.setup()
    cuda = torch.device("cuda")
    module = module.to(cuda)
    for i, batch in enumerate(datamodule.train_dataloader()):
        for key in batch:
            batch[key] = batch[key].to(cuda)
        batch = datamodule.on_after_batch_transfer(batch, 0)
        module.training_step(batch, i)
        break
    print("sanity check on module forward pass done")


def check_trainer(cfg):
    trainer = pl_instantiate.instantiate_trainer(
        cfg["trainer"],
        cfg["callbacks"],
        cfg["logger"],
        cfg.get("seed_everything", None),
    )
    model = registry.get_lightningmodule(cfg["model_name"])(**cfg["model"])
    datamodule = registry.get_datamodule(cfg["data_name"])(**cfg["data"])

    trainer.fit(model, datamodule)


def check_visualization(cfg):
    pl_module = registry.get_lightningmodule(cfg["model_name"])(**cfg["model"])
    datamodule = registry.get_datamodule(cfg["data_name"])(**cfg["data"])

    datamodule.setup()
    cuda = torch.device("cuda")
    pl_module = pl_module.to(cuda)
    batch = next(iter(datamodule.train_dataloader()))
    for key in batch:
        # only take the first element in batch
        batch[key] = batch[key][0].unsqueeze(0).to(pl_module.device)
    batch = datamodule.transform(batch)
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
    animation.write_gif("./outputs/recon.gif", fps=fps, verbose=False, logger=None)
    animation.write_videofile(
        "./outputs/recon.mp4", fps=fps, verbose=False, logger=None
    )

    # video tensor of size T H W C: https://moviepy.readthedocs.io/en/latest/ref/VideoClip/VideoClip.html?highlight=Numpy#moviepy.video.VideoClip.ImageClip.get_frame
    frames = [frame for frame in animation.iter_frames(fps=fps)]
    vid_tensor = torch.from_numpy(np.stack(frames, axis=0)).unsqueeze(0)
    # 1 T H W C -> 1 T C H W required by tensorboard: https://pytorch.org/docs/stable/tensorboard.html
    vid_tensor = vid_tensor.permute(0, 1, 4, 2, 3)

    logger = TensorBoardLogger("tb_logs", name="tMultiMAE")
    logger.experiment.add_video("test/reconstruction_video", vid_tensor, 0, fps=fps)


def main():
    # skip the program name in sys.argv
    cfg = cli.parse(sys.argv[1:])

    # check_datamodule(cfg)
    # check_module(cfg)
    # check_visualization(cfg)
    check_trainer(cfg)


if __name__ == "__main__":
    main()
