import sys
import os
from torchlightning_utils import cli, pl_instantiate
from registry.registry import registry
import models

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def main():
    # skip the program name in sys.argv
    cfg = cli.parse(sys.argv[1:])
    trainer = pl_instantiate.instantiate_trainer(
        cfg["trainer"],
        cfg["callbacks"],
        cfg["logger"],
        cfg.get("seed_everything", None),
    )
    model = registry.get_lightningmodule(cfg["model_name"])(**cfg["model"])
    datamodule = registry.get_datamodule(cfg["data_name"])(**cfg["data"])

    trainer.fit(model, datamodule)


if __name__ == "__main__":
    main()
