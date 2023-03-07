from registry.registry import registry
from pytorch_lightning import LightningModule


def _try_register_MultiMAE_modules():
    try:
        from .multiMAE_module import MultiMAEPretrain  # noqa: F401
        from .habitat_datamodule import HabitatDataModule, HabitatTransformGPU
        from .imagenet_datamodule import ImagenetDataModule

        print("import MultiMAE modules success")
    except ImportError as e:
        multiMAE_import_error = e

        print("import MultiMAE module failed", e)

        @registry.register_lightningmodule(name="MultiMAEerr")
        class MultiMAEImportError(LightningModule):
            def __init__(self, *args, **kwargs):
                raise multiMAE_import_error

    # from .multiMAE_module import MultiMAEPretrain  # noqa: F401
    # from .habitat_datamodule import HabitatDataModule, HabitatTransformGPU
    # from .imagenet_datamodule import ImagenetDataModule
