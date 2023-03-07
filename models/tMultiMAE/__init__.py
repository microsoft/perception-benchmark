from registry.registry import registry
from pytorch_lightning import LightningModule


def _try_register_tMultiMAE_modules():
    try:
        from .tMultiMAE_module import tMultiMAEPretrain  # noqa: F401
        from .habitatvideo_datamodule import (
            HabitatVideoDataModule,
            HabitatVideoTransform,
        )

        print("import tMultiMAE modules success")
    except ImportError as e:
        tMultiMAE_import_error = e

        print("import tMultiMAE module failed", e)

        @registry.register_lightningmodule(name="tMultiMAEerr")
        class tMultiMAEImportError(LightningModule):
            def __init__(self, *args, **kwargs):
                raise tMultiMAE_import_error

    # from .tMultiMAE_module import tMultiMAEPretrain  # noqa: F401
    # from .habitatvideo_datamodule import (
    #     HabitatVideoDataModule,
    #     HabitatVideoTransform,
    # )

    # print("import tMultiMAE modules success!")
