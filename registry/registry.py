from typing import Optional, Type
import numpy as np
import torch
from pytorch_lightning import LightningModule, LightningDataModule
from torchlightning_utils.base_registry import BaseRegistry


class Registry(BaseRegistry):
    """Registry for various entities

    Args:
        BaseRegistry (_type_): _description_

    Returns:
        _type_: _description_
    """

    @classmethod
    def register_torchnnmodule(cls, to_register=None, *, name: Optional[str] = None):
        r"""Register an input transform to registry with key :p:`name`
        :param name: Key with which the task will be registered.
        """

        return cls._register_impl(
            "inputtransform", to_register, name, assert_type=torch.nn.Module
        )

    @classmethod
    def register_datamodule(cls, to_register=None, *, name: Optional[str] = None):
        r"""Register a pytorch lightning datamodule to registry with key :p:`name`
        :param name: Key with which the task will be registered.
        """

        return cls._register_impl(
            "datamodule", to_register, name, assert_type=LightningDataModule
        )

    @classmethod
    def register_lightningmodule(cls, to_register=None, *, name: Optional[str] = None):
        r"""Register a pytorch lightning module to registry with key :p:`name`
        :param name: Key with which the task will be registered.
        """

        return cls._register_impl(
            "lightningmodule", to_register, name, assert_type=LightningModule
        )

    @classmethod
    def get_torchnnmodule(cls, name: str) -> Type[torch.nn.Module]:
        return cls._get_impl("inputtransform", name)

    @classmethod
    def get_datamodule(cls, name: str) -> Type[LightningModule]:
        return cls._get_impl("datamodule", name)

    @classmethod
    def get_lightningmodule(cls, name: str) -> Type[LightningModule]:
        return cls._get_impl("lightningmodule", name)


registry = Registry()
