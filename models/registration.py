from .MultiMAE import _try_register_MultiMAE_modules
from .tMultiMAE import _try_register_tMultiMAE_modules
from registry.registry import registry
import logging


def make_module(id_module, **kwargs):
    logging.info("initializing module {}".format(id_module))
    _module = registry.get_lightningmodule(id_module)
    assert _module is not None, "Could not find module with name {}".format(id_module)
    return _module(**kwargs)


_try_register_MultiMAE_modules()
_try_register_tMultiMAE_modules()
