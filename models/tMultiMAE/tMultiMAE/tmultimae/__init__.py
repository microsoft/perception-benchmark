from .criterion import MaskedCrossEntropyLoss, MaskedL1Loss, MaskedMSELoss
from .input_adapters import PatchedInputAdapter, SemSegInputAdapter
from .tmultimae import tMultiMAE, tMultiViT
from .output_adapters import (
    ConvNeXtAdapter,
    DPTOutputAdapter,
    LinearOutputAdapter,
    SegmenterMaskTransformerAdapter,
    SpatialTemporalOutputAdapter,
)
