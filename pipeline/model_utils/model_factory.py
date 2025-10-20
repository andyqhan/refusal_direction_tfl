from pipeline.model_utils.model_base import ModelBase
from pipeline.model_utils.transformerlens_model import TransformerLensModel


def construct_model_base(model_path: str) -> ModelBase:
    """
    Construct a model base instance using TransformerLens.

    Args:
        model_path: Path or identifier for the HuggingFace model

    Returns:
        ModelBase instance powered by TransformerLens
    """
    return TransformerLensModel(model_path)
