from pipeline.model_utils.model_base import ModelBase

def construct_model_base(model_path: str) -> ModelBase:

    # Check for Qwen3 first (before generic 'qwen') since 'qwen3' contains 'qwen'
    if 'qwen3' in model_path.lower() or 'qwen2' in model_path.lower():
        from pipeline.model_utils.qwen3_model import Qwen3Model
        return Qwen3Model(model_path)
    elif 'qwen' in model_path.lower():
        from pipeline.model_utils.qwen_model import QwenModel
        return QwenModel(model_path)
    elif 'llama-3' in model_path.lower():
        from pipeline.model_utils.llama3_model import Llama3Model
        return Llama3Model(model_path)
    elif 'llama' in model_path.lower():
        from pipeline.model_utils.llama2_model import Llama2Model
        return Llama2Model(model_path)
    # Check for Gemma 3 first (before generic 'gemma') since 'gemma-3' contains 'gemma'
    elif 'gemma-3' in model_path.lower() or 'gemma3' in model_path.lower():
        from pipeline.model_utils.gemma3_model import Gemma3Model
        return Gemma3Model(model_path)
    elif 'gemma' in model_path.lower():
        from pipeline.model_utils.gemma_model import GemmaModel
        return GemmaModel(model_path) 
    elif 'yi' in model_path.lower():
        from pipeline.model_utils.yi_model import YiModel
        return YiModel(model_path)
    else:
        raise ValueError(f"Unknown model family: {model_path}")
