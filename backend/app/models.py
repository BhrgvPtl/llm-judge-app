import os
from typing import Dict
import torch # New: Import torch to check for CUDA
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from huggingface_hub import login

HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN:
    try:
        login(HF_TOKEN)
    except Exception:
        # Fails silently if the token is invalid or already logged in
        pass

_model_cache: Dict[str, object] = {}
_tokenizer_cache: Dict[str, object] = {}
_pipe_cache: Dict[str, object] = {}

# Set device to GPU index 0 (if CUDA is available) or CPU (-1)
DEVICE = 0 if torch.cuda.is_available() else -1


def get_text_generation_pipeline(model_id: str):
    """
    Pipeline loader that uses GPU if available.
    """
    if model_id in _pipe_cache:
        return _pipe_cache[model_id]

    tokenizer = _tokenizer_cache.get(model_id)
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
        )
        _tokenizer_cache[model_id] = tokenizer

    model = _model_cache.get(model_id)
    if model is None:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            # Use half-precision for memory optimization on GPU
            torch_dtype=torch.bfloat16 if DEVICE >= 0 else None,
        )
        # REMOVED: model.to("cpu")
        model.eval()
        _model_cache[model_id] = model

    text_gen = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=DEVICE, # Passes the device index (0 for GPU, -1 for CPU)
    )

    _pipe_cache[model_id] = text_gen
    return text_gen