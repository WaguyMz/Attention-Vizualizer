"""
Model service: singleton/cached loading of transformer models for attention visualization.
Uses Streamlit's st.cache_resource when in Streamlit context, else module-level singleton.
Supports Jina embeddings v5 (Qwen3-based) and other HuggingFace encoder models.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Tuple

if TYPE_CHECKING:
    import torch
    from transformers import PreTrainedModel, PreTrainedTokenizer

# Module-level cache (used when not in Streamlit or as fallback)
_model: Optional[Any] = None
_tokenizer: Optional[Any] = None
_model_name_loaded: Optional[str] = None

# BERT base uncased: standard encoder, 12 layers, 12 heads, widely used for attention viz
# https://huggingface.co/google-bert/bert-base-uncased
DEFAULT_MODEL_NAME = "google-bert/bert-base-uncased"

# Fallback when Jina load fails (e.g. Qwen3Model dtype init bug): same backbone as Jina v5 small
JINA_ATTENTION_FALLBACK_MODEL = "Qwen/Qwen3-0.6B-Base"

# Models that require trust_remote_code (custom forward)
TRUST_REMOTE_CODE_MODELS = (
    "jinaai/jina-embeddings-v5-text-small",
    "jinaai/jina-embeddings-v5-text-nano",
)


def _load_model_impl(model_name: str) -> Tuple[Any, Any]:
    """Actual loading logic. Called once per model_name."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    use_trust_remote = model_name in TRUST_REMOTE_CODE_MODELS or "jinaai/jina-embeddings" in model_name
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=use_trust_remote,
    )
    # Eager attention required for output_attentions (sdpa/flash_attention don't return weights)
    attn_kwargs = {"attn_implementation": "eager"}
    try:
        model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=use_trust_remote,
            torch_dtype=torch.float32,  # avoid dtype being passed to inner Qwen3Model in a way that raises
            **attn_kwargs,
        )
    except TypeError as e:
        if "dtype" in str(e) and "jinaai/jina-embeddings" in model_name:
            # Jina wrapper passes dtype to Qwen3Model which doesn't accept it; use same backbone for attention
            model = AutoModel.from_pretrained(
                JINA_ATTENTION_FALLBACK_MODEL,
                output_attentions=True,
                **attn_kwargs,
            )
            tokenizer = AutoTokenizer.from_pretrained(JINA_ATTENTION_FALLBACK_MODEL)
            model_name = JINA_ATTENTION_FALLBACK_MODEL
        else:
            raise
    # Ensure we get attention weights from the forward pass
    if hasattr(model, "config"):
        model.config.output_attentions = True
    model.eval()
    return model, tokenizer


def get_model_and_tokenizer(
    model_name: str = DEFAULT_MODEL_NAME,
    use_streamlit_cache: bool = True,
):
    """
    Return (model, tokenizer) for the given model name.
    Uses a single global load per model_name: no reload on repeated calls.
    When use_streamlit_cache is True and streamlit is available, uses @st.cache_resource.
    """
    global _model, _tokenizer, _model_name_loaded

    try:
        import streamlit as st
        in_streamlit = True
    except Exception:
        in_streamlit = False

    if in_streamlit and use_streamlit_cache:
        return _cached_load_streamlit(model_name)

    # Non-Streamlit or cache disabled: use module-level singleton
    if _model is None or _tokenizer is None or _model_name_loaded != model_name:
        _model, _tokenizer = _load_model_impl(model_name)
        _model_name_loaded = model_name
    return _model, _tokenizer


def _cached_load_streamlit(model_name: str):
    """Streamlit-specific cached loader (one load per model_name per session)."""
    import streamlit as st

    @st.cache_resource(show_spinner="Loading model & tokenizer...")
    def _load(model_name: str):
        return _load_model_impl(model_name)

    return _load(model_name)


def get_model_name_loaded() -> Optional[str]:
    return _model_name_loaded
