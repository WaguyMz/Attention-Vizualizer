"""
Attention computation utilities: raw attention, rollout (with residuals), and aggregated views.
"""
from __future__ import annotations

from typing import List, Tuple

import torch


def _get_attentions_from_model(model: "torch.nn.Module", input_ids: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, bool]:
    """
    Run model forward and return (attention_stack, used_inner).
    If the top-level model doesn't return attentions, try inner .model / .transformer.
    """
    model.eval()
    with torch.no_grad():
        try:
            outputs = model(input_ids, output_attentions=True, **kwargs)
        except TypeError:
            outputs = model(input_ids, **kwargs)

    if getattr(outputs, "attentions", None) is not None and len(outputs.attentions) > 0:
        return torch.stack([a.cpu() for a in outputs.attentions], dim=0), False

    # Jina and similar wrappers: try inner transformer (e.g. .model, .transformer, .encoder)
    for attr in ("model", "transformer", "encoder", "bert", "qwen"):
        inner = getattr(model, attr, None)
        if inner is not None and hasattr(inner, "forward"):
            try:
                with torch.no_grad():
                    inner_out = inner(input_ids, output_attentions=True, **kwargs)
                if getattr(inner_out, "attentions", None) is not None and len(inner_out.attentions) > 0:
                    return torch.stack([a.cpu() for a in inner_out.attentions], dim=0), True
            except Exception:
                continue
    raise ValueError("Model did not return attention weights. Try a standard encoder (e.g. BERT, Qwen3) with output_attentions=True.")


def get_attention_tensors(
    model: "torch.nn.Module",
    tokenizer: "PreTrainedTokenizer",
    text: str,
    device: str = "cpu",
    max_length: int = 512,
) -> Tuple[torch.Tensor, List[str], torch.Tensor]:
    """
    Run model on text and return attention weights, token list, and input_ids.
    Returns:
        attention: (num_layers, batch, num_heads, seq_len, seq_len)
        tokens: list of token strings
        input_ids: (1, seq_len)
    """
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=True,
    )
    input_ids = inputs["input_ids"].to(device)
    kwargs = {}
    if "token_type_ids" in inputs:
        kwargs["token_type_ids"] = inputs["token_type_ids"].to(device)
    if "attention_mask" in inputs:
        kwargs["attention_mask"] = inputs["attention_mask"].to(device)

    attention_stack = _get_attentions_from_model(model, input_ids, **kwargs)[0]
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu().tolist())
    return attention_stack, tokens, input_ids.cpu()


def raw_attention(
    attention_stack: torch.Tensor,
    layer: int,
    head: int,
) -> torch.Tensor:
    """
    Get raw attention for one layer and one head.
    attention_stack: (num_layers, batch, num_heads, seq, seq)
    Returns: (seq, seq) numpy-friendly tensor.
    """
    return attention_stack[layer, 0, head].clone()


def attention_rollout(
    attention_stack: torch.Tensor,
    start_layer: int = 0,
    end_layer: int | None = None,
    add_residual: bool = True,
) -> torch.Tensor:
    """
    Compute attention rollout (recursive combination with residual).
    attention_stack: (num_layers, batch, num_heads, seq, seq)
    Returns: (seq, seq) averaged over heads, then rollout over layers.
    """
    num_layers, batch, num_heads, seq_len, _ = attention_stack.shape
    if end_layer is None:
        end_layer = num_layers
    device = attention_stack.device
    dtype = attention_stack.dtype

    # Average over heads: (num_layers, batch, seq, seq)
    layer_attn = attention_stack[:, 0].mean(dim=1)

    if add_residual:
        eye = torch.eye(seq_len, device=device, dtype=dtype).unsqueeze(0).unsqueeze(0)
        eye = eye.expand(num_layers, 1, seq_len, seq_len)
        layer_attn = layer_attn + eye.squeeze(1)
        layer_attn = layer_attn / layer_attn.sum(dim=-1, keepdim=True).clamp(min=1e-9)

    joint = layer_attn[start_layer]
    for i in range(start_layer + 1, end_layer):
        joint = layer_attn[i] @ joint
    return joint.squeeze(0).cpu()


def averaged_attention(
    attention_stack: torch.Tensor,
    layer: int | None = None,
    head: int | None = None,
    layers: List[int] | None = None,
) -> torch.Tensor:
    """
    Average attention over heads (and optionally over given layers).
    If layer/head are set, return that single head.
    If layers is set, average over those layers and all heads.
    Otherwise average over all heads of layer 0.
    """
    num_layers, batch, num_heads, seq_len, _ = attention_stack.shape
    if layer is not None and head is not None:
        return attention_stack[layer, 0, head].cpu().clone()
    if layers is not None:
        return attention_stack[layers, 0].mean(dim=(0, 1)).cpu().clone()
    if layer is not None:
        return attention_stack[layer, 0].mean(dim=0).cpu().clone()
    return attention_stack[:, 0].mean(dim=(0, 1)).cpu().clone()
