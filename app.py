"""
Streamlit Attention Visualizer for transformer encoders (e.g. Jina v5, BERT).
- Heatmap of self-attention (layer/head selection)
- Model view (grid of heads per layer), Rollout, Head view
- Smart model loading: singleton/cache to avoid reloading
"""
from __future__ import annotations

import numpy as np
import streamlit as st
import torch

from model_service import get_model_and_tokenizer, DEFAULT_MODEL_NAME
from attention_utils import (
    get_attention_tensors,
    raw_attention,
    attention_rollout,
    averaged_attention,
)
from viz_components import (
    heatmap_single,
    heatmap_dark,
    model_view_grid,
    head_view_tokens_bars,
)

# Page config
st.set_page_config(
    page_title="Attention Visualizer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for attention.streamlit.app-like styling
st.markdown(
    """
    <style>
    .stApp { }
    h1, h2, h3 { color: #fafafa !important; }
    .stSelectbox label, .stSlider label { color: #ccc !important; }
    div[data-testid="stSidebar"] { background: #0e1117; }
    .viz-container { padding: 1rem 0; border-radius: 8px; }
    </style>
    """,
    unsafe_allow_html=True,
)

def main():
    st.markdown(
        '<p style="color: #4A90D9; font-size: 2.25rem; font-weight: 700;">🔍 Attention Visualizer</p>',
        unsafe_allow_html=True,
    )
    st.caption("Self-attention heatmaps for transformer encoders — Jina v5, BERT, Qwen3, etc.")

    sidebar = st.sidebar
    sidebar.header("Settings")

    model_name = sidebar.text_input(
        "Model",
        value=DEFAULT_MODEL_NAME,
        help="HuggingFace model id (e.g. jinaai/jina-embeddings-v5-text-small, prajjwal1/bert-small)",
    )
    text_input = sidebar.text_area(
        "Input text",
        value="The cat sat on the mat.",
        height=100,
        help="Sentence to run through the model.",
    )
    dark_theme = sidebar.checkbox("Dark theme", value=True)
    include_special_tokens = sidebar.checkbox(
        "Include special tokens ([CLS], [SEP], [PAD])",
        value=False,
        help="Show [CLS], [SEP], and [PAD] in heatmaps (they often act as attention sinks).",
    )
    max_length = sidebar.slider("Max tokens", 16, 128, 64, help="Truncate input to this many tokens.")

    if not text_input.strip():
        st.info("Enter some text in the sidebar and run.")
        return

    with st.spinner("Loading model (cached after first run)…"):
        try:
            model, tokenizer = get_model_and_tokenizer(model_name, use_streamlit_cache=True)
        except Exception as e:
            st.error(f"Failed to load model: {e}")
            return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    with st.spinner("Running forward pass…"):
        try:
            attention_stack, tokens, _ = get_attention_tensors(
                model, tokenizer, text_input, device=device, max_length=max_length
            )
        except Exception as e:
            st.error(f"Forward pass failed: {e}")
            return

    num_layers, _, num_heads, seq_len, _ = attention_stack.shape
    st.sidebar.success(f"Model loaded · {num_layers} layers, {num_heads} heads, {len(tokens)} tokens")

    # Optionally filter out special tokens ([CLS], [SEP], [PAD]) – they often act as attention sinks.
    special_tokens = {"[CLS]", "[SEP]", "[PAD]"}
    if include_special_tokens:
        visible_indices = list(range(len(tokens)))
        tokens_visible = tokens.copy()
    else:
        visible_indices = [i for i, tok in enumerate(tokens) if tok not in special_tokens]
        if not visible_indices:
            visible_indices = list(range(len(tokens)))
        tokens_visible = [tokens[i] for i in visible_indices]

    # Layer / head selectors
    sidebar.subheader("Layer & head")
    layer_idx = sidebar.selectbox(
        "Layer",
        range(num_layers),
        format_func=lambda i: f"Layer {i}",
        index=min(1, num_layers - 1),
    )
    head_idx = sidebar.selectbox(
        "Head",
        range(num_heads),
        format_func=lambda i: f"Head {i}",
        index=0,
    )

    viz_type = sidebar.radio(
        "Visualization",
        [
            "Heatmap (single head)",
            "Model view (all heads, one layer)",
            "Attention rollout",
            "Head view (from token)",
        ],
        index=0,
    )

    if viz_type == "Heatmap (single head)":
        raw = raw_attention(attention_stack, layer_idx, head_idx)
        arr = raw.numpy()
        # Drop [CLS], [SEP], [PAD] from both axes
        arr = arr[np.ix_(visible_indices, visible_indices)]
        title = f"Layer {layer_idx} · Head {head_idx}"
        if dark_theme:
            fig = heatmap_dark(arr, tokens_visible, title=title)
        else:
            fig = heatmap_single(arr, tokens_visible, title=title)
        st.plotly_chart(
            fig,
            use_container_width=True,
            config={"displayModeBar": True, "responsive": True},
        )

    elif viz_type == "Model view (all heads, one layer)":
        fig = model_view_grid(
            attention_stack,
            tokens_visible,
            layer=layer_idx,
            num_heads=num_heads,
            max_heads=min(12, num_heads),
            dark=dark_theme,
            visible_indices=visible_indices if not include_special_tokens else None,
        )
        st.plotly_chart(
            fig,
            use_container_width=True,
            config={"displayModeBar": True, "responsive": True},
        )

    elif viz_type == "Attention rollout":
        start = sidebar.slider("Rollout start layer", 0, num_layers - 1, 0)
        end = sidebar.slider("Rollout end layer", 0, num_layers, num_layers)
        if end <= start:
            end = start + 1
        rollout = attention_rollout(attention_stack, start_layer=start, end_layer=end, add_residual=True)
        arr = rollout.numpy()
        arr = arr[np.ix_(visible_indices, visible_indices)]
        title = f"Attention rollout (layers {start}–{end - 1})"
        if dark_theme:
            fig = heatmap_dark(arr, tokens_visible, title=title)
        else:
            fig = heatmap_single(arr, tokens_visible, title=title)
        st.plotly_chart(
            fig,
            use_container_width=True,
            config={"displayModeBar": True, "responsive": True},
        )

    elif viz_type == "Head view (from token)":
        # Select from visible (non-special) tokens only
        from_token = sidebar.selectbox(
            "Attention from token",
            range(len(tokens_visible)),
            format_func=lambda i: f"{visible_indices[i]}: {tokens_visible[i]}",
            index=min(1, max(0, len(tokens_visible) - 1)),
        )
        raw = raw_attention(attention_stack, layer_idx, head_idx)
        arr = raw.numpy()
        arr_vis = arr[np.ix_(visible_indices, visible_indices)]
        fig = head_view_tokens_bars(arr_vis, tokens_visible, from_token, top_k=15)
        st.plotly_chart(
            fig,
            use_container_width=True,
            config={"displayModeBar": True, "responsive": True},
        )

    # Token list below
    with st.expander("Token list"):
        st.write(" | ".join(tokens_visible))
        if include_special_tokens:
            st.caption("All tokens shown, including **[CLS]**, **[SEP]**, **[PAD]**.")
        else:
            st.caption("Special tokens **[CLS]**, **[SEP]**, **[PAD]** are hidden from the visualizations.")


if __name__ == "__main__":
    main()
