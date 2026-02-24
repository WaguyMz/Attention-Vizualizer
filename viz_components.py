"""
Visualization components: heatmaps and attention.streamlit.app-style views.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


# Light, educative palette: few steps, low color variation (easy to read)
# [position, color]: 0 = low attention, 1 = high attention
ATTENTION_COLORSCALE_LIGHT = [
    [0.0, "#f5f5f5"],   # very light gray
    [0.4, "#bbdefb"],    # light blue
    [0.7, "#64b5f6"],   # medium blue
    [1.0, "#1565c0"],   # solid blue
]
ATTENTION_COLORSCALE_DARK = [
    [0.0, "#37474f"],   # dark gray
    [0.4, "#546e7a"],
    [0.7, "#4fc3f7"],   # light cyan
    [1.0, "#e3f2fd"],   # very light (high attention)
]

def heatmap_single(
    attn: np.ndarray,
    tokens: List[str],
    title: str = "Attention",
    colorscale: Optional[List] = None,
    height: Optional[int] = 420,
) -> go.Figure:
    """Single attention heatmap with token labels."""
    if colorscale is None:
        colorscale = ATTENTION_COLORSCALE_LIGHT
    fig = go.Figure(
        data=go.Heatmap(
            z=attn,
            x=tokens,
            y=tokens,
            colorscale=colorscale,
            hoverongaps=False,
            text=np.round(attn, 3),
            texttemplate="%{text}",
            textfont={"size": 10},
            showscale=True,
            colorbar=dict(title="Attention"),
        )
    )
    fig.update_layout(
        title=title,
        xaxis=dict(tickangle=-45, tickfont=dict(size=10)),
        yaxis=dict(autorange="reversed"),
        margin=dict(l=120, r=40, t=50, b=120),
        height=height or 420,
        template="plotly_white",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#fafafa",
    )
    return fig


def heatmap_dark(
    attn: np.ndarray,
    tokens: List[str],
    title: str = "Attention",
    height: Optional[int] = 420,
) -> go.Figure:
    """Dark-themed heatmap with light, low-variation palette."""
    fig = go.Figure(
        data=go.Heatmap(
            z=attn,
            x=tokens,
            y=tokens,
            colorscale=ATTENTION_COLORSCALE_DARK,
            hoverongaps=False,
            text=np.round(attn, 3),
            texttemplate="%{text}",
            textfont={"size": 9, "color": "#e0e0e0"},
            showscale=True,
            colorbar=dict(title="Attention", tickfont=dict(color="#ccc")),
        )
    )
    fig.update_layout(
        title=dict(text=title, font=dict(color="#fff", size=14)),
        xaxis=dict(
            tickangle=-45,
            tickfont=dict(size=9, color="#e0e0e0"),
            gridcolor="rgba(128,128,128,0.3)",
        ),
        yaxis=dict(
            autorange="reversed",
            tickfont=dict(size=9, color="#e0e0e0"),
            gridcolor="rgba(128,128,128,0.3)",
        ),
        margin=dict(l=120, r=40, t=50, b=120),
        height=height or 420,
        template="plotly_dark",
        paper_bgcolor="#1e1e1e",
        plot_bgcolor="#2d2d2d",
    )
    return fig


def model_view_grid(
    attention_stack,
    tokens: List[str],
    layer: int,
    num_heads: int,
    max_heads: int = 12,
    dark: bool = True,
    height_per_row: int = 140,
) -> go.Figure:
    """
    Grid of heatmaps: one small heatmap per head for the selected layer (attention.streamlit.app model view style).
    attention_stack: (num_layers, batch, num_heads, seq, seq) tensor or numpy.
    """
    if isinstance(attention_stack, torch.Tensor):
        attention_stack = attention_stack.cpu().numpy()
    heads_to_show = min(num_heads, max_heads)
    ncols = 4
    nrows = (heads_to_show + ncols - 1) // ncols
    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        subplot_titles=[f"Head {h}" for h in range(heads_to_show)],
        vertical_spacing=0.08,
        horizontal_spacing=0.06,
    )
    seq_len = len(tokens)
    for i in range(heads_to_show):
        row, col = i // ncols + 1, i % ncols + 1
        attn = attention_stack[layer, 0, i]
        if hasattr(attn, "cpu"):
            attn = attn.cpu().numpy()
        fig.add_trace(
            go.Heatmap(
                z=attn,
                x=tokens,
                y=tokens,
                colorscale=ATTENTION_COLORSCALE_DARK if dark else ATTENTION_COLORSCALE_LIGHT,
                showscale=(i == 0),
                hoverongaps=False,
            ),
            row=row,
            col=col,
        )
    fig.update_layout(
        title=dict(text=f"Layer {layer} — all heads", font=dict(size=14)),
        height=nrows * height_per_row + 80,
        template="plotly_dark" if dark else "plotly_white",
        margin=dict(l=80, r=40, t=60, b=80),
        paper_bgcolor="#1e1e1e" if dark else "#ffffff",
        plot_bgcolor="#2d2d2d" if dark else "#fafafa",
    )
    fig.update_yaxes(autorange="reversed")
    return fig


def head_view_tokens_bars(
    attn: np.ndarray,
    tokens: List[str],
    from_token_idx: int,
    top_k: int = 15,
) -> go.Figure:
    """
    Bar chart: for a chosen 'from' token, show attention weights to other tokens (head view style).
    """
    row = attn[from_token_idx]
    indices = np.argsort(row)[::-1][:top_k]
    fig = go.Figure(
        data=[
            go.Bar(
                x=[tokens[i] for i in indices],
                y=row[indices],
                marker=dict(
                    color=row[indices],
                    colorscale=ATTENTION_COLORSCALE_LIGHT,
                    cmin=0,
                    cmax=1,
                    showscale=True,
                    colorbar=dict(title="Attention", len=0.4),
                ),
                text=[f"{row[i]:.3f}" for i in indices],
                textposition="outside",
                textfont=dict(size=11),
            )
        ]
    )
    fig.update_layout(
        title=f"Attention from token: \"{tokens[from_token_idx]}\"",
        xaxis_title="Token",
        yaxis_title="Attention weight",
        template="plotly_white",
        height=360,
        margin=dict(b=120),
    )
    return fig
