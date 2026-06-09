"""Cross-attention/self-attention action transformer for SmolVLM-VLA."""

from __future__ import annotations

import math
from functools import partial
from typing import Final, Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------- Small utils ----------------------------------

def _to_2tuple(x) -> Tuple:
    """Minimal replacement for timm.layers.to_2tuple."""
    if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
        t = tuple(x)
        return (t[0], t[1]) if len(t) >= 2 else (t[0], t[0])
    return (x, x)


def _has_sdp_attention() -> bool:
    """Check if we can use PyTorch fused scaled_dot_product_attention."""
    return hasattr(F, "scaled_dot_product_attention")


# ---------------------------------- MLP --------------------------------------

class Mlp(nn.Module):
    """MLP used in ViT-style blocks."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        norm_layer: type[nn.Module] | None = None,
        bias: bool | Tuple[bool, bool] = True,
        drop: float | Tuple[float, float] = 0.0,
        use_conv: bool = False,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = _to_2tuple(bias)
        drop_probs = _to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = nn.GELU(approximate="tanh")
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


# -------------------------------- Attention ----------------------------------

class Attention(nn.Module):
    """Multi-Head Self-Attention with optional fused SDPA fallback."""

    fused_attn: Final[bool]

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: type[nn.Module] = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = _has_sdp_attention()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        B, T, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, T, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, T, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttention(nn.Module):
    """Multi-head cross-attention from action queries to VLM memory."""

    fused_attn: Final[bool]

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = _has_sdp_attention()

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.out_proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        memory_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, T_query, C = query.shape
        if memory.shape[0] != B or memory.shape[2] != C:
            raise ValueError(
                "Cross-attention memory must have shape [B, T_memory, C] "
                f"matching query batch/hidden dimensions, got {tuple(memory.shape)}."
            )

        T_memory = memory.shape[1]
        q = self.q_proj(query).reshape(B, T_query, self.num_heads, self.head_dim)
        k = self.k_proj(memory).reshape(B, T_memory, self.num_heads, self.head_dim)
        v = self.v_proj(memory).reshape(B, T_memory, self.num_heads, self.head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attention_mask = None
        if memory_mask is not None:
            memory_mask = memory_mask.to(device=query.device, dtype=torch.bool)
            if memory_mask.shape != (B, T_memory):
                raise ValueError(
                    f"memory_mask must have shape {(B, T_memory)}, "
                    f"got {tuple(memory_mask.shape)}."
                )
            if not torch.all(memory_mask.any(dim=1)):
                raise ValueError("Each sample must contain at least one valid VLM token.")
            attention_mask = memory_mask[:, None, None, :]

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attention_mask,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
        else:
            scores = (q * self.scale) @ k.transpose(-2, -1)
            if attention_mask is not None:
                scores = scores.masked_fill(
                    ~attention_mask,
                    torch.finfo(scores.dtype).min,
                )
            attention = scores.softmax(dim=-1)
            attention = self.attn_drop(attention)
            x = attention @ v

        x = x.transpose(1, 2).reshape(B, T_query, C)
        x = self.out_proj(x)
        return self.proj_drop(x)


# ------------------------------- Utilities -----------------------------------

def basic_init(module: nn.Module) -> None:
    """Apply basic initialization to Linear layers."""
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)


def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 100) -> torch.Tensor:
    """Create sinusoidal timestep embeddings."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=t.dtype, device=t.device)
        / half
    )
    args = t[:, None] * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class CrossSelfBlock(nn.Module):
    """Pre-norm CrossAttention -> SelfAttention -> MLP action block."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.cross_norm_q = nn.LayerNorm(hidden_size)
        self.cross_norm_kv = nn.LayerNorm(hidden_size)
        self.cross_attn = CrossAttention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            attn_drop=dropout,
            proj_drop=dropout,
        )
        self.self_norm = nn.LayerNorm(hidden_size)
        self.self_attn = Attention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            attn_drop=dropout,
            proj_drop=dropout,
        )
        self.mlp_norm = nn.LayerNorm(hidden_size)
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=int(hidden_size * mlp_ratio),
            drop=dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        memory_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.cross_attn(
            self.cross_norm_q(x),
            self.cross_norm_kv(memory),
            memory_mask,
        )
        x = x + self.self_attn(self.self_norm(x))
        x = x + self.mlp(self.mlp_norm(x))
        return x


class SmolVLMCrossSelfActionTransformer(nn.Module):
    """Flow-matching action head with repeated VLM cross- and action self-attention."""

    def __init__(
        self,
        hidden_size: int = 768,
        vlm_hidden_size: int = 576,
        depth: int = 9,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dim_action: int = 26,
        dim_propio: int = 21,
        dim_time: int = 32,
        max_num_actions: int = 10,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if max_num_actions < 1:
            raise ValueError("max_num_actions must be positive")

        self.hidden_size = hidden_size
        self.dim_action = dim_action
        self.dim_time = dim_time
        self.dim_propio = dim_propio

        self.vlm_proj = nn.Linear(vlm_hidden_size, hidden_size)
        self.action_encoder = nn.Linear(
            dim_action + dim_propio + dim_time,
            hidden_size,
        )
        self.action_pos_emb = nn.Parameter(
            torch.zeros(1, max_num_actions, hidden_size),
            requires_grad=True,
        )
        nn.init.normal_(self.action_pos_emb, std=0.02)

        self.blocks = nn.ModuleList(
            [
                CrossSelfBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )
        self.final_norm = nn.LayerNorm(hidden_size)
        self.action_decoder = nn.Linear(hidden_size, dim_action)
        self.apply(basic_init)

    def forward(
        self,
        vlm_features: torch.Tensor,
        action_with_noise: torch.Tensor,
        proprio: torch.Tensor,
        t: torch.Tensor,
        vlm_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, num_actions = action_with_noise.shape[:2]
        if num_actions > self.action_pos_emb.shape[1]:
            raise ValueError(
                f"Action horizon {num_actions} exceeds configured maximum "
                f"{self.action_pos_emb.shape[1]}."
            )

        time_emb = timestep_embedding(t, self.dim_time)
        time_tokens = time_emb.unsqueeze(1).expand(B, num_actions, self.dim_time)
        proprio_tokens = proprio.unsqueeze(1).expand(B, num_actions, proprio.shape[-1])
        action_input = torch.cat(
            [action_with_noise, proprio_tokens, time_tokens],
            dim=-1,
        )
        x = self.action_encoder(action_input)
        x = x + self.action_pos_emb[:, :num_actions].to(dtype=x.dtype)

        memory = self.vlm_proj(vlm_features)
        for block in self.blocks:
            x = block(x, memory, vlm_attention_mask)

        return self.action_decoder(self.final_norm(x))


__all__ = [
    "SmolVLMCrossSelfActionTransformer",
    "CrossSelfBlock",
    "Attention",
    "CrossAttention",
    "Mlp",
    "timestep_embedding",
]
