#!/usr/bin/env python3
"""Neural network baselines for path-based travel time estimation.

Provides three baseline architectures that can be trained on the same
PKDD15 grid-token data as DynaPathLLM:

- ``LSTMTTE``: bidirectional LSTM + masked mean pool + MLP head.
- ``TransformerTTE``: TransformerEncoder + masked mean pool + MLP head.
- ``PathLLMStatic``: topology-semantic fusion + backbone (GPT-2 or Transformer).

All models can run in ``no_llm`` mode where GPT-2 is replaced by a small
TransformerEncoder, enabling fast debugging without HuggingFace Transformers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from transformers.models.gpt2.modeling_gpt2 import GPT2Model
except Exception:
    GPT2Model = None


# ---------------------------------------------------------------------------
# Reuse utilities from the DynaPath module
# ---------------------------------------------------------------------------


class MaskedMeanPool(nn.Module):
    """Mean-pool sequence states while ignoring padding."""

    def forward(self, seq: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_f = mask.to(dtype=seq.dtype).unsqueeze(-1)
        denom = mask_f.sum(dim=1).clamp_min(1.0)
        return (seq * mask_f).sum(dim=1) / denom


# ---------------------------------------------------------------------------
# Baseline 1: LSTM path encoder
# ---------------------------------------------------------------------------


class LSTMTTE(nn.Module):
    """Bidirectional LSTM path encoder for travel time estimation.

    Parameters
    ----------
    road_size: int
        Number of road/grid tokens including pad.
    hidden_dim: int
        LSTM hidden size and embedding dimension.
    num_layers: int
        Number of LSTM layers.
    dropout: float
        Dropout applied after LSTM and in regressor.
    pad_id: Optional[int]
        Padding token id used when ``attention_mask`` is not provided.
    """

    def __init__(
        self,
        road_size: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.2,
        pad_id: Optional[int] = None,
    ):
        super().__init__()
        self.pad_id = road_size - 1 if pad_id is None else pad_id
        self.hidden_dim = hidden_dim

        self.token_embedding = nn.Embedding(road_size, hidden_dim)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        lstm_out = hidden_dim * 2  # bidirectional
        self.pool = MaskedMeanPool()
        self.norm = nn.LayerNorm(lstm_out)
        self.dropout = nn.Dropout(dropout)
        self.regressor = nn.Sequential(
            nn.Linear(lstm_out, lstm_out // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_out // 2, 1),
        )

    def forward(
        self,
        road_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        if attention_mask is None:
            attention_mask = road_ids.ne(self.pad_id)
        mask = attention_mask.to(dtype=torch.bool)

        emb = self.token_embedding(road_ids)
        lstm_out, _ = self.lstm(emb)
        pooled = self.norm(self.pool(lstm_out, mask))
        pooled = self.dropout(pooled)
        pred = self.regressor(pooled).squeeze(-1)
        return {"prediction": pred, "path_embedding": pooled}


# ---------------------------------------------------------------------------
# Baseline 2: Transformer path encoder
# ---------------------------------------------------------------------------


class TransformerTTE(nn.Module):
    """Transformer-encoder path model for travel time estimation.

    Parameters
    ----------
    road_size: int
        Number of road/grid tokens including pad.
    hidden_dim: int
        Transformer dimension and embedding size. Must be divisible by nhead.
    nhead: int
        Number of attention heads.
    num_layers: int
        Number of Transformer encoder layers.
    dropout: float
        Dropout probability.
    pad_id: Optional[int]
        Padding token id.
    """

    def __init__(
        self,
        road_size: int,
        hidden_dim: int = 256,
        nhead: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
        pad_id: Optional[int] = None,
    ):
        super().__init__()
        self.pad_id = road_size - 1 if pad_id is None else pad_id

        self.token_embedding = nn.Embedding(road_size, hidden_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, 512, hidden_dim) * 0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.pool = MaskedMeanPool()
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        road_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        if attention_mask is None:
            attention_mask = road_ids.ne(self.pad_id)
        mask = attention_mask.to(dtype=torch.bool)

        emb = self.token_embedding(road_ids)
        seq_len = emb.size(1)
        # add positional embeddings truncated/padded to current sequence length
        pos = self.pos_embedding[:, :seq_len, :]
        emb = emb + pos

        hidden = self.encoder(emb, src_key_padding_mask=~mask)
        pooled = self.norm(self.pool(hidden, mask))
        pooled = self.dropout(pooled)
        pred = self.regressor(pooled).squeeze(-1)
        return {"prediction": pred, "path_embedding": pooled}


# ---------------------------------------------------------------------------
# Baseline 3: Path-LLM static-only (no dynamic modality)
# ---------------------------------------------------------------------------


@dataclass
class PathLLMStaticConfig:
    road_size: int
    hidden_dim: int = 768
    llm_name: str = "gpt2"
    use_llm: bool = True
    freeze_llm: bool = True
    pad_id: Optional[int] = None
    dropout: float = 0.1


class TPFusion(nn.Module):
    """Path-LLM style gated fusion for two aligned modalities."""

    def __init__(self, dim_a: int, dim_b: int, hidden_dim: int):
        super().__init__()
        self.proj_a = nn.Linear(dim_a, hidden_dim, bias=False)
        self.proj_b = nn.Linear(dim_b, hidden_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gate = torch.sigmoid(self.proj_a(a) + self.proj_b(b) + self.bias)
        fused = gate * a + (1.0 - gate) * b
        return fused, gate


class PathLLMStatic(nn.Module):
    """Path-LLM style static path representation model.

    Uses topology and semantic embeddings, TP-fusion gating, and a
    GPT-2 (or Transformer) backbone. No dynamic traffic modality.
    """

    def __init__(
        self,
        config: PathLLMStaticConfig,
        topo_embeddings: Optional[torch.Tensor] = None,
        semantic_embeddings: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.config = config
        self.pad_id = config.road_size - 1 if config.pad_id is None else config.pad_id

        # topology embeddings
        if topo_embeddings is not None:
            self.topo_embeddings = nn.Embedding.from_pretrained(
                topo_embeddings.float(), freeze=False
            )
        else:
            self.topo_embeddings = nn.Embedding(config.road_size, config.hidden_dim)
            nn.init.normal_(self.topo_embeddings.weight, mean=0.0, std=0.02)

        # semantic embeddings
        if semantic_embeddings is not None:
            self.semantic_embeddings = nn.Embedding.from_pretrained(
                semantic_embeddings.float(), freeze=False
            )
        else:
            self.semantic_embeddings = nn.Embedding(config.road_size, config.hidden_dim)
            nn.init.normal_(self.semantic_embeddings.weight, mean=0.0, std=0.02)

        self.topo_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.semantic_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.fusion = TPFusion(config.hidden_dim, config.hidden_dim, config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.pool = MaskedMeanPool()

        if config.use_llm:
            if GPT2Model is None:
                raise ImportError("transformers required when use_llm=True")
            self.backbone = GPT2Model.from_pretrained(
                config.llm_name,
                output_attentions=False,
                output_hidden_states=False,
            )
            if config.freeze_llm:
                for name, param in self.backbone.named_parameters():
                    param.requires_grad = ("ln" in name) or ("wpe" in name)
        else:
            layer = nn.TransformerEncoderLayer(
                d_model=config.hidden_dim,
                nhead=8,
                dim_feedforward=config.hidden_dim * 4,
                dropout=config.dropout,
                batch_first=True,
            )
            self.backbone = nn.TransformerEncoder(layer, num_layers=2)

        self.norm = nn.LayerNorm(config.hidden_dim)
        self.regressor = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, 1),
        )

    def forward(
        self,
        road_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        if attention_mask is None:
            attention_mask = road_ids.ne(self.pad_id)
        mask = attention_mask.to(dtype=torch.bool)

        topo = self.topo_proj(self.topo_embeddings(road_ids))
        sem = self.semantic_proj(self.semantic_embeddings(road_ids))
        topo = F.gelu(topo)
        sem = F.gelu(sem)
        fused, _ = self.fusion(topo, sem)
        fused = self.dropout(fused)

        if self.config.use_llm:
            hidden = self.backbone(
                inputs_embeds=fused,
                attention_mask=attention_mask.to(dtype=torch.long),
            ).last_hidden_state
        else:
            hidden = self.backbone(fused, src_key_padding_mask=~mask)

        pooled = self.norm(self.pool(hidden, mask))
        pred = self.regressor(pooled).squeeze(-1)
        return {"prediction": pred, "path_embedding": pooled}
