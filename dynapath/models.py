#!/usr/bin/env python3
"""Path-LLM-style DynaPath model with static-dynamic multimodal fusion.

The module mirrors the useful structure of Path-LLM:

1. topology/text embeddings are aligned with contrastive objectives;
2. TPfusion-style gates fuse static topology and semantic modalities;
3. dynamic traffic states are encoded separately and fused with static
   representations through reliability-aware gates;
4. fused path embeddings bypass token IDs and enter GPT-2 through
   ``inputs_embeds``.

The code is intentionally self-contained so it can be used with the current
grid-token PKDD15 tensors and later with map-matched OSM road segments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from transformers.models.gpt2.modeling_gpt2 import GPT2Model
except Exception:  # pragma: no cover - lets static checks pass without transformers.
    GPT2Model = None


@dataclass
class DynaPathLLMConfig:
    """Configuration for DynaPathLLM.

    Attributes:
        road_size: Number of road/grid tokens, including the padding token.
        dynamic_dim: Per-token dynamic feature dimension.
        hidden_dim: Shared hidden size. Use 768 when connecting to GPT-2.
        llm_name: HuggingFace model name or local GPT-2 directory.
        use_llm: Whether to use GPT-2. If false, a small Transformer encoder is used.
        freeze_llm: Freeze GPT-2 except positional embeddings and layer norms.
        pad_id: Padding token id in ``road_ids``.
        align_temperature: Temperature for contrastive alignment losses.
        lambda_ts: Weight of road-level topology-semantic alignment.
        lambda_sd: Weight of path-level static-dynamic alignment.
        regression_unit: Unit of target values, only recorded for scripts/docs.
    """

    road_size: int
    dynamic_dim: int = 6
    hidden_dim: int = 768
    llm_name: str = "gpt2"
    use_llm: bool = True
    freeze_llm: bool = True
    pad_id: Optional[int] = None
    align_temperature: float = 0.025
    lambda_ts: float = 0.1
    lambda_sd: float = 0.1
    dropout: float = 0.1
    regression_unit: str = "seconds"


class TPFusion(nn.Module):
    """Path-LLM TPfusion-style gated fusion for two aligned modalities."""

    def __init__(self, dim_a: int, dim_b: int, hidden_dim: int):
        super().__init__()
        self.proj_a = nn.Linear(dim_a, hidden_dim, bias=False)
        self.proj_b = nn.Linear(dim_b, hidden_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gate = torch.sigmoid(self.proj_a(a) + self.proj_b(b) + self.bias)
        fused = gate * a + (1.0 - gate) * b
        return fused, gate


class DynamicStateEncoder(nn.Module):
    """Encode per-token dynamic traffic features."""

    def __init__(self, dynamic_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dynamic_dim),
            nn.Linear(dynamic_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, dynamic_x: torch.Tensor) -> torch.Tensor:
        return self.net(dynamic_x)


class ReliabilityAwareFusion(nn.Module):
    """Fuse static and dynamic token states using explicit reliability signals."""

    def __init__(self, hidden_dim: int, reliability_dim: int = 1):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2 + reliability_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        static_z: torch.Tensor,
        dynamic_z: torch.Tensor,
        reliability: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        beta = self.gate(torch.cat([static_z, dynamic_z, reliability], dim=-1))
        fused = (1.0 - beta) * static_z + beta * dynamic_z
        return fused, beta


class MaskedMeanPool(nn.Module):
    """Mean-pool path states while ignoring padding."""

    def forward(self, seq: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.to(dtype=seq.dtype).unsqueeze(-1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        return (seq * mask).sum(dim=1) / denom


def info_nce(
    query: torch.Tensor,
    key: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Symmetric in-batch InfoNCE loss."""

    query = F.normalize(query, dim=-1)
    key = F.normalize(key, dim=-1)
    logits = query @ key.transpose(0, 1) / temperature
    labels = torch.arange(query.size(0), device=query.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


def feature_level_alignment_loss(
    topo: torch.Tensor,
    sem: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Feature-level alignment from Path-LLM, applied column-wise."""

    topo_f = F.normalize(topo.transpose(0, 1), dim=-1)
    sem_f = F.normalize(sem.transpose(0, 1), dim=-1)
    logits = topo_f @ sem_f.transpose(0, 1) / temperature
    labels = torch.arange(topo_f.size(0), device=topo.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


class DynaPathLLM(nn.Module):
    """Static-dynamic decoupled multimodal path representation model."""

    def __init__(
        self,
        config: DynaPathLLMConfig,
        topo_embeddings: Optional[torch.Tensor] = None,
        semantic_embeddings: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.config = config
        self.pad_id = config.road_size - 1 if config.pad_id is None else config.pad_id

        self.topo_embeddings = self._make_embedding(topo_embeddings, "topo")
        self.semantic_embeddings = self._make_embedding(semantic_embeddings, "semantic")

        self.topo_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.semantic_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.static_fusion = TPFusion(config.hidden_dim, config.hidden_dim, config.hidden_dim)
        self.dynamic_encoder = DynamicStateEncoder(
            config.dynamic_dim, config.hidden_dim, config.dropout
        )
        self.dynamic_fusion = ReliabilityAwareFusion(config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.pool = MaskedMeanPool()

        if config.use_llm:
            if GPT2Model is None:
                raise ImportError(
                    "transformers is required when DynaPathLLMConfig.use_llm=True"
                )
            self.backbone = GPT2Model.from_pretrained(
                config.llm_name,
                output_attentions=False,
                output_hidden_states=False,
            )
            if config.freeze_llm:
                self._freeze_llm_like_pathllm()
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

    def _make_embedding(self, weights: Optional[torch.Tensor], name: str) -> nn.Embedding:
        if weights is None:
            emb = nn.Embedding(self.config.road_size, self.config.hidden_dim)
            nn.init.normal_(emb.weight, mean=0.0, std=0.02)
            return emb
        if weights.size(-1) != self.config.hidden_dim:
            raise ValueError(
                f"{name} embedding dim {weights.size(-1)} != hidden_dim "
                f"{self.config.hidden_dim}"
            )
        return nn.Embedding.from_pretrained(weights.float(), freeze=False)

    def _freeze_llm_like_pathllm(self) -> None:
        """Freeze GPT-2 blocks except layer norms and positional embeddings."""

        for name, param in self.backbone.named_parameters():
            param.requires_grad = ("ln" in name) or ("wpe" in name)

    def encode_static(
        self, road_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        topo = self.topo_proj(self.topo_embeddings(road_ids))
        sem = self.semantic_proj(self.semantic_embeddings(road_ids))
        topo = F.gelu(topo)
        sem = F.gelu(sem)
        static_z, ts_gate = self.static_fusion(topo, sem)
        return static_z, topo, sem, ts_gate

    def encode_dynamic(
        self,
        dynamic_x: torch.Tensor,
        reliability_index: int = -1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dynamic_z = self.dynamic_encoder(dynamic_x)
        if reliability_index < 0:
            reliability_index = dynamic_x.size(-1) + reliability_index
        reliability = dynamic_x[..., reliability_index : reliability_index + 1]
        reliability = reliability.clamp(0.0, 1.0)
        return dynamic_z, reliability

    def forward(
        self,
        road_ids: torch.Tensor,
        dynamic_x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
        reliability_index: int = -1,
        return_loss: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Run DynaPathLLM.

        Args:
            road_ids: Long tensor ``[B, L]``.
            dynamic_x: Float tensor ``[B, L, dynamic_dim]``.
            attention_mask: Bool/0-1 tensor ``[B, L]``. If omitted, pad_id is used.
            targets: Optional TTE labels ``[B]``.
            reliability_index: Dynamic feature index used as reliability scalar.
            return_loss: Whether to compute available losses.
        """

        if attention_mask is None:
            attention_mask = road_ids.ne(self.pad_id)
        attention_mask = attention_mask.to(device=road_ids.device, dtype=torch.bool)

        static_z, topo, sem, ts_gate = self.encode_static(road_ids)
        dynamic_z, reliability = self.encode_dynamic(dynamic_x, reliability_index)
        token_z, sd_gate = self.dynamic_fusion(static_z, dynamic_z, reliability)
        token_z = self.dropout(token_z)

        if self.config.use_llm:
            hidden = self.backbone(
                inputs_embeds=token_z,
                attention_mask=attention_mask.to(dtype=torch.long),
            ).last_hidden_state
        else:
            hidden = self.backbone(token_z, src_key_padding_mask=~attention_mask)

        pooled = self.norm(self.pool(hidden, attention_mask))
        pred = self.regressor(pooled).squeeze(-1)

        out = {
            "prediction": pred,
            "path_embedding": pooled,
            "token_embedding": hidden,
            "topology_semantic_gate": ts_gate,
            "static_dynamic_gate": sd_gate,
        }
        if return_loss:
            out.update(
                self.losses(
                    pred=pred,
                    targets=targets,
                    topo=topo,
                    sem=sem,
                    static_z=static_z,
                    dynamic_z=dynamic_z,
                    attention_mask=attention_mask,
                )
            )
        return out

    def losses(
        self,
        pred: torch.Tensor,
        targets: Optional[torch.Tensor],
        topo: torch.Tensor,
        sem: torch.Tensor,
        static_z: torch.Tensor,
        dynamic_z: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        losses: dict[str, torch.Tensor] = {}

        valid = attention_mask.reshape(-1)
        topo_valid = topo.reshape(-1, topo.size(-1))[valid]
        sem_valid = sem.reshape(-1, sem.size(-1))[valid]
        if topo_valid.size(0) > 1:
            ins = info_nce(topo_valid, sem_valid, self.config.align_temperature)
            fea = feature_level_alignment_loss(
                topo_valid, sem_valid, self.config.align_temperature
            )
            losses["loss_ts_align"] = 0.5 * (ins + fea)
        else:
            losses["loss_ts_align"] = pred.sum() * 0.0

        static_path = self.pool(static_z, attention_mask)
        dynamic_path = self.pool(dynamic_z, attention_mask)
        if static_path.size(0) > 1:
            losses["loss_sd_align"] = info_nce(
                static_path, dynamic_path, self.config.align_temperature
            )
        else:
            losses["loss_sd_align"] = pred.sum() * 0.0

        if targets is not None:
            losses["loss_tte"] = F.mse_loss(pred, targets.float())
        else:
            losses["loss_tte"] = pred.sum() * 0.0

        losses["loss"] = (
            losses["loss_tte"]
            + self.config.lambda_ts * losses["loss_ts_align"]
            + self.config.lambda_sd * losses["loss_sd_align"]
        )
        return losses
