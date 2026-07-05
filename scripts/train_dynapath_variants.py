#!/usr/bin/env python3
"""Train DynaPathLLM model variants for architecture ablation.

Supports the following variants:

============  ============================================================
Variant       Description
============  ============================================================
full          Complete DynaPathLLM: TPfusion + DynamicEncoder +
              ReliabilityAwareFusion + backbone + TS/SD alignment losses.
no_align      Full architecture, no TS & SD alignment losses
              (lambda_ts=0, lambda_sd=0).
no_sd_align   Full architecture, only SD alignment removed (lambda_sd=0).
concat        Replace ReliabilityAwareFusion with simple concatenation
              of static and dynamic representations.
simple_gate   Gated fusion WITHOUT explicit reliability input.
static_only   Static-only: TPfusion + backbone, no dynamic modality.
============  ============================================================

Usage:
    # Full model (no GPT-2 required):
    python scripts/train_dynapath_variants.py \\
      --data-dir data/processed/pkdd15_grid_120k_clean \\
      --variant full --no-llm --epochs 10

    # Run multiple variants:
    python scripts/train_dynapath_variants.py \\
      --data-dir data/processed/pkdd15_grid_120k_clean \\
      --variant full,concat,simple_gate,static_only,no_align,no_sd_align \\
      --no-llm --epochs 10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def require_torch():
    try:
        import torch
        from torch.utils.data import DataLoader
    except Exception as exc:
        raise SystemExit("PyTorch is required.") from exc
    return torch, DataLoader


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = np.abs(y_true - y_pred)
    return {
        "mae": float(err.mean()),
        "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        "mape": float(np.mean(err / np.maximum(y_true, 1e-6))),
        "mare": float(err.sum() / max(float(np.sum(y_true)), 1e-6)),
    }


def build_loaders(data_dir, batch_size, num_workers, DataLoader):
    from dynapath.data import DynaPathNPYDataset, dynapath_collate

    train_set = DynaPathNPYDataset(data_dir, "train")
    val_set = DynaPathNPYDataset(data_dir, "val")
    test_set = DynaPathNPYDataset(data_dir, "test")

    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=dynapath_collate,
    )
    return (
        DataLoader(train_set, shuffle=True, **loader_kwargs),
        DataLoader(val_set, shuffle=False, **loader_kwargs),
        DataLoader(test_set, shuffle=False, **loader_kwargs),
        train_set, val_set, test_set,
    )


def run_epoch(
    model, loader, optimizer, device, torch, train: bool,
    use_dynamic: bool = True,
) -> dict:
    model.train(train)
    total = {"loss": 0.0, "loss_tte": 0.0, "loss_ts_align": 0.0, "loss_sd_align": 0.0}
    count = 0
    preds, trues = [], []

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.set_grad_enabled(train):
            if use_dynamic:
                out = model(
                    road_ids=batch["road_ids"],
                    dynamic_x=batch["dynamic_x"],
                    attention_mask=batch["attention_mask"],
                    targets=batch["targets"],
                    return_loss=True,
                )
            else:
                out = model(
                    road_ids=batch["road_ids"],
                    attention_mask=batch["attention_mask"],
                    targets=batch["targets"],
                    return_loss=True,
                )

            if train:
                optimizer.zero_grad()
                out["loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        bs = batch["targets"].size(0)
        count += bs
        for key in total:
            total[key] += float(out.get(key, 0.0)) * bs
        preds.extend(out["prediction"].detach().cpu().numpy().tolist())
        trues.extend(batch["targets"].detach().cpu().numpy().tolist())

    result = {k: v / max(count, 1) for k, v in total.items()}
    result.update(regression_metrics(np.array(trues), np.array(preds)))
    return result


def load_optional_embedding(path: str | None, torch):
    if path is None:
        return None
    return torch.tensor(np.load(path), dtype=torch.float32)


def resolve_embedding_path(data_dir: Path, explicit_path: str | None, name: str) -> str | None:
    if explicit_path is not None:
        return explicit_path
    candidate = data_dir / f"{name}_embeddings.npy"
    if candidate.exists():
        return str(candidate)
    return None


def build_embedding_module(torch, road_size: int, hidden_dim: int, weights=None):
    import torch.nn as nn

    if weights is not None:
        return nn.Embedding.from_pretrained(weights.float(), freeze=False)
    emb = nn.Embedding(road_size, hidden_dim)
    nn.init.normal_(emb.weight, mean=0.0, std=0.02)
    return emb


# ---------------------------------------------------------------------------
# Variant model builders
# ---------------------------------------------------------------------------

def make_full_model(config, torch, topo_embeddings=None, semantic_embeddings=None):
    """Complete DynaPathLLM."""
    from dynapath.models import DynaPathLLM
    return DynaPathLLM(config, topo_embeddings=topo_embeddings, semantic_embeddings=semantic_embeddings)


def make_no_align_model(config, torch, topo_embeddings=None, semantic_embeddings=None):
    """DynaPathLLM with alignment losses disabled."""
    from dynapath.models import DynaPathLLM
    config.lambda_ts = 0.0
    config.lambda_sd = 0.0
    return DynaPathLLM(config, topo_embeddings=topo_embeddings, semantic_embeddings=semantic_embeddings)


def make_no_sd_align_model(config, torch, topo_embeddings=None, semantic_embeddings=None):
    """DynaPathLLM without static-dynamic alignment."""
    from dynapath.models import DynaPathLLM
    config.lambda_sd = 0.0
    return DynaPathLLM(config, topo_embeddings=topo_embeddings, semantic_embeddings=semantic_embeddings)


def make_concat_model(config, torch, topo_embeddings=None, semantic_embeddings=None):
    """Replace reliability-aware fusion with simple concatenation.

    This model concatenates static and dynamic representations (2*hidden_dim)
    and projects back to hidden_dim, then feeds into the backbone.
    """
    import torch.nn as nn
    import torch.nn.functional as F

    class ConcatFusionDynaPath(nn.Module):
        def __init__(self, cfg, topo_init=None, sem_init=None):
            super().__init__()
            self.config = cfg
            self.pad_id = cfg.road_size - 1 if cfg.pad_id is None else cfg.pad_id

            self.topo_emb = build_embedding_module(torch, cfg.road_size, cfg.hidden_dim, topo_init)
            self.sem_emb = build_embedding_module(torch, cfg.road_size, cfg.hidden_dim, sem_init)

            self.topo_proj = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
            self.sem_proj = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)

            from dynapath.models import DynamicStateEncoder
            self.static_fusion = nn.Sequential(
                nn.Linear(cfg.hidden_dim * 2, cfg.hidden_dim),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            )
            self.dynamic_encoder = DynamicStateEncoder(
                cfg.dynamic_dim, cfg.hidden_dim, cfg.dropout
            )
            # Concat fusion: [static; dynamic] -> hidden_dim
            self.concat_fusion = nn.Sequential(
                nn.Linear(cfg.hidden_dim * 2, cfg.hidden_dim),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            )
            self.dropout = nn.Dropout(cfg.dropout)
            from dynapath.models import MaskedMeanPool
            self.pool = MaskedMeanPool()

            if cfg.use_llm:
                from transformers.models.gpt2.modeling_gpt2 import GPT2Model
                self.backbone = GPT2Model.from_pretrained(
                    cfg.llm_name, output_attentions=False, output_hidden_states=False
                )
                if cfg.freeze_llm:
                    for n, p in self.backbone.named_parameters():
                        p.requires_grad = ("ln" in n) or ("wpe" in n)
            else:
                layer = nn.TransformerEncoderLayer(
                    d_model=cfg.hidden_dim, nhead=8,
                    dim_feedforward=cfg.hidden_dim * 4,
                    dropout=cfg.dropout, batch_first=True,
                )
                self.backbone = nn.TransformerEncoder(layer, num_layers=2)

            self.norm = nn.LayerNorm(cfg.hidden_dim)
            self.regressor = nn.Sequential(
                nn.Linear(cfg.hidden_dim, cfg.hidden_dim // 2),
                nn.GELU(), nn.Dropout(cfg.dropout),
                nn.Linear(cfg.hidden_dim // 2, 1),
            )

        def forward(self, road_ids, dynamic_x=None, attention_mask=None,
                    targets=None, return_loss=True):
            if attention_mask is None:
                attention_mask = road_ids.ne(self.pad_id)
            mask = attention_mask.to(dtype=torch.bool)

            topo = F.gelu(self.topo_proj(self.topo_emb(road_ids)))
            sem = F.gelu(self.sem_proj(self.sem_emb(road_ids)))
            static_z = self.static_fusion(torch.cat([topo, sem], dim=-1))
            token_z = self.dropout(static_z)

            if dynamic_x is not None:
                dynamic_z = self.dynamic_encoder(dynamic_x)
                token_z = self.concat_fusion(torch.cat([static_z, dynamic_z], dim=-1))
                token_z = self.dropout(token_z)

            if self.config.use_llm:
                hidden = self.backbone(
                    inputs_embeds=token_z,
                    attention_mask=attention_mask.to(dtype=torch.long),
                ).last_hidden_state
            else:
                hidden = self.backbone(token_z, src_key_padding_mask=~mask)

            pooled = self.norm(self.pool(hidden, mask))
            pred = self.regressor(pooled).squeeze(-1)

            out = {"prediction": pred, "path_embedding": pooled}
            if return_loss and targets is not None:
                out["loss_tte"] = F.mse_loss(pred, targets.float())
                out["loss_ts_align"] = pred.sum() * 0.0
                out["loss_sd_align"] = pred.sum() * 0.0
                out["loss"] = out["loss_tte"]
            return out

    return ConcatFusionDynaPath(config, topo_embeddings, semantic_embeddings)


def make_simple_gate_model(config, torch, topo_embeddings=None, semantic_embeddings=None):
    """Gated fusion WITHOUT explicit reliability input."""
    import torch.nn as nn
    import torch.nn.functional as F

    class SimpleGateDynaPath(nn.Module):
        def __init__(self, cfg, topo_init=None, sem_init=None):
            super().__init__()
            self.config = cfg
            self.pad_id = cfg.road_size - 1 if cfg.pad_id is None else cfg.pad_id

            self.topo_emb = build_embedding_module(torch, cfg.road_size, cfg.hidden_dim, topo_init)
            self.sem_emb = build_embedding_module(torch, cfg.road_size, cfg.hidden_dim, sem_init)

            self.topo_proj = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
            self.sem_proj = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)

            from dynapath.models import TPFusion, DynamicStateEncoder
            self.static_fusion = TPFusion(cfg.hidden_dim, cfg.hidden_dim, cfg.hidden_dim)
            self.dynamic_encoder = DynamicStateEncoder(
                cfg.dynamic_dim, cfg.hidden_dim, cfg.dropout
            )
            # Simple gate: only static_z + dynamic_z, no reliability
            self.gate = nn.Sequential(
                nn.Linear(cfg.hidden_dim * 2, cfg.hidden_dim),
                nn.GELU(),
                nn.Linear(cfg.hidden_dim, 1),
                nn.Sigmoid(),
            )
            self.dropout = nn.Dropout(cfg.dropout)
            from dynapath.models import MaskedMeanPool
            self.pool = MaskedMeanPool()

            if cfg.use_llm:
                from transformers.models.gpt2.modeling_gpt2 import GPT2Model
                self.backbone = GPT2Model.from_pretrained(
                    cfg.llm_name, output_attentions=False, output_hidden_states=False
                )
                if cfg.freeze_llm:
                    for n, p in self.backbone.named_parameters():
                        p.requires_grad = ("ln" in n) or ("wpe" in n)
            else:
                layer = nn.TransformerEncoderLayer(
                    d_model=cfg.hidden_dim, nhead=8,
                    dim_feedforward=cfg.hidden_dim * 4,
                    dropout=cfg.dropout, batch_first=True,
                )
                self.backbone = nn.TransformerEncoder(layer, num_layers=2)

            self.norm = nn.LayerNorm(cfg.hidden_dim)
            self.regressor = nn.Sequential(
                nn.Linear(cfg.hidden_dim, cfg.hidden_dim // 2),
                nn.GELU(), nn.Dropout(cfg.dropout),
                nn.Linear(cfg.hidden_dim // 2, 1),
            )

        def forward(self, road_ids, dynamic_x=None, attention_mask=None,
                    targets=None, return_loss=True):
            if attention_mask is None:
                attention_mask = road_ids.ne(self.pad_id)
            mask = attention_mask.to(dtype=torch.bool)

            topo = F.gelu(self.topo_proj(self.topo_emb(road_ids)))
            sem = F.gelu(self.sem_proj(self.sem_emb(road_ids)))
            static_z, _ = self.static_fusion(topo, sem)
            token_z = self.dropout(static_z)

            if dynamic_x is not None:
                dynamic_z = self.dynamic_encoder(dynamic_x)
                beta = self.gate(torch.cat([static_z, dynamic_z], dim=-1))
                token_z = (1.0 - beta) * static_z + beta * dynamic_z
                token_z = self.dropout(token_z)

            if self.config.use_llm:
                hidden = self.backbone(
                    inputs_embeds=token_z,
                    attention_mask=attention_mask.to(dtype=torch.long),
                ).last_hidden_state
            else:
                hidden = self.backbone(token_z, src_key_padding_mask=~mask)

            pooled = self.norm(self.pool(hidden, mask))
            pred = self.regressor(pooled).squeeze(-1)

            out = {"prediction": pred, "path_embedding": pooled}
            if return_loss and targets is not None:
                out["loss_tte"] = F.mse_loss(pred, targets.float())
                out["loss_ts_align"] = pred.sum() * 0.0
                out["loss_sd_align"] = pred.sum() * 0.0
                out["loss"] = out["loss_tte"]
            return out

    return SimpleGateDynaPath(config, topo_embeddings, semantic_embeddings)


def make_static_only_model(config, torch, topo_embeddings=None, semantic_embeddings=None):
    """Static-only baseline: TPfusion + backbone, no dynamic modality."""
    import torch.nn as nn
    import torch.nn.functional as F

    class StaticOnlyDynaPath(nn.Module):
        def __init__(self, cfg, topo_init=None, sem_init=None):
            super().__init__()
            self.config = cfg
            self.pad_id = cfg.road_size - 1 if cfg.pad_id is None else cfg.pad_id

            self.topo_emb = build_embedding_module(torch, cfg.road_size, cfg.hidden_dim, topo_init)
            self.sem_emb = build_embedding_module(torch, cfg.road_size, cfg.hidden_dim, sem_init)

            self.topo_proj = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
            self.sem_proj = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
            from dynapath.models import TPFusion
            self.static_fusion = TPFusion(cfg.hidden_dim, cfg.hidden_dim, cfg.hidden_dim)
            self.dropout = nn.Dropout(cfg.dropout)
            from dynapath.models import MaskedMeanPool
            self.pool = MaskedMeanPool()

            if cfg.use_llm:
                from transformers.models.gpt2.modeling_gpt2 import GPT2Model
                self.backbone = GPT2Model.from_pretrained(
                    cfg.llm_name, output_attentions=False, output_hidden_states=False
                )
                if cfg.freeze_llm:
                    for n, p in self.backbone.named_parameters():
                        p.requires_grad = ("ln" in n) or ("wpe" in n)
            else:
                layer = nn.TransformerEncoderLayer(
                    d_model=cfg.hidden_dim, nhead=8,
                    dim_feedforward=cfg.hidden_dim * 4,
                    dropout=cfg.dropout, batch_first=True,
                )
                self.backbone = nn.TransformerEncoder(layer, num_layers=2)

            self.norm = nn.LayerNorm(cfg.hidden_dim)
            self.regressor = nn.Sequential(
                nn.Linear(cfg.hidden_dim, cfg.hidden_dim // 2),
                nn.GELU(), nn.Dropout(cfg.dropout),
                nn.Linear(cfg.hidden_dim // 2, 1),
            )

        def forward(self, road_ids, dynamic_x=None, attention_mask=None,
                    targets=None, return_loss=True):
            if attention_mask is None:
                attention_mask = road_ids.ne(self.pad_id)
            mask = attention_mask.to(dtype=torch.bool)

            topo = F.gelu(self.topo_proj(self.topo_emb(road_ids)))
            sem = F.gelu(self.sem_proj(self.sem_emb(road_ids)))
            token_z, _ = self.static_fusion(topo, sem)
            token_z = self.dropout(token_z)

            if self.config.use_llm:
                hidden = self.backbone(
                    inputs_embeds=token_z,
                    attention_mask=attention_mask.to(dtype=torch.long),
                ).last_hidden_state
            else:
                hidden = self.backbone(token_z, src_key_padding_mask=~mask)

            pooled = self.norm(self.pool(hidden, mask))
            pred = self.regressor(pooled).squeeze(-1)

            out = {"prediction": pred, "path_embedding": pooled}
            if return_loss and targets is not None:
                out["loss_tte"] = F.mse_loss(pred, targets.float())
                out["loss_ts_align"] = pred.sum() * 0.0
                out["loss_sd_align"] = pred.sum() * 0.0
                out["loss"] = out["loss_tte"]
            return out

    return StaticOnlyDynaPath(config, topo_embeddings, semantic_embeddings)


# ---------------------------------------------------------------------------
# Variant registry
# ---------------------------------------------------------------------------

VARIANT_BUILDERS = {
    "full": make_full_model,
    "no_align": make_no_align_model,
    "no_sd_align": make_no_sd_align_model,
    "concat": make_concat_model,
    "simple_gate": make_simple_gate_model,
    "static_only": make_static_only_model,
}

VARIANT_DESCRIPTIONS = {
    "full": "Complete DynaPathLLM with TPfusion, DynamicEncoder, "
            "ReliabilityAwareFusion, and TS/SD alignment losses.",
    "no_align": "Full architecture without TS and SD alignment losses (lambda=0).",
    "no_sd_align": "Full architecture without SD alignment loss only.",
    "concat": "Simple concatenation fusion replacing reliability-aware gating.",
    "simple_gate": "Gated fusion without explicit reliability signal.",
    "static_only": "Static-only model (TPfusion + backbone, no dynamic modality).",
}


def train_one_variant(
    variant_name: str,
    config,
    device,
    torch,
    DataLoader,
    data_dir: str,
    output_dir: Path,
    batch_size: int,
    epochs: int,
    lr: float,
    num_workers: int,
    seed: int,
    topo_embeddings,
    semantic_embeddings,
) -> dict:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    train_loader, val_loader, test_loader, train_set, val_set, test_set = build_loaders(
        data_dir, batch_size, num_workers, DataLoader
    )

    builder = VARIANT_BUILDERS[variant_name]
    model = builder(
        config,
        torch,
        topo_embeddings=topo_embeddings,
        semantic_embeddings=semantic_embeddings,
    ).to(device)
    use_dynamic = variant_name != "static_only"

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=1e-3,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5,
    )

    history = []
    best_val_mae = float("inf")
    best_path = output_dir / f"best_{variant_name}.pt"
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        train_m = run_epoch(model, train_loader, optimizer, device, torch, train=True,
                            use_dynamic=use_dynamic)
        val_m = run_epoch(model, val_loader, optimizer, device, torch, train=False,
                          use_dynamic=use_dynamic)
        scheduler.step(val_m["mae"])
        history.append({"epoch": epoch, "train": train_m, "val": val_m})

        if val_m["mae"] < best_val_mae:
            best_val_mae = val_m["mae"]
            torch.save({"model": model.state_dict(), "variant": variant_name}, best_path)

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    test_m = run_epoch(model, test_loader, optimizer, device, torch, train=False,
                       use_dynamic=use_dynamic)

    return {
        "variant": variant_name,
        "description": VARIANT_DESCRIPTIONS.get(variant_name, ""),
        "use_llm": config.use_llm,
        "num_train": len(train_set),
        "num_val": len(val_set),
        "num_test": len(test_set),
        "hidden_dim": config.hidden_dim,
        "epochs": epochs,
        "best_epoch": min(history, key=lambda h: h["val"]["mae"])["epoch"],
        "best_val_mae": best_val_mae,
        "history": history,
        "test": test_m,
        "checkpoint": str(best_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DynaPathLLM variants")
    parser.add_argument("--data-dir", default="data/processed/pkdd15_grid_120k_clean")
    parser.add_argument("--output-dir", default="reports/dynapath_variants")
    parser.add_argument("--variant", default="full",
                        help="Comma-separated: full,no_align,no_sd_align,concat,simple_gate,static_only")
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=768)
    parser.add_argument("--lambda-ts", type=float, default=0.1)
    parser.add_argument("--lambda-sd", type=float, default=0.1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--topo-embeddings", default=None)
    parser.add_argument("--semantic-embeddings", default=None)
    args = parser.parse_args()

    torch, DataLoader = require_torch()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from dynapath.data import DynaPathNPYDataset
    train_set = DynaPathNPYDataset(args.data_dir, "train")
    topo_path = resolve_embedding_path(Path(args.data_dir), args.topo_embeddings, "topo")
    semantic_path = resolve_embedding_path(
        Path(args.data_dir), args.semantic_embeddings, "semantic"
    )
    topo_embeddings = load_optional_embedding(topo_path, torch)
    semantic_embeddings = load_optional_embedding(semantic_path, torch)

    from dynapath.models import DynaPathLLMConfig
    base_config = DynaPathLLMConfig(
        road_size=train_set.road_size,
        dynamic_dim=train_set.dynamic_x.shape[-1],
        hidden_dim=args.hidden_dim,
        use_llm=not args.no_llm,
        pad_id=train_set.pad_id,
        lambda_ts=args.lambda_ts,
        lambda_sd=args.lambda_sd,
    )

    variant_names = [v.strip() for v in args.variant.split(",") if v.strip()]
    for v in variant_names:
        if v not in VARIANT_BUILDERS:
            raise ValueError(f"Unknown variant: {v}. Choose from: {list(VARIANT_BUILDERS)}")

    print(f"Device: {device}")
    print(f"Road size: {train_set.road_size}, Dynamic dim: {train_set.dynamic_x.shape[-1]}")
    print(f"Use LLM: {not args.no_llm}")
    print(f"Training variants: {variant_names}")

    all_results = {
        "data_dir": args.data_dir,
        "topo_embeddings": topo_path,
        "semantic_embeddings": semantic_path,
        "use_llm": not args.no_llm,
        "results": [],
    }

    for vname in variant_names:
        print(f"\n{'='*60}")
        print(f"Training variant: {vname}")
        print(f"  {VARIANT_DESCRIPTIONS[vname]}")
        print(f"{'='*60}")

        # Each variant gets a fresh config copy
        import copy
        config = copy.deepcopy(base_config)

        result = train_one_variant(
            variant_name=vname,
            config=config,
            device=device,
            torch=torch,
            DataLoader=DataLoader,
            data_dir=args.data_dir,
            output_dir=output_dir,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.learning_rate,
            num_workers=args.num_workers,
            seed=args.seed,
            topo_embeddings=topo_embeddings,
            semantic_embeddings=semantic_embeddings,
        )
        all_results["results"].append(result)
        t = result["test"]
        print(f"  Test MAE: {t['mae']:.2f}s, RMSE: {t['rmse']:.2f}s, "
              f"MAPE: {t['mape']:.4f}, MARE: {t['mare']:.4f}")

    summary_path = output_dir / "variant_metrics.json"
    summary_path.write_text(
        json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nSaved summary to {summary_path}")

    # Print comparison table
    print("\n=== DynaPathLLM Variant Comparison ===")
    print(f"{'Variant':<16s} {'MAE/s':>8s} {'RMSE/s':>8s} {'MAPE':>8s} {'MARE':>8s}")
    print("-" * 52)
    for r in all_results["results"]:
        t = r["test"]
        print(f"{r['variant']:<16s} {t['mae']:>8.2f} {t['rmse']:>8.2f} "
              f"{t['mape']:>8.4f} {t['mare']:>8.4f}")


if __name__ == "__main__":
    main()
