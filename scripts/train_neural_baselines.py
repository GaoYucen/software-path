#!/usr/bin/env python3
"""Train neural network baselines on PKDD15 grid-token data.

Trains and evaluates LSTM, Transformer, and PathLLM-Static models.
Designed for reproducibility: chronological split, fixed seed, no label leakage.

Usage:
    # Quick debug (no GPT-2 required):
    python scripts/train_neural_baselines.py \\
      --data-dir data/processed/pkdd15_grid_120k_clean \\
      --no-llm --epochs 5 --batch-size 16

    # Full run with all baselines:
    python scripts/train_neural_baselines.py \\
      --data-dir data/processed/pkdd15_grid_120k_clean \\
      --epochs 20 --batch-size 16 --models lstm,transformer,pathllm
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
        raise SystemExit(
            "PyTorch is required. Install torch before running this script."
        ) from exc
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
        train_set,
        val_set,
        test_set,
    )


def run_epoch(model, loader, optimizer, device, torch, train: bool) -> dict:
    model.train(train)
    total_loss = 0.0
    count = 0
    preds, trues = [], []

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.set_grad_enabled(train):
            try:
                out = model(
                    road_ids=batch["road_ids"],
                    dynamic_x=batch["dynamic_x"],
                    attention_mask=batch["attention_mask"],
                )
            except TypeError:
                out = model(
                    road_ids=batch["road_ids"],
                    attention_mask=batch["attention_mask"],
                )

            loss = torch.nn.functional.mse_loss(out["prediction"], batch["targets"])
            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        bs = batch["targets"].size(0)
        count += bs
        total_loss += float(loss.detach().cpu()) * bs
        preds.extend(out["prediction"].detach().cpu().numpy().tolist())
        trues.extend(batch["targets"].detach().cpu().numpy().tolist())

    result = {"loss": total_loss / max(count, 1)}
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


def make_model(
    name: str,
    road_size: int,
    pad_id: int,
    hidden_dim: int,
    use_llm: bool,
    torch,
    topo_embeddings=None,
    semantic_embeddings=None,
):
    """Factory: create a neural baseline model by name."""
    from dynapath.baselines import (
        LSTMTTE,
        PathLLMStatic,
        PathLLMStaticConfig,
        TransformerTTE,
    )

    name = name.lower().strip()
    if name == "lstm":
        return LSTMTTE(
            road_size=road_size,
            hidden_dim=min(hidden_dim, 512),
            num_layers=2,
            dropout=0.2,
            pad_id=pad_id,
        )
    elif name == "transformer":
        return TransformerTTE(
            road_size=road_size,
            hidden_dim=min(hidden_dim, 512),
            nhead=8,
            num_layers=2,
            dropout=0.1,
            pad_id=pad_id,
        )
    elif name in ("pathllm", "pathllm_static"):
        config = PathLLMStaticConfig(
            road_size=road_size,
            hidden_dim=hidden_dim,
            use_llm=use_llm,
            pad_id=pad_id,
        )
        return PathLLMStatic(
            config,
            topo_embeddings=topo_embeddings,
            semantic_embeddings=semantic_embeddings,
        )
    else:
        raise ValueError(f"Unknown model name: {name}")


def train_one_model(
    model_name: str,
    road_size: int,
    pad_id: int,
    hidden_dim: int,
    use_llm: bool,
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

    model = make_model(
        model_name,
        road_size,
        pad_id,
        hidden_dim,
        use_llm,
        torch,
        topo_embeddings=topo_embeddings,
        semantic_embeddings=semantic_embeddings,
    ).to(device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=1e-3,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    history = []
    best_val_mae = float("inf")
    best_path = output_dir / f"best_{model_name}.pt"
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, device, torch, train=True)
        val_metrics = run_epoch(model, val_loader, optimizer, device, torch, train=False)
        scheduler.step(val_metrics["mae"])
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})

        if val_metrics["mae"] < best_val_mae:
            best_val_mae = val_metrics["mae"]
            torch.save(
                {"model": model.state_dict(), "model_name": model_name},
                best_path,
            )

    # Restore best and evaluate on test
    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    test_metrics = run_epoch(model, test_loader, optimizer, device, torch, train=False)

    result = {
        "model": model_name,
        "use_llm": use_llm,
        "num_train": len(train_set),
        "num_val": len(val_set),
        "num_test": len(test_set),
        "hidden_dim": hidden_dim,
        "epochs": epochs,
        "best_epoch": min(
            history, key=lambda h: h["val"]["mae"]
        )["epoch"],
        "best_val_mae": best_val_mae,
        "history": history,
        "test": test_metrics,
        "checkpoint": str(best_path),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train neural TTE baselines")
    parser.add_argument("--data-dir", default="data/processed/pkdd15_grid_120k_clean")
    parser.add_argument("--output-dir", default="reports/neural_baselines")
    parser.add_argument("--models", default="lstm,transformer,pathllm",
                        help="Comma-separated: lstm,transformer,pathllm")
    parser.add_argument("--no-llm", action="store_true",
                        help="Use small Transformer instead of GPT-2")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=768)
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

    # Get road_size and pad_id from a dummy dataset
    from dynapath.data import DynaPathNPYDataset
    train_set = DynaPathNPYDataset(args.data_dir, "train")
    road_size = train_set.road_size
    pad_id = train_set.pad_id
    topo_path = resolve_embedding_path(Path(args.data_dir), args.topo_embeddings, "topo")
    semantic_path = resolve_embedding_path(
        Path(args.data_dir), args.semantic_embeddings, "semantic"
    )
    topo_embeddings = load_optional_embedding(topo_path, torch)
    semantic_embeddings = load_optional_embedding(semantic_path, torch)

    use_llm = not args.no_llm
    model_names = [m.strip() for m in args.models.split(",") if m.strip()]

    print(f"Device: {device}")
    print(f"Road size: {road_size}, Pad ID: {pad_id}")
    print(f"Use LLM: {use_llm}")
    print(f"Training models: {model_names}")

    all_results = {
        "data_dir": args.data_dir,
        "topo_embeddings": topo_path,
        "semantic_embeddings": semantic_path,
        "use_llm": use_llm,
        "results": [],
    }

    for mname in model_names:
        print(f"\n{'='*60}")
        print(f"Training: {mname}")
        print(f"{'='*60}")
        result = train_one_model(
            model_name=mname,
            road_size=road_size,
            pad_id=pad_id,
            hidden_dim=args.hidden_dim,
            use_llm=use_llm,
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
        print(f"  Test MAE: {result['test']['mae']:.2f}s, RMSE: {result['test']['rmse']:.2f}s")

    # Save summary
    summary_path = output_dir / "baseline_metrics.json"
    summary_path.write_text(json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved summary to {summary_path}")

    # Print comparison table
    print("\n=== Neural Baseline Comparison ===")
    print(f"{'Model':<20s} {'MAE/s':>8s} {'RMSE/s':>8s} {'MAPE':>8s} {'MARE':>8s}")
    print("-" * 56)
    for r in all_results["results"]:
        t = r["test"]
        print(f"{r['model']:<20s} {t['mae']:>8.2f} {t['rmse']:>8.2f} "
              f"{t['mape']:>8.4f} {t['mare']:>8.4f}")


if __name__ == "__main__":
    main()
