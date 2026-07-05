#!/usr/bin/env python3
"""Train a Path-LLM-style DynaPath model on processed PKDD15 arrays.

This script requires PyTorch. Default mode uses GPT-2 through HuggingFace
Transformers and passes fused path embeddings via ``inputs_embeds``. Use
``--no-llm`` only for local shape/debug runs when GPT-2 is unavailable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def require_torch():
    try:
        import torch
        from torch.utils.data import DataLoader
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            "PyTorch is required for train_dynapath_llm.py. "
            "Install torch and transformers before running this script."
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
        "mare": float(err.sum() / np.maximum(y_true.sum(), 1e-6)),
    }


def load_optional_embedding(path: str | None, torch):
    if path is None:
        return None
    return torch.tensor(np.load(path), dtype=torch.float32)


def run_epoch(model, loader, optimizer, device, torch, train: bool) -> dict[str, float]:
    model.train(train)
    total = {"loss": 0.0, "loss_tte": 0.0, "loss_ts_align": 0.0, "loss_sd_align": 0.0}
    count = 0
    preds = []
    trues = []

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.set_grad_enabled(train):
            out = model(
                road_ids=batch["road_ids"],
                dynamic_x=batch["dynamic_x"],
                attention_mask=batch["attention_mask"],
                targets=batch["targets"],
            )
            if train:
                optimizer.zero_grad()
                out["loss"].backward()
                optimizer.step()

        bs = batch["targets"].size(0)
        count += bs
        for key in total:
            total[key] += float(out[key].detach().cpu()) * bs
        preds.extend(out["prediction"].detach().cpu().numpy().tolist())
        trues.extend(batch["targets"].detach().cpu().numpy().tolist())

    metrics = {key: value / max(count, 1) for key, value in total.items()}
    metrics.update(regression_metrics(trues, preds))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/processed/pkdd15_grid_120k_clean")
    parser.add_argument("--output-dir", default="reports/dynapath_llm_debug")
    parser.add_argument("--topo-embeddings", default=None)
    parser.add_argument("--semantic-embeddings", default=None)
    parser.add_argument("--llm-name", default="gpt2")
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=768)
    parser.add_argument("--lambda-ts", type=float, default=0.1)
    parser.add_argument("--lambda-sd", type=float, default=0.1)
    parser.add_argument("--target-scale", type=float, default=1.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    torch, DataLoader = require_torch()

    from dynapath.data import DynaPathNPYDataset, dynapath_collate
    from dynapath.models import DynaPathLLM, DynaPathLLMConfig

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_set = DynaPathNPYDataset(args.data_dir, "train", target_scale=args.target_scale)
    val_set = DynaPathNPYDataset(args.data_dir, "val", target_scale=args.target_scale)
    test_set = DynaPathNPYDataset(args.data_dir, "test", target_scale=args.target_scale)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=dynapath_collate,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=dynapath_collate,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=dynapath_collate,
    )

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    topo_embeddings = load_optional_embedding(args.topo_embeddings, torch)
    semantic_embeddings = load_optional_embedding(args.semantic_embeddings, torch)
    config = DynaPathLLMConfig(
        road_size=train_set.road_size,
        dynamic_dim=train_set.dynamic_x.shape[-1],
        hidden_dim=args.hidden_dim,
        llm_name=args.llm_name,
        use_llm=not args.no_llm,
        pad_id=train_set.pad_id,
        lambda_ts=args.lambda_ts,
        lambda_sd=args.lambda_sd,
    )
    model = DynaPathLLM(config, topo_embeddings, semantic_embeddings).to(device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        weight_decay=1e-3,
    )

    history = []
    best_val = float("inf")
    best_path = out_dir / "best_dynapath_llm.pt"
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, device, torch, train=True)
        val_metrics = run_epoch(model, val_loader, optimizer, device, torch, train=False)
        item = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(item)
        print(json.dumps(item, indent=2))
        if val_metrics["mae"] < best_val:
            best_val = val_metrics["mae"]
            torch.save({"model": model.state_dict(), "config": config.__dict__}, best_path)

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    test_metrics = run_epoch(model, test_loader, optimizer, device, torch, train=False)
    report = {
        "status": "pass",
        "data_dir": args.data_dir,
        "output_dir": str(out_dir),
        "use_llm": not args.no_llm,
        "llm_name": args.llm_name,
        "target_scale": args.target_scale,
        "num_train": len(train_set),
        "num_val": len(val_set),
        "num_test": len(test_set),
        "history": history,
        "test": test_metrics,
        "checkpoint": str(best_path),
    }
    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
