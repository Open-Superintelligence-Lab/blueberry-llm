from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Iterable, List

import torch

from llm import MoEModelConfig
from experiments.compute import (
    ComputeBudget,
    d_ff_expert_for_fixed_active,
    d_ff_expert_for_fixed_total_params,
)
from experiments.logging_utils import append_row_csv
from experiments.train_eval import run_single_experiment


def parse_experts_list(text: str) -> List[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def build_config(
    regime: str,
    d_model: int,
    n_heads: int,
    n_layers: int,
    d_ff_dense: int,
    e: int,
    top_k: int,
    base_batch: int,
    steps: int,
    seq_len: int,
    collect_router_stats: bool,
) -> MoEModelConfig:
    top_k = min(top_k, e)
    if regime == "flops":
        d_ff_exp = d_ff_expert_for_fixed_active(d_ff_dense, top_k)
    elif regime == "params":
        d_ff_exp = d_ff_expert_for_fixed_total_params(d_ff_ref=d_ff_dense, e_ref=8, e_target=e)
    else:
        raise ValueError("regime must be 'flops' or 'params'")

    cfg = MoEModelConfig(
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff_exp,
        batch_size=base_batch,
        max_steps=steps,
        max_seq_len=seq_len,
        num_experts=e,
        expert_top_k=top_k,
        use_amp=True,
    )
    # Track router stats if requested
    setattr(cfg, "collect_router_stats", bool(collect_router_stats))
    return cfg


def run_grid(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"results_{args.regime}.csv"

    experts = parse_experts_list(args.experts)
    for e in experts:
        for s in range(args.seeds):
            cfg = build_config(
                regime=args.regime,
                d_model=args.d_model,
                n_heads=args.n_heads,
                n_layers=args.n_layers,
                d_ff_dense=args.d_ff_dense,
                e=e,
                top_k=args.top_k,
                base_batch=args.batch_size,
                steps=args.max_steps,
                seq_len=args.seq_len,
                collect_router_stats=args.collect_router_stats,
            )

            # Important: vocab_size resolved by data loader; run experiment
            t0 = time.time()
            model, metrics = run_single_experiment(cfg)
            metrics.update({
                "seed": s,
                "regime": args.regime,
                "timestamp": int(t0),
            })
            # Compute proxies
            budget = ComputeBudget(
                d_model=cfg.d_model,
                top_k=cfg.expert_top_k,
                d_ff_expert=cfg.d_ff,
                num_experts=cfg.num_experts,
            )
            metrics.update(budget.as_dict())

            append_row_csv(csv_path, metrics)


def main() -> None:
    p = argparse.ArgumentParser("Run MoE scaling experiments")
    p.add_argument("--regime", choices=["flops", "params"], default="flops")
    p.add_argument("--experts", default="1,2,4,8,16,32", help="Comma-separated expert counts")
    p.add_argument("--top-k", dest="top_k", type=int, default=2)
    p.add_argument("--d-model", dest="d_model", type=int, default=384)
    p.add_argument("--n-heads", dest="n_heads", type=int, default=8)
    p.add_argument("--n-layers", dest="n_layers", type=int, default=6)
    p.add_argument("--d-ff-dense", dest="d_ff_dense", type=int, default=1536)
    p.add_argument("--seq-len", dest="seq_len", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--out-dir", default="experiments/out")
    p.add_argument("--collect-router-stats", action="store_true")
    args = p.parse_args()
    run_grid(args)


if __name__ == "__main__":
    main()

