# Blueberry LLM — MoE Scaling Laws

!!! tip "Purpose"
    Study whether more, narrower experts outperform fewer, wider experts at matched compute (FLOPs) and under matched parameter budgets.

This project adds a small, reproducible harness to run MoE sweeps, log metrics, and render publication-ready plots using Seaborn. It uses MkDocs Material for documentation.

## Hypothesis

> At the same per-token compute, more/narrower experts yield better perplexity than fewer/wider experts.

## Fairness Regimes

1. Fixed FLOPs (primary): keep `top_k * d_ff_expert` constant.
2. Fixed total expert parameters (secondary): keep `E * d_ff_expert` constant.

## Deliverables

- CSV logs with metrics (compute proxies, PPL, throughput, router stats)
- Plots A1–D2 (see Plots page)
- Reproducible commands and exact configs

## Repo Map (new)

- `experiments/compute.py`: compute proxies and helpers
- `experiments/run_scaling.py`: sweep runner (CLI)
- `experiments/train_eval.py`: compact train+eval loop
- `experiments/plotting.py`: Seaborn plots
- `mkdocs.yml` + `docs/`: this documentation

