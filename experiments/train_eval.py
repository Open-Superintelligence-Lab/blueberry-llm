from __future__ import annotations

import time
from dataclasses import asdict
from typing import Dict, Tuple

import torch

from llm import (
    MoEModelConfig,
    MoEMinimalLLM,
    TextTokenDataset,
    load_and_cache_data,
    evaluate_model,
)


def _gather_router_stats(model: torch.nn.Module) -> Dict[str, object]:
    """Collect simple router stats if the MoE layers track them.

    The llm.MixtureOfExperts layer may expose attributes when
    collect_stats=True. We aggregate across layers.
    """
    stats: Dict[str, object] = {}
    usage_sums = None
    entropy_sum = 0.0
    steps = 0
    experts = None

    for m in model.modules():
        if hasattr(m, "collect_stats") and getattr(m, "collect_stats"):
            if hasattr(m, "stats_usage_counts"):
                vec = getattr(m, "stats_usage_counts")
                if vec is not None:
                    v = vec.detach().cpu().to(torch.float64)
                    usage_sums = v if usage_sums is None else usage_sums + v
                    experts = v.numel()
            if hasattr(m, "stats_entropy_sum") and hasattr(m, "stats_steps"):
                entropy_sum += float(getattr(m, "stats_entropy_sum") or 0.0)
                steps += int(getattr(m, "stats_steps") or 0)

    if usage_sums is not None and usage_sums.sum().item() > 0:
        usage_frac = (usage_sums / usage_sums.sum()).tolist()
        stats["router_usage_frac"] = usage_frac
        stats["router_num_experts"] = experts
    if steps > 0:
        stats["router_entropy_mean"] = entropy_sum / steps
    return stats


def run_single_experiment(config: MoEModelConfig) -> Tuple[MoEMinimalLLM, Dict[str, object]]:
    """Run a short train+eval according to `config` and return metrics.

    Returns a trained model (for potential checkpointing) and a metrics dict
    suitable for CSV logging.
    """
    # Load data (tokenizer + tokens)
    texts, tokenizer, tokens = load_and_cache_data(config)
    dataset = TextTokenDataset(tokens, config.max_seq_len)

    # Split (10% val)
    val_size = max(1, len(dataset) // 10)
    train_size = max(1, len(dataset) - val_size)
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=2
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=2
    )

    # Build and train
    start_t = time.time()
    model = MoEMinimalLLM(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Minimal training loop: reuse evaluate_model and loss logic from llm.train_moe_model
    model.train()
    optim = torch.optim.AdamW(model.parameters(), lr=config.muon_lr * 0.1)
    steps = 0
    tokens_per_step = config.batch_size * config.max_seq_len
    for x, y in train_loader:
        if steps >= config.max_steps:
            break
        x, y = x.to(device), y.to(device)
        logits, aux_loss = model(x, return_aux_loss=True)
        ce = torch.nn.functional.cross_entropy(
            logits.view(-1, config.vocab_size), y.view(-1)
        )
        loss = ce + (aux_loss if aux_loss is not None else 0.0)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optim.step()
        steps += 1

    wall_s = time.time() - start_t
    tokens_seen = steps * tokens_per_step
    tps = tokens_seen / max(1e-6, wall_s)

    # Final eval
    final = evaluate_model(model, val_loader, config)

    # Compose metrics
    metrics: Dict[str, object] = {
        "d_model": config.d_model,
        "n_layers": config.n_layers,
        "n_heads": config.n_heads,
        "num_experts": config.num_experts,
        "expert_top_k": config.expert_top_k,
        "d_ff": config.d_ff,
        "batch_size": config.batch_size,
        "max_steps": config.max_steps,
        "seq_len": config.max_seq_len,
        "wall_time_s": wall_s,
        "tokens_seen": tokens_seen,
        "tokens_per_second": tps,
    }
    metrics.update(final)
    metrics.update(asdict(config))  # include config fields (safe dataclass fields)
    metrics.update(_gather_router_stats(model))
    return model, metrics

