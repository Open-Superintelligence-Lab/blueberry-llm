# Experiments

!!! info "Fixed Notation"
    - `d_model`: hidden size
    - `d_ff_expert`: expert FFN width
    - `E`: number of experts
    - `top_k`: experts activated per token

## Compute Proxies

- Active params (proxy FLOPs/forward): `2 * d_model * (top_k * d_ff_expert)`
- Total expert params: `2 * d_model * (E * d_ff_expert)`

Attention cost is held constant across the grid. These proxies are sufficient to match relative compute between configurations.

## Grids

Primary (Fixed FLOPs):

- Choose target `d_ff_dense` (e.g., 1536)
- For each `E`, set `top_k = min(2, E)` and `d_ff_expert = d_ff_dense / top_k`

Secondary (Fixed Params):

- Reference `(E_ref=8, d_ff_ref=768)`
- Set `d_ff_expert = (E_ref * d_ff_ref) / E`

## Training Budget

- Same tokens seen per run (e.g., 50M warmup; 250M final)
- 3 seeds per point (mean ± 95% CI reported in plots)

## Logged Metrics

- Core: validation loss/perplexity, train throughput (tokens/s), wall-clock
- Router: load-balance loss (via aux), entropy, usage histogram
- System: peak memory (optional), environment snapshot

