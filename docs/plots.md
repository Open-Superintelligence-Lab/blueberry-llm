# Plots (Named)

### Plot A1 — Perplexity vs Experts (Fixed FLOPs)
- y: validation perplexity
- x: `E`
- hue: seed (ribbon optional)

### Plot A2 — Learning Curves @Fixed FLOPs
- y: validation perplexity
- x: tokens seen
- hue: `E`

### Plot A4 — Throughput vs Experts (Fixed FLOPs)
- y: tokens/s
- x: `E`

### Plot A6 — Router Entropy vs Experts
- y: mean router entropy (per step)
- x: `E`

### Plot B1 — Perplexity vs Experts (Fixed Params)
- y: validation perplexity
- x: `E`

### Plot C1 — Perplexity vs FLOPs/Token (All Regimes)
- y: validation perplexity
- x: forward FLOPs/token proxy
- hue: `E`
- style: regime (`flops` vs `params`)

!!! info "Saved to"
    By default plots are written to `experiments/plots/`.

