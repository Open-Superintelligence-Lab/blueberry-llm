Goal is to release LLM for $100

1. Make sure experiments are easily run on more compute


Why does Qwen3-Next do this:
1. Adopt the output gating mechanism from our prior work [2] to reduce low-rank issues in attention.
2. Increase the dimension per attention head from 128 to 256.
3. Apply rotary position encoding only to the first 25% of position dimensions — improving extrapolation to longer sequences.


# MoE





Note:
- The efficiency or throughput improvement depends highly on the implementation.





Do we need to try different random seeds? That would 3x number of experiments.