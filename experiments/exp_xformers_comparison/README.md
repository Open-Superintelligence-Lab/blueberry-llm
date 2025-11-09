# XFormers Attention Comparison Experiment

This experiment compares the performance of xformers memory-efficient attention vs standard PyTorch scaled_dot_product_attention on the smollm dataset.

## Experiment Setup

- **Dataset**: HuggingFaceTB/smollm-corpus (cosmopedia-v2)
- **Tokenizer**: HuggingFaceTB/SmolLM-135M
- **Model**: MoE Transformer with current configuration from `configs/moe_config.py`

## Experiments

1. **Standard Attention**: Uses PyTorch `F.scaled_dot_product_attention`
2. **XFormers Attention**: Uses xformers `memory_efficient_attention`

## Metrics Compared

- Training time (seconds)
- Validation loss
- Validation accuracy
- Validation perplexity

## Running the Experiment

```bash
cd /root/blueberry-llm
python experiments/exp_xformers_comparison/compare_attention.py
```

## Results

Results will be saved to `experiments/exp_xformers_comparison/results/comparison_results.json`

