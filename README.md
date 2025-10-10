# Blueberry LLM

**Open Superintelligence Lab** - Open research for everyone. We publish all of our research for the sake of accelerating science. Learn real AI research from a real research lab.

## Quick Start

```bash
pip install -r requirements.txt

python train_moe.py
```

## About

Purpose of this repository is to research better, faster, smarter LLMs.

This repository contains cutting-edge language model experiments and architectures. We believe scientists do their best work when given freedom to explore, so this is a space for your independent research and discovery.

Fork this repository, create a new experiment in `experiments/` folder, then create a pull request to merge it back.

## Experiments

> Each experiment below is validated on a specific git tag. 
> Later commits may introduce breaking changes. 
> To run an experiment with correct version of the repo, checkout its validated tag using: `git checkout <tag-name>`

| Experiment | Researcher | Validated Tag | Research Question | Key Findings |
|------------|-----------|---------------|-------------------|--------------|
| *Your experiments will be added here* | | | | |
| [Exp3: PLASA + GDN Hybrid](experiments/exp3_plasa_gdn_hybrid/) | Overtimepog [GitHub](https://github.com/overtimepog) | `git checkout experiments-v1.0` | 1. Can per-layer adaptive sparse attention (PLASA) with progressive sparsity scheduling improve upon the uniform sparse attention tested in Exp1? <br><br> 2. Does the PROGRESSIVE_SPARSE schedule align with transformer layer hierarchy (dense early layers, aggressive sparse middle layers, moderate sparse late layers)? <br><br> 3. Which combination produces the best efficiency-performance tradeoff across 11 patterns (pure architectures + PLASA hybrids + Original hybrids)? | **🏆 Full PLASA architecture (all 4 layers) achieves best results across all metrics!** <br><br> 1. **Full PLASA dominates**: Val Loss 4.30 (33.9% better than Exp1 DSA), Accuracy 51.69% (154.7% improvement), Perplexity 73.81, Training Time 35.5s (74% faster than hybrids). All 5 PLASA patterns occupy top 5 ranks. <br><br> 2. **Progressive sparsity validated**: Dense→Aggressive→Moderate schedule (k=L, k=L/4, k=L/2) confirms middle layer redundancy hypothesis. PLASA achieves 17.7% better average performance than full attention baseline across all patterns. <br><br> 3. **Pure PLASA optimal**: P→P→P→P outperforms all hybrid configurations. 18.4% lower loss and 39.4% higher accuracy vs best full attention, with minimal parameter overhead (+0.3%) and massive speed advantage. |
| [Exp1: DSA + GDN Hybrid](experiments/exp1_dsa_gdn_hybrid/) | Vuk Rosić [YouTube](https://www.youtube.com/channel/UC7XJj9pv_11a11FUxCMz15g) [GitHub](https://github.com/vukrosic) | `git checkout experiments-v1.0` | 1. Can replacing full attention with DeepSeek Sparse Attention (DSA) improve the efficiency and performance of a hybrid attention architecture that combines full attention and Gated DeltaNet (GDN)? <br><br> 2. Which combination of attention mechanisms across layers produces the best efficiency-performance tradeoff: (1) Full Attention + GDN, (2) DSA + GDN, (3) DSA only, or (4) Full Attention only? |1. Trains faster in the beginning, but full attention seems to surpass it with more training. Future work is to investigate this further. <br><br> 2. Currently L → F → F → L (Gated Deltanet → Full Attention → Full Attention → Gated Deltanet). Future work is to investigate this further. |

## Getting Started

1. **Fork this repository** - Click the "Fork" button at the top right of this page to create your own copy
2. Clone your fork: `git clone https://github.com/YOUR-USERNAME/blueberry-llm.git`
3. Install dependencies: `pip install -r requirements.txt`
4. Read `CONTRIBUTING.md` for contribution guidelines
5. Create your own experiment and merge it
6. Explore the `experiments/` folder for ongoing research and inspiration
7. Once you finish with your research, create a pull request to merge it back to this repo

## Philosophy

We don't prescribe what to research. Instead, we provide:
- Freedom to explore interesting ideas
- Infrastructure to test hypotheses
- A collaborative environment for learning

## Structure

- **`experiments/`** - Research experiments with their own documentation
- **`models/`** - Model architectures and implementations (DeepSeek, Qwen3-Next)
- **`training/`** - Training scripts and utilities
- **`configs/`** - Configuration files

## Contributing

See `CONTRIBUTING.md` for guidelines on how to contribute to this project.