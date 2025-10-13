# Final Results: Baseline vs Reasoning (1000 steps)

## Winner: Baseline MoE (15.9% better)

### Best Baseline MoE
**Configuration:** LR=6e-4, warmup=0, MoE only

**Results:**
- ✅ **Val Loss:** 4.89
- ✅ **Perplexity:** 133
- ✅ **Accuracy:** 27.23%
- ⏱️ **Training Time:** 2.1 min
- 🚀 **Speed:** 8.3 steps/s

### Best Recursive Reasoning
**Configuration:** LR=3e-4, warmup=100, H=1 L=1 (minimal)

**Results:**
- 📊 **Val Loss:** 5.81
- 📊 **Perplexity:** 335
- 📊 **Accuracy:** 20.28%
- ⏱️ **Training Time:** 3.2 min
- 🚀 **Speed:** 5.3 steps/s

## Comparison

| Metric | Baseline | Reasoning | Advantage |
|--------|----------|-----------|-----------|
| Val Loss | **4.89** | 5.81 | 15.9% better |
| Perplexity | **133** | 335 | 60.3% better |
| Accuracy | **27.23%** | 20.28% | 34.3% better |
| Speed | **8.3/s** | 5.3/s | 56.6% faster |
| Time | **2.1m** | 3.2m | 34.4% faster |

## Reasoning Ablations (15 configs)

### Architectural Variations
1. **R01_baseline** - H=2, L=2: 7.64 val loss
2. **R02_minimal** - H=1, L=1: **7.59** ✅ (best)
3. **R03_deep** - H=3, L=3: 7.64
4. **R04_more_H** - H=3, L=1: 7.63
5. **R05_more_L** - H=1, L=3: 7.63
6. **R14_asym_H** - H=2, L=1: 7.64
7. **R15_asym_L** - H=1, L=2: 7.63

### ACT Settings
8. **R06_no_act** - No halting: 7.64
9. **R07_high_explore** - Explore 30%: 7.64
10. **R08_low_explore** - Explore 1%: 7.64
11. **R09_extended_halt** - Max 7 steps: 7.63

### Learning Rates
12. **R10_low_lr** - LR=1e-4: 8.45 (worst)
13. **R11_high_lr** - LR=6e-4: 7.63
14. **R12_med_lr** - LR=2e-4: 7.64
15. **R13_optimized** - No dropout + high LR: 7.63

**Best:** R02_minimal (7.59 val loss)  
**Worst:** R10_low_lr (8.45 val loss)  
**Range:** 11.3% difference

### Key Findings

1. **Minimal recursion wins** - H=1, L=1 beats deeper configs
2. **ACT settings don't matter much** - Similar performance regardless
3. **Learning rate critical** - Too low (1e-4) significantly hurts
4. **Simpler is better** - For limited compute, minimal recursion optimal

## Conclusion

**For 1000-step training:**
- Baseline MoE wins decisively (16% better loss)
- 60% faster training without recursive overhead
- Recursive models need 2000+ steps to potentially catch up

**Recommendation:** Use baseline MoE for quick experiments. Consider recursive reasoning only for extensive training (5000+ steps).
