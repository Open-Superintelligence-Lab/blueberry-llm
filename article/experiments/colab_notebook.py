#!/usr/bin/env python3
"""
Google Colab Notebook for AMP vs FP32 Experiments
Copy and paste this into a Colab cell to run the experiments
"""

# Cell 1: Setup and Installation
"""
# First, upload the blueberry-llm repository to Colab
# You can either:
# 1. Upload the entire blueberry-llm folder to Colab
# 2. Clone from GitHub: !git clone https://github.com/Open-Superintelligence-Lab/blueberry-llm.git

# Then run the setup script
!python blueberry-llm/article/experiments/setup_colab.py
"""

# Cell 2: Import and Setup
"""
import os
import sys
import torch
import matplotlib.pyplot as plt

# Check GPU availability
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# Add the experiment directory to path
sys.path.append('/content/blueberry-llm/article/experiments')
"""

# Cell 3: Quick Test (10 steps each)
"""
# Change to the project root directory
import os
os.chdir('/content/blueberry-llm')

# Run quick test to verify everything works
!python article/experiments/quick_test.py
"""

# Cell 4: Full Experiments (1000 steps each)
"""
# Run full experiments
!python article/experiments/amp_experiment_runner.py --full
"""

# Cell 5: Analyze Results
"""
# Analyze and visualize results
from analyze_results import load_results, print_detailed_results, create_performance_comparison, create_amp_benefit_analysis

# Load the results
results = load_results('amp_experiment_results_full_YYYYMMDD_HHMMSS.json')  # Replace with actual filename

# Print detailed results
print_detailed_results(results)

# Create visualizations
create_performance_comparison(results)
create_amp_benefit_analysis(results)

print("📊 Analysis complete! Check the generated PNG files.")
"""

# Cell 6: Key Findings Summary
"""
# Print key findings
import pandas as pd

df = pd.DataFrame(results)
df_success = df[df['success'] == True].copy()

if len(df_success) > 0:
    # Calculate AMP benefits
    benefit_data = []
    for model_size in df_success['config'].apply(lambda x: f"{x['d_model']}d×{x['n_layers']}L").unique():
        for batch_size in df_success['config'].apply(lambda x: x['batch_size']).unique():
            fp32_data = df_success[
                (df_success['config'].apply(lambda x: f"{x['d_model']}d×{x['n_layers']}L") == model_size) & 
                (df_success['config'].apply(lambda x: x['batch_size']) == batch_size) & 
                (df_success['config'].apply(lambda x: x['use_amp']) == False)
            ]
            amp_data = df_success[
                (df_success['config'].apply(lambda x: f"{x['d_model']}d×{x['n_layers']}L") == model_size) & 
                (df_success['config'].apply(lambda x: x['batch_size']) == batch_size) & 
                (df_success['config'].apply(lambda x: x['use_amp']) == True)
            ]
            
            if len(fp32_data) > 0 and len(amp_data) > 0:
                fp32_speed = fp32_data['tokens_per_second'].iloc[0]
                amp_speed = amp_data['tokens_per_second'].iloc[0]
                speed_improvement = (amp_speed - fp32_speed) / fp32_speed * 100
                
                benefit_data.append({
                    'model_size': model_size,
                    'batch_size': batch_size,
                    'speed_improvement': speed_improvement,
                    'amp_better': speed_improvement > 0
                })
    
    benefit_df = pd.DataFrame(benefit_data)
    
    print("🎯 KEY FINDINGS:")
    print("=" * 50)
    print(f"Experiments where AMP is faster: {sum(benefit_df['amp_better'])}/{len(benefit_df)}")
    
    if len(benefit_df) > 0:
        best_amp = benefit_df.loc[benefit_df['speed_improvement'].idxmax()]
        worst_amp = benefit_df.loc[benefit_df['speed_improvement'].idxmin()]
        
        print(f"\nBest AMP improvement: {best_amp['speed_improvement']:.1f}% speedup")
        print(f"Configuration: {best_amp['model_size']}, batch={best_amp['batch_size']}")
        
        print(f"\nWorst AMP performance: {worst_amp['speed_improvement']:.1f}% change")
        print(f"Configuration: {worst_amp['model_size']}, batch={worst_amp['batch_size']}")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        amp_better = benefit_df[benefit_df['amp_better']]
        if len(amp_better) > 0:
            print("✅ Use AMP for:")
            for _, row in amp_better.iterrows():
                print(f"   - {row['model_size']} models with batch size {row['batch_size']}")
        
        amp_worse = benefit_df[~benefit_df['amp_better']]
        if len(amp_worse) > 0:
            print("❌ Avoid AMP for:")
            for _, row in amp_worse.iterrows():
                print(f"   - {row['model_size']} models with batch size {row['batch_size']}")
"""

# Instructions for Colab usage:
"""
INSTRUCTIONS FOR GOOGLE COLAB:

1. Upload the blueberry-llm repository to Colab
2. Copy each cell above into separate Colab cells
3. Run Cell 1 (Setup) first
4. Run Cell 2 (Import) to check GPU
5. Run Cell 3 (Quick Test) to verify everything works
6. If quick test passes, run Cell 4 (Full Experiments)
7. Run Cell 5 (Analysis) to generate charts
8. Run Cell 6 (Findings) to see key insights

Expected runtime:
- Quick test: 5-10 minutes
- Full experiments: 30-60 minutes
- Analysis: 1-2 minutes

The experiments will generate:
- JSON result files
- Performance comparison charts
- AMP benefit analysis heatmaps
- Detailed results tables
"""
