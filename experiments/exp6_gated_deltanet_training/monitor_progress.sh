#!/bin/bash
# Real-time monitoring of LR ablation progress

clear
echo "╔════════════════════════════════════════════════════════════════════════╗"
echo "║         LR ABLATION STUDY - LIVE PROGRESS MONITOR                     ║"
echo "╚════════════════════════════════════════════════════════════════════════╝"
echo ""

# Get current experiment from log
CURRENT_EXP=$(grep -E "Experiment [0-9]/7" lr_ablation.log 2>/dev/null | tail -1)
CURRENT_LR=$(grep -E "Testing Learning Rate:" lr_ablation.log 2>/dev/null | tail -1)
LATEST_STEP=$(grep -E "Step [0-9]+/1000" lr_ablation.log 2>/dev/null | tail -1)
LATEST_EVAL=$(grep -E "Val Loss:" lr_ablation.log 2>/dev/null | tail -3)

echo "📊 CURRENT STATUS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "$CURRENT_EXP"
echo "$CURRENT_LR"
echo ""
echo "$LATEST_STEP"
echo ""

echo "📈 LATEST EVALUATION"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "$LATEST_EVAL"
echo ""

# Check if progress file exists
if [ -f "lr_ablation/lr_ablation_progress.json" ]; then
    echo "✅ COMPLETED EXPERIMENTS"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    python3 << 'PYTHON'
import json
try:
    with open('lr_ablation/lr_ablation_progress.json', 'r') as f:
        data = json.load(f)
    
    print(f"Progress: {data['completed_experiments']}/{data['total_experiments']} experiments")
    print()
    print(f"{'LR':<15} {'Best Val Loss':<15} {'Final Val Loss':<15}")
    print("-" * 45)
    
    for result in data['results_so_far']:
        lr = f"{result['lr']:.2e}"
        best = f"{result['best_val_loss']:.4f}"
        final = f"{result['final_val_loss']:.4f}" if result['final_val_loss'] else "N/A"
        print(f"{lr:<15} {best:<15} {final:<15}")
    
    # Find best so far
    best_result = min(data['results_so_far'], key=lambda x: x['best_val_loss'])
    print()
    print(f"🏆 Best so far: LR {best_result['lr']:.2e} with loss {best_result['best_val_loss']:.4f}")
except:
    pass
PYTHON
    echo ""
fi

# Check for plots
if [ -f "lr_ablation/lr_ablation_comparison_partial.png" ]; then
    echo "🖼️  PLOTS AVAILABLE"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    ls -lh lr_ablation/*.png 2>/dev/null | awk '{print "  📊 " $9 " (" $5 ")"}'
    echo ""
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "💡 Refresh this view: bash monitor_progress.sh"
echo "📝 Watch live log: tail -f lr_ablation.log"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

