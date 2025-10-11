#!/bin/bash
# Live monitor for 10k training run

echo "Starting 10k training monitor (updates every 5 seconds)..."
echo "Press Ctrl+C to stop"
echo ""
sleep 2

while true; do
    clear
    echo "╔════════════════════════════════════════════════════════════════════════╗"
    echo "║            GATED DELTANET - 10K TRAINING MONITOR (H100)               ║"
    echo "╚════════════════════════════════════════════════════════════════════════╝"
    echo ""
    
    # Extract latest metrics
    LATEST_STEP=$(grep -E "Step [0-9]+/10000" training_10k.log 2>/dev/null | tail -1)
    LATEST_EVAL=$(grep -E "Val Loss:" training_10k.log 2>/dev/null | tail -1)
    LATEST_ACC=$(grep -E "Val Accuracy:" training_10k.log 2>/dev/null | tail -1)
    LATEST_PPL=$(grep -E "Val Perplexity:" training_10k.log 2>/dev/null | tail -1)
    BEST_MODEL=$(grep "New best validation loss:" training_10k.log 2>/dev/null | tail -1)
    
    # Calculate progress
    CURRENT_STEP=$(echo "$LATEST_STEP" | grep -oP 'Step \K[0-9]+' | head -1)
    if [ -n "$CURRENT_STEP" ]; then
        PROGRESS=$((CURRENT_STEP * 100 / 10000))
        ELAPSED_MIN=$((CURRENT_STEP / 156))  # ~2.6 steps/s = ~156 steps/min
        REMAINING_MIN=$(((10000 - CURRENT_STEP) / 156))
    else
        PROGRESS=0
        ELAPSED_MIN=0
        REMAINING_MIN=65
    fi
    
    echo "📊 TRAINING PROGRESS"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Step: ${CURRENT_STEP:-0}/10000 (${PROGRESS}%)"
    echo "Time: ${ELAPSED_MIN}m elapsed, ~${REMAINING_MIN}m remaining"
    echo ""
    echo "$LATEST_STEP"
    echo ""
    
    echo "📈 LATEST EVALUATION"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    if [ -n "$LATEST_EVAL" ]; then
        echo "$LATEST_EVAL"
        echo "$LATEST_ACC"
        echo "$LATEST_PPL"
    else
        echo "Waiting for first evaluation (step 100)..."
    fi
    echo ""
    
    if [ -n "$BEST_MODEL" ]; then
        echo "🏆 BEST MODEL"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "$BEST_MODEL"
        echo ""
    fi
    
    echo "🖥️  GPU STATUS"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits 2>/dev/null | \
    awk -F', ' '{printf "  GPU %s: %s\n  └─ %d%% util | %dMB/%dMB (%.1f%%) | %d°C\n", $1, $2, $3, $4, $5, ($4/$5)*100, $6}'
    
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "⏰ $(date '+%Y-%m-%d %H:%M:%S') | Refreshing every 5s"
    echo "💾 Checkpoints: checkpoints/ | Logs: training_10k.log"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    sleep 5
done

