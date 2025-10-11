#!/bin/bash
# Watch the LR ablation progress live with auto-refresh

echo "Starting live monitor (updates every 3 seconds)..."
echo "Press Ctrl+C to stop"
echo ""
sleep 2

while true; do
    clear
    bash monitor_progress.sh
    
    # Show GPU status
    echo ""
    echo "🖥️  GPU STATUS"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits | \
    awk -F', ' '{printf "  GPU %s: %s | %d%% util | %dMB/%dMB (%.1f%%) | %d°C\n", $1, $2, $3, $4, $5, ($4/$5)*100, $6}'
    
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "⏰ $(date '+%Y-%m-%d %H:%M:%S') | Auto-refreshing every 3s | Ctrl+C to stop"
    
    sleep 3
done

