#!/bin/bash
# Quick runner for MoE ablations

case "$1" in
  plot)
    python plot_results.py
    ;;
  all)
    python ablation_batch_vs_seqlen.py
    ;;
  custom)
    # Example: ./run.sh custom 32 512 0.015 2 100
    python ablation_batch_vs_seqlen.py --batch $2 --seqlen $3 --lr $4 --grad-accum ${5:-1} --steps ${6:-20} --name "custom_$2x$3_lr$4"
    ;;
  *)
    echo "Usage:"
    echo "  ./run.sh all                                    # Run all ablations"
    echo "  ./run.sh plot                                   # Plot results"
    echo "  ./run.sh custom <batch> <seqlen> <lr> [accum] [steps]"
    echo ""
    echo "Examples:"
    echo "  ./run.sh custom 32 512 0.015                    # batch=32, seqlen=512, lr=0.015"
    echo "  ./run.sh custom 16 1024 0.01 4 50               # with grad_accum=4, steps=50"
    ;;
esac

