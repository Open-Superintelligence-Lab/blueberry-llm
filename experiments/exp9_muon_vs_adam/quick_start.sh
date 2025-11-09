#!/bin/bash
# Quick start script for exp9 Muon vs Adam experiments

echo "=================================================="
echo "  Experiment 9: Muon vs Adam Optimizer Comparison"
echo "=================================================="
echo ""
echo "This script will help you run the optimizer comparison experiments."
echo ""

# Show menu
echo "Available options:"
echo "  1) Quick comparison (muon_baseline vs adam_baseline) - ~20 min"
echo "  2) Run all experiments - ~70 min"
echo "  3) List available experiments"
echo "  4) View results (if experiments have been run)"
echo "  5) Custom experiments"
echo "  q) Quit"
echo ""

read -p "Select an option (1-5, q): " choice

case $choice in
    1)
        echo ""
        echo "Running quick comparison..."
        python run_experiments.py --quick
        ;;
    2)
        echo ""
        echo "Running all experiments (this will take ~70 minutes)..."
        read -p "Are you sure? (y/n): " confirm
        if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
            python run_experiments.py --all
        else
            echo "Cancelled."
        fi
        ;;
    3)
        echo ""
        python run_experiments.py --list
        ;;
    4)
        echo ""
        python view_experiments.py
        ;;
    5)
        echo ""
        echo "Available experiments:"
        python run_experiments.py --list
        echo ""
        read -p "Enter experiment names (space-separated): " exps
        python run_experiments.py -e $exps
        ;;
    q|Q)
        echo "Exiting."
        exit 0
        ;;
    *)
        echo "Invalid option. Please try again."
        exit 1
        ;;
esac

echo ""
echo "=================================================="
echo "To view results, run: python view_experiments.py"
echo "=================================================="

