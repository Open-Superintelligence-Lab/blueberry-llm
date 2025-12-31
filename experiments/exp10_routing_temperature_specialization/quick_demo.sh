#!/bin/bash
# Quick demo script to run a subset of temperature experiments

echo "========================================="
echo "Temperature Routing Experiment - Quick Demo"
echo "========================================="
echo ""
echo "This script runs a subset of temperature experiments:"
echo "  1. temp_0.7 - Sharp routing"
echo "  2. temp_1.0 - Baseline"
echo "  3. temp_2.0 - Soft routing"
echo ""
echo "Each experiment runs for 500 steps (~2-3 minutes on GPU)"
echo ""

cd /root/blueberry-llm/experiments/exp10_routing_temperature_specialization

# Run 3 representative temperatures
python run_experiment.py --experiments temp_0.7 temp_1.0 temp_2.0 --output-dir ./results

echo ""
echo "========================================="
echo "Generating visualizations..."
echo "========================================="
echo ""

# Generate plots and analysis
python plot_results.py --results-dir ./results --output-dir ./analysis
python analyze_specialization.py --results-dir ./results --output-dir ./analysis

echo ""
echo "========================================="
echo "Demo complete!"
echo "========================================="
echo ""
echo "Results saved in:"
echo "  - ./results/     - Individual experiment results"
echo "  - ./analysis/    - Comparative plots and analysis"
echo ""

