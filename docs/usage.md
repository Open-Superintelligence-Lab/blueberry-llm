# Usage

## Install

```bash
chmod +x setup.sh && ./setup.sh
pip install pandas seaborn matplotlib mkdocs mkdocs-material
```

## Run a Fixed-FLOPs Grid

```bash
python experiments/run_scaling.py \
  --regime flops \
  --experts 1,2,4,8,16,32 \
  --top-k 2 \
  --d-model 384 --n-heads 8 --n-layers 6 --d-ff-dense 1536 \
  --seq-len 512 --batch-size 16 --max-steps 200 \
  --seeds 3 \
  --collect-router-stats \
  --out-dir experiments/out
```

## Run a Fixed-Params Grid

```bash
python experiments/run_scaling.py --regime params --experts 1,2,4,8,16,32 --out-dir experiments/out
```

## Generate Plots

```bash
python experiments/plotting.py --csv experiments/out/results_flops.csv --out experiments/plots
```

## Build Docs

```bash
mkdocs build
mkdocs serve -a 0.0.0.0:8000
```

!!! note
    Data download for training uses Hugging Face datasets and may require network access.

