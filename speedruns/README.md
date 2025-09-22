# T4 Speedrun Challenge

**Goal**: Achieve the lowest validation loss in 5 minutes on Tesla T4 GPU.

## Rules
- **Time**: 5 minutes
- **Hardware**: Tesla T4 (16GB VRAM)
- **Metric**: Validation loss (lower = better)

## How to Compete
1. Copy template: `cp -r speedrun/submissions/template-change-this speedrun/submissions/your-username`
2. Modify your folder
3. Run: `python speedrun/submissions/your-username/speedrun.py`
4. Submit PR

## Structure
```
speedrun/
├── submissions/
│   ├── template-change-this/
│   └── your-username/
└── README.md
```

## Customize
- `config.py` - Model architecture, hyperparameters
- `speedrun.py` - Training loop, optimizations