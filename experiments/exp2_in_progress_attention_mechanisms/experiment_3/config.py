"""
Configuration file for Experiment 3: Dynamic Sparsity and Adaptive Attention Patterns

This file contains all configuration parameters for the experiment, making it easy
to modify experimental settings without changing the main code.
"""

from types import SimpleNamespace


class ModelConfig(SimpleNamespace):
    """Model architecture configuration."""
    
    def __init__(self):
        # Core model parameters
        self.d_model = 512                    # Hidden dimension
        self.n_layers = 6                     # Number of transformer layers
        self.n_heads = 8                      # Number of attention heads
        self.d_ff = 2048                      # Feed-forward dimension
        self.vocab_size = 10000               # Vocabulary size
        self.max_position_embeddings = 2048   # Maximum sequence length
        
        # MoE parameters
        self.num_experts = 8                  # Number of experts in MoE
        self.top_k = 2                        # Top-k experts to use
        self.dropout = 0.1                    # Dropout rate
        
        # Adaptive sparsity parameters
        self.indexer_heads = 4                # Number of indexer heads
        self.indexer_dim = 64                 # Indexer dimension
        self.complexity_dim = 64              # Complexity analyzer dimension
        self.num_proxy_heads = 4              # Number of proxy heads for entropy estimation
        
        # Sparsity bounds
        self.min_sparsity = 0.1               # Minimum sparsity ratio (10% dense)
        self.max_sparsity = 0.9               # Maximum sparsity ratio (90% sparse)


class TrainingConfig(SimpleNamespace):
    """Training configuration."""
    
    def __init__(self):
        # Optimization parameters
        self.learning_rate = 1e-3             # Learning rate
        self.weight_decay = 0.01              # Weight decay
        self.beta1 = 0.9                      # Adam beta1
        self.beta2 = 0.999                    # Adam beta2
        self.eps = 1e-8                       # Adam epsilon
        
        # Training schedule
        self.steps = 2000                     # Number of training steps
        self.warmup_steps = 100               # Warmup steps
        self.eval_every = 200                 # Evaluation frequency
        
        # Data parameters
        self.batch_size = 16                  # Batch size
        self.gradient_accumulation_steps = 1  # Gradient accumulation
        self.max_grad_norm = 1.0              # Gradient clipping
        
        # Regularization
        self.dropout = 0.1                    # Dropout rate
        self.label_smoothing = 0.0            # Label smoothing


class ExperimentConfig(SimpleNamespace):
    """Experiment configuration."""
    
    def __init__(self):
        # Experiment parameters
        self.sequence_lengths = [64, 128, 256, 512, 1024, 2048]  # Sequence lengths to test
        self.random_seeds = [42, 123, 456]                       # Random seeds for reproducibility
        
        # Model variants to test
        self.model_variants = [
            {
                'name': 'dense',
                'type': 'dense',
                'sparsity_ratio': 1.0,
                'description': 'Full dense attention (baseline)'
            },
            {
                'name': 'fixed_sparse_25',
                'type': 'fixed_sparse',
                'sparsity_ratio': 0.25,
                'description': 'Fixed 25% sparsity (75% dense)'
            },
            {
                'name': 'fixed_sparse_50',
                'type': 'fixed_sparse',
                'sparsity_ratio': 0.5,
                'description': 'Fixed 50% sparsity (baseline)'
            },
            {
                'name': 'fixed_sparse_75',
                'type': 'fixed_sparse',
                'sparsity_ratio': 0.75,
                'description': 'Fixed 75% sparsity (25% dense)'
            },
            {
                'name': 'adaptive_sparse',
                'type': 'adaptive_sparse',
                'sparsity_ratio': None,
                'description': 'Adaptive sparsity (experimental)'
            },
        ]
        
        # Dataset parameters
        self.dataset_name = 'TinyStories'
        self.dataset_path = '/root/deepseek-sparse-attention-research/data'
        self.max_samples_per_split = 10000    # Limit samples for faster experiments
        
        # Output parameters
        self.results_dir = 'results'
        self.save_model_checkpoints = False   # Whether to save model checkpoints
        self.log_level = 'INFO'               # Logging level
        
        # Hardware parameters
        self.device = 'auto'                  # Device selection ('auto', 'cuda', 'cpu')
        self.num_workers = 0                  # DataLoader workers (0 for debugging)
        self.pin_memory = True                # Pin memory for faster GPU transfer
        
        # Analysis parameters
        self.confidence_level = 0.95          # Confidence level for statistical tests
        self.min_runs_for_analysis = 2        # Minimum runs needed for analysis
        self.save_attention_weights = False   # Whether to save attention weights


class AdaptiveSparsityConfig(SimpleNamespace):
    """Configuration specific to adaptive sparsity components."""
    
    def __init__(self):
        # Lightning Indexer configuration
        self.indexer_heads = 4                # Number of indexer heads
        self.indexer_dim = 64                 # Indexer dimension
        self.indexer_activation = 'relu'      # Indexer activation function
        
        # Content Complexity Analyzer
        self.complexity_analyzer_dim = 64     # Complexity analyzer hidden dimension
        self.use_token_diversity = True       # Whether to use token diversity
        self.use_perplexity_estimation = True # Whether to use perplexity estimation
        self.use_sequence_variance = True     # Whether to use sequence variance
        
        # Attention Entropy Estimator
        self.entropy_proxy_heads = 4          # Number of proxy heads for entropy
        self.entropy_hidden_dim = 32          # Entropy estimator hidden dimension
        
        # Adaptive K Calculator
        self.initial_length_weight = 0.4      # Initial weight for length factor
        self.initial_complexity_weight = 0.3  # Initial weight for complexity factor
        self.initial_entropy_weight = 0.3     # Initial weight for entropy factor
        self.initial_base_sparsity = 0.5      # Initial base sparsity level
        
        # Sparsity bounds
        self.min_sparsity_ratio = 0.1         # Minimum sparsity (10% dense)
        self.max_sparsity_ratio = 0.9         # Maximum sparsity (90% sparse)
        
        # Length adaptation
        self.length_adapter_hidden_dim = 16   # Length adapter hidden dimension
        self.use_length_adaptation = True     # Whether to use length adaptation


def get_full_config():
    """Get complete configuration combining all configs."""
    return SimpleNamespace(
        model=ModelConfig(),
        training=TrainingConfig(),
        experiment=ExperimentConfig(),
        adaptive_sparsity=AdaptiveSparsityConfig()
    )


def print_config_summary(config):
    """Print a summary of the configuration."""
    print("="*80)
    print("EXPERIMENT 3 CONFIGURATION SUMMARY")
    print("="*80)
    
    print(f"\nModel Configuration:")
    print(f"  Architecture: {config.model.n_layers} layers, {config.model.n_heads} heads")
    print(f"  Hidden dimension: {config.model.d_model}")
    print(f"  Vocabulary size: {config.model.vocab_size}")
    print(f"  Max sequence length: {config.model.max_position_embeddings}")
    print(f"  MoE: {config.model.num_experts} experts, top-{config.model.top_k}")
    
    print(f"\nTraining Configuration:")
    print(f"  Learning rate: {config.training.learning_rate}")
    print(f"  Batch size: {config.training.batch_size}")
    print(f"  Training steps: {config.training.steps}")
    print(f"  Evaluation frequency: every {config.training.eval_every} steps")
    
    print(f"\nExperiment Configuration:")
    print(f"  Sequence lengths: {config.experiment.sequence_lengths}")
    print(f"  Random seeds: {config.experiment.random_seeds}")
    print(f"  Model variants: {len(config.experiment.model_variants)}")
    for variant in config.experiment.model_variants:
        print(f"    - {variant['name']}: {variant['description']}")
    
    print(f"\nAdaptive Sparsity Configuration:")
    print(f"  Indexer heads: {config.adaptive_sparsity.indexer_heads}")
    print(f"  Indexer dimension: {config.adaptive_sparsity.indexer_dim}")
    print(f"  Sparsity bounds: {config.adaptive_sparsity.min_sparsity_ratio:.1%} - {config.adaptive_sparsity.max_sparsity_ratio:.1%}")
    print(f"  Length adaptation: {'Enabled' if config.adaptive_sparsity.use_length_adaptation else 'Disabled'}")
    
    print("="*80)


def validate_config(config):
    """Validate configuration parameters."""
    errors = []
    warnings = []
    
    # Model validation
    if config.model.d_model <= 0:
        errors.append("d_model must be positive")
    if config.model.n_layers <= 0:
        errors.append("n_layers must be positive")
    if config.model.n_heads <= 0:
        errors.append("n_heads must be positive")
    if config.model.d_model % config.model.n_heads != 0:
        errors.append("d_model must be divisible by n_heads")
    
    # Training validation
    if config.training.learning_rate <= 0:
        errors.append("learning_rate must be positive")
    if config.training.batch_size <= 0:
        errors.append("batch_size must be positive")
    if config.training.steps <= 0:
        errors.append("steps must be positive")
    
    # Experiment validation
    if len(config.experiment.sequence_lengths) == 0:
        errors.append("sequence_lengths cannot be empty")
    if any(seq_len <= 0 for seq_len in config.experiment.sequence_lengths):
        errors.append("all sequence lengths must be positive")
    if max(config.experiment.sequence_lengths) > config.model.max_position_embeddings:
        warnings.append("some sequence lengths exceed max_position_embeddings")
    
    if len(config.experiment.random_seeds) < 2:
        warnings.append("using only one random seed reduces statistical reliability")
    
    # Adaptive sparsity validation
    if config.adaptive_sparsity.min_sparsity_ratio >= config.adaptive_sparsity.max_sparsity_ratio:
        errors.append("min_sparsity_ratio must be less than max_sparsity_ratio")
    
    if config.adaptive_sparsity.indexer_heads <= 0:
        errors.append("indexer_heads must be positive")
    
    # Report errors and warnings
    if errors:
        print("CONFIGURATION ERRORS:")
        for error in errors:
            print(f"  ERROR: {error}")
        raise ValueError("Configuration validation failed")
    
    if warnings:
        print("CONFIGURATION WARNINGS:")
        for warning in warnings:
            print(f"  WARNING: {warning}")
    
    print("Configuration validation passed!")


# Default configuration
DEFAULT_CONFIG = get_full_config()


if __name__ == "__main__":
    # Test configuration
    config = get_full_config()
    print_config_summary(config)
    validate_config(config)
