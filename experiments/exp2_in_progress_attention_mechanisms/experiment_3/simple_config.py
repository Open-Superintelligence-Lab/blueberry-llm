"""
Simple config class for DeepSeek models.

This provides a minimal config object that can be used with DeepSeek models
when we only have a dictionary configuration.
"""

from types import SimpleNamespace


class SimpleDeepSeekConfig(SimpleNamespace):
    """Simple config class for DeepSeek models."""
    
    def __init__(self, config_dict):
        super().__init__()
        
        # Required attributes for DeepSeekV3Attention
        self.attention_dropout = config_dict.get('dropout', 0.1)
        self.hidden_size = config_dict['d_model']
        self.num_attention_heads = config_dict['n_heads']
        self.max_position_embeddings = config_dict['max_position_embeddings']
        
        # RoPE configuration
        self.rope_theta = 10000.0
        self.q_lora_rank = config_dict.get('q_lora_rank', 64)
        self.qk_rope_head_dim = config_dict['d_model'] // config_dict['n_heads'] // 2
        self.kv_lora_rank = config_dict.get('kv_lora_rank', 64)
        self.v_head_dim = config_dict['d_model'] // config_dict['n_heads']
        self.qk_nope_head_dim = config_dict['d_model'] // config_dict['n_heads'] // 2
        
        # Additional attributes
        self.num_experts = config_dict.get('num_experts', 8)
        self.top_k = config_dict.get('top_k', 2)
        self.dropout = config_dict.get('dropout', 0.1)
        self.d_model = config_dict['d_model']
        self.n_heads = config_dict['n_heads']
        self.d_ff = config_dict['d_ff']
        self.vocab_size = config_dict['vocab_size']
        self.n_layers = config_dict['n_layers']
        
        # DeepSeek specific attributes
        self.rope_scaling = None
        self.rope_theta = 10000.0
        self.attention_bias = False
        self.attention_dropout = config_dict.get('dropout', 0.1)
        self.use_sliding_window = False
        self.sliding_window = None
        self.max_sliding_window = None
        self.use_alibi = False
        self.alibi_bias_max = 8
        self.new_decoder_architecture = True
        self.use_parallel_residual = True
        self.is_causal = True
