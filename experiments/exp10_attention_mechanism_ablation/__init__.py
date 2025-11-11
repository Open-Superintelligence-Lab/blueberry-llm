"""
Experiment 10: Attention Mechanism Ablation Study

Systematically test and compare different attention mechanisms:
1. Standard Multi-Head Attention (MHA)
2. Multi-Head Latent Attention (MLA) 
3. Grouped Query Attention (GQA)
4. Multi-Query Attention (MQA)
5. Linear Attention (via Gated DeltaNet from FLA)
6. Sparse Attention (DeepSeek-style)
7. Sliding Window Attention
8. No Attention (Identity/Skip)

Each mechanism is tested in isolation with the same model architecture
to measure the direct impact on training dynamics and final performance.
"""

