"""
Configuration for GPT model on MLX
Target: 10M parameters (optimized for M2 Pro 16GB)
"""

# Model architecture
# For ~50M parameters: needed to learn TinyStories vocabulary distribution
# 22M was too small to learn which 5K tokens (out of 50K) are used in TinyStories
# Params ≈ vocab_size * d_model + n_layers * (4 * d_model^2 + 4 * d_model * d_ff)
# With d_model=384, d_ff=1536, n_layers=8, vocab_size=50257: ~50M params
n_layers = 8       # Increased from 6
n_heads = 8        # Increased from 6 (must divide d_model)
d_model = 384      # Increased from 192 (384 = 8 heads * 48 dim per head)
d_ff = 1536        # Increased from 768 (4 * d_model)
context_length = 512  # Reduced from 1024 for faster training
vocab_size = 50257  # GPT-2 tokenizer vocab size
dropout = 0.1

# Training hyperparameters  
batch_size = 12  # Reduced from 16 (50M model uses more memory)
learning_rate = 3e-4  # Standard for larger models - can handle higher LR
weight_decay = 0.1
max_iters = 30000  # More iterations needed for 50M model to converge
warmup_iters = 2000  # Longer warmup for larger model
lr_decay_iters = 30000  # Match max_iters for full decay schedule
min_lr = 3e-5  # 10% of max learning rate

# Evaluation
eval_interval = 500
eval_iters = 50  # Reduced from 100 for faster evaluation

# Checkpointing
save_interval = 1000
checkpoint_dir = "checkpoints"

# Data
max_tokens = 2_000_000  # Subset of TinyStories (~2M tokens - simpler data, use more)
train_split = 0.9

# System
seed = 42
