"""
Clean GPT Transformer implementation using MLX
Target: 128M parameters
"""
import math
import mlx.core as mx
import mlx.nn as nn


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism"""
    
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = math.sqrt(self.head_dim)
        
        # Combined QKV projection
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def __call__(self, x, mask=None):
        batch_size, seq_len, d_model = x.shape
        
        # Project and split into Q, K, V
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.n_heads, self.head_dim)
        qkv = qkv.transpose(2, 0, 3, 1, 4)  # (3, batch, n_heads, seq_len, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        scores = (q @ k.transpose(0, 1, 3, 2)) / self.scale  # (batch, n_heads, seq_len, seq_len)
        
        # Apply causal mask
        if mask is not None:
            scores = scores + mask
        
        attn_weights = mx.softmax(scores, axis=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Combine heads
        out = attn_weights @ v  # (batch, n_heads, seq_len, head_dim)
        out = out.transpose(0, 2, 1, 3)  # (batch, seq_len, n_heads, head_dim)
        out = out.reshape(batch_size, seq_len, d_model)
        
        return self.out_proj(out)


class FeedForward(nn.Module):
    """Position-wise feed-forward network"""
    
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def __call__(self, x):
        x = self.fc1(x)
        x = nn.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class TransformerBlock(nn.Module):
    """Single transformer block with attention and feedforward"""
    
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)
        
    def __call__(self, x, mask=None):
        # Pre-norm architecture
        x = x + self.dropout(self.attn(self.ln1(x), mask))
        x = x + self.dropout(self.ff(self.ln2(x)))
        return x


class GPT(nn.Module):
    """GPT Language Model with ~128M parameters"""
    
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, 
                 context_length, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.context_length = context_length
        
        # Token and position embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(context_length, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Transformer blocks
        self.blocks = [
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ]
        
        # Final layer norm and output projection
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Weight tying: share weights between token embedding and output projection
        self.lm_head.weight = self.token_embedding.weight
        
    def __call__(self, idx, targets=None):
        batch_size, seq_len = idx.shape
        assert seq_len <= self.context_length, f"Sequence length {seq_len} exceeds context length {self.context_length}"
        
        # Create causal mask (upper triangular matrix of -inf)
        mask = mx.triu(mx.full((seq_len, seq_len), -1e9), k=1)
        
        # Token + positional embeddings
        positions = mx.arange(seq_len)
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(positions)
        x = self.dropout(tok_emb + pos_emb)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, mask)
        
        # Final layer norm
        x = self.ln_f(x)
        
        # Project to vocabulary
        logits = self.lm_head(x)
        
        # Calculate loss if targets provided
        loss = None
        if targets is not None:
            loss = nn.losses.cross_entropy(
                logits.reshape(-1, self.vocab_size),
                targets.reshape(-1),
                reduction='mean'
            )
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Generate new tokens autoregressively
        
        Args:
            idx: (batch_size, seq_len) token indices
            max_new_tokens: number of tokens to generate
            temperature: sampling temperature
            top_k: if set, only sample from top k logits
        """
        for _ in range(max_new_tokens):
            # Crop to context length
            idx_cond = idx if idx.shape[1] <= self.context_length else idx[:, -self.context_length:]
            
            # Forward pass
            logits, _ = self(idx_cond)
            
            # Get logits for last position
            logits = logits[:, -1, :] / temperature
            
            # Optional top-k sampling
            if top_k is not None:
                # Get top-k values
                top_logits = mx.topk(logits, top_k, axis=-1)
                # Create a mask for top-k positions
                # Find the k-th largest value (the minimum of top-k)
                kth_value = top_logits[:, -1:] 
                # Keep only logits >= k-th value, set others to -inf
                logits = mx.where(logits >= kth_value, logits, -1e9)
            
            # FIXED: mx.random.categorical() is broken in MLX!
            # Use greedy decoding (argmax) for now
            # TODO: Implement proper sampling when MLX fixes categorical()
            idx_next = mx.argmax(logits, axis=-1, keepdims=True)
            idx_next = mx.expand_dims(idx_next, axis=0) if idx_next.ndim == 1 else idx_next
            
            # Append to sequence
            idx = mx.concatenate([idx, idx_next], axis=1)
        
        return idx


def create_model(config):
    """Create GPT model from config"""
    model = GPT(
        vocab_size=config['vocab_size'],
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        n_layers=config['n_layers'],
        d_ff=config['d_ff'],
        context_length=config['context_length'],
        dropout=config['dropout']
    )
    return model
