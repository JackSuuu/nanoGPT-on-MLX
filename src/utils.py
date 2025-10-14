"""
Utility functions for training and checkpointing
"""
import os
import json
import mlx.core as mx
import mlx.nn as nn
from pathlib import Path


def count_parameters(model):
    """Count total trainable parameters"""
    total = 0
    params = model.parameters()
    
    def count_nested(name, obj, path=""):
        nonlocal total
        if isinstance(obj, dict):
            for key, value in obj.items():
                new_path = f"{path}.{key}" if path else key
                count_nested(key, value, new_path)
        elif isinstance(obj, list):
            for i, value in enumerate(obj):
                new_path = f"{path}[{i}]" if path else f"[{i}]"
                count_nested(f"[{i}]", value, new_path)
        elif hasattr(obj, 'shape'):
            size = obj.size
            total += size
            print(f"{path}: {size:,}")
    
    count_nested("", params)
    print(f"\nTotal parameters: {total:,} ({total/1e6:.2f}M)")
    print(f"Note: With weight tying, embedding weights are shared with output layer")
    return total


def save_checkpoint(model, optimizer, iteration, loss, config, checkpoint_dir):
    """Save model checkpoint"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Save model weights
    checkpoint_path = Path(checkpoint_dir) / f"checkpoint_{iteration}.npz"
    model.save_weights(str(checkpoint_path))
    
    # Save metadata separately
    metadata = {
        'iteration': iteration,
        'loss': float(loss),
        'config': config
    }
    metadata_path = Path(checkpoint_dir) / f"checkpoint_{iteration}_meta.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Save config separately for easy access
    config_path = Path(checkpoint_dir) / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Checkpoint saved to {checkpoint_path}")


def load_checkpoint(checkpoint_path, model, optimizer=None):
    """Load model checkpoint"""
    # Load model weights
    model.load_weights(str(checkpoint_path))
    
    # Load metadata
    metadata_path = str(checkpoint_path).replace('.npz', '_meta.json')
    iteration = 0
    loss = 0.0
    
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            iteration = metadata.get('iteration', 0)
            loss = metadata.get('loss', 0.0)
    
    print(f"Checkpoint loaded from {checkpoint_path}")
    print(f"Iteration: {iteration}, Loss: {loss:.4f}")
    
    return iteration, loss


def get_lr(iteration, config):
    """Learning rate schedule with warmup and cosine decay"""
    warmup_iters = config['warmup_iters']
    lr_decay_iters = config['lr_decay_iters']
    learning_rate = config['learning_rate']
    min_lr = config['min_lr']
    
    # Linear warmup
    if iteration < warmup_iters:
        return learning_rate * iteration / warmup_iters
    
    # Cosine decay
    if iteration > lr_decay_iters:
        return min_lr
    
    decay_ratio = (iteration - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + mx.cos(mx.array(mx.pi * decay_ratio)))
    return min_lr + coeff * (learning_rate - min_lr)
