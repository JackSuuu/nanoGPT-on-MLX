"""
Memory profiling and analysis for the model
"""
import argparse
import mlx.core as mx
import numpy as np
from pathlib import Path

from src.model import create_model
from src.utils import load_checkpoint, count_parameters
from src import config as cfg


def analyze_model_memory(model):
    """Analyze model memory usage"""
    print("Model Memory Analysis")
    print("-" * 70)
    
    # Count parameters
    total_params = count_parameters(model)
    
    # Estimate memory for parameters (float32 = 4 bytes)
    param_memory_mb = (total_params * 4) / (1024 ** 2)
    
    print(f"\nParameter Memory:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Memory (float32): {param_memory_mb:.2f} MB")
    print(f"  Memory (float16): {param_memory_mb/2:.2f} MB")
    
    return {
        'total_parameters': total_params,
        'param_memory_mb_fp32': param_memory_mb,
        'param_memory_mb_fp16': param_memory_mb / 2
    }


def estimate_activation_memory(config, batch_size):
    """Estimate activation memory during forward/backward pass"""
    print("\nActivation Memory Estimate")
    print("-" * 70)
    
    d_model = config['d_model']
    n_layers = config['n_layers']
    seq_len = config['context_length']
    
    # Per-layer activation memory (approximate)
    # Attention: Q, K, V, attention weights, output
    attn_memory = batch_size * seq_len * d_model * 4  # Q, K, V, output
    attn_memory += batch_size * config['n_heads'] * seq_len * seq_len  # attention weights
    
    # Feedforward: intermediate activation
    ff_memory = batch_size * seq_len * config['d_ff']
    
    # Total per layer
    per_layer_memory = (attn_memory + ff_memory) * 4 / (1024 ** 2)  # float32, convert to MB
    
    # Total for all layers (approximate)
    total_activation_mb = per_layer_memory * n_layers
    
    # Add embedding and output
    embedding_memory = batch_size * seq_len * d_model * 4 / (1024 ** 2)
    output_memory = batch_size * seq_len * config['vocab_size'] * 4 / (1024 ** 2)
    
    total_memory = total_activation_mb + embedding_memory + output_memory
    
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Per-layer activation: {per_layer_memory:.2f} MB")
    print(f"  Total layers activation: {total_activation_mb:.2f} MB")
    print(f"  Embedding memory: {embedding_memory:.2f} MB")
    print(f"  Output logits memory: {output_memory:.2f} MB")
    print(f"  Total activation memory: {total_memory:.2f} MB")
    print(f"  Backward pass (est. 2x): {total_memory * 2:.2f} MB")
    
    return {
        'activation_memory_mb': total_memory,
        'backward_memory_mb': total_memory * 2,
        'per_layer_memory_mb': per_layer_memory
    }


def test_memory_usage(model, batch_sizes=[1, 2, 4, 8, 16], seq_length=1024):
    """Test actual memory usage with different batch sizes"""
    print("\nMemory Usage Test")
    print("-" * 70)
    print(f"Testing with sequence length: {seq_length}")
    print()
    
    vocab_size = model.vocab_size
    results = []
    
    for batch_size in batch_sizes:
        try:
            # Create input
            x = mx.random.randint(0, vocab_size, (batch_size, seq_length))
            
            # Forward pass
            logits, _ = model(x)
            mx.eval(logits)
            
            # Calculate sizes
            input_size = x.size * 4 / (1024 ** 2)
            output_size = logits.size * 4 / (1024 ** 2)
            
            print(f"  Batch size {batch_size:2d}: ✓")
            print(f"    Input:  {input_size:.2f} MB")
            print(f"    Output: {output_size:.2f} MB")
            
            results.append({
                'batch_size': batch_size,
                'success': True,
                'input_mb': input_size,
                'output_mb': output_size
            })
            
        except Exception as e:
            print(f"  Batch size {batch_size:2d}: ✗ ({str(e)[:50]})")
            results.append({
                'batch_size': batch_size,
                'success': False,
                'error': str(e)
            })
    
    return results


def recommend_batch_size(config, available_memory_gb=16):
    """Recommend optimal batch size for given memory"""
    print("\nBatch Size Recommendation")
    print("-" * 70)
    print(f"Available memory: {available_memory_gb} GB")
    
    # Calculate parameter memory
    total_params = config['vocab_size'] * config['d_model'] * 2  # embeddings (with weight tying)
    total_params += config['n_layers'] * (
        12 * config['d_model'] ** 2 +  # attention
        2 * config['d_model'] * config['d_ff']  # feedforward
    )
    param_memory_gb = (total_params * 4) / (1024 ** 3)
    
    # Estimate available memory for activations
    available_for_activations = available_memory_gb - param_memory_gb - 1  # 1GB buffer
    
    # Estimate activation memory per sample
    seq_len = config['context_length']
    d_model = config['d_model']
    n_layers = config['n_layers']
    
    # Rough estimate: memory per sample
    memory_per_sample = (
        seq_len * d_model * n_layers * 8  # activations through layers
        + seq_len * config['vocab_size'] * 4  # output logits
    ) / (1024 ** 3)
    
    # Recommend batch size (with safety margin)
    recommended_batch = int(available_for_activations / memory_per_sample * 0.7)
    recommended_batch = max(1, min(recommended_batch, 32))  # clamp between 1-32
    
    print(f"  Parameter memory: {param_memory_gb:.2f} GB")
    print(f"  Available for activations: {available_for_activations:.2f} GB")
    print(f"  Estimated memory per sample: {memory_per_sample*1024:.2f} MB")
    print(f"  Recommended batch size: {recommended_batch}")
    
    return recommended_batch


def main():
    parser = argparse.ArgumentParser(description="Analyze model memory usage")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Path to model checkpoint (optional)")
    parser.add_argument("--test_batches", action="store_true",
                       help="Test different batch sizes")
    parser.add_argument("--available_memory", type=float, default=16,
                       help="Available memory in GB (default: 16)")
    
    args = parser.parse_args()
    
    # Load config
    config = {k: v for k, v in vars(cfg).items() if not k.startswith('_')}
    
    print("=" * 70)
    print("Model Memory Analysis")
    print("=" * 70)
    
    # Create model
    print("\nCreating model...")
    model = create_model(config)
    
    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        load_checkpoint(args.checkpoint, model)
    
    # Analyze model memory
    print("\n" + "=" * 70)
    model_memory = analyze_model_memory(model)
    
    # Estimate activation memory
    print("\n" + "=" * 70)
    activation_memory = estimate_activation_memory(config, batch_size=8)
    
    # Recommend batch size
    print("\n" + "=" * 70)
    recommended_batch = recommend_batch_size(config, args.available_memory)
    
    # Test batch sizes
    if args.test_batches:
        print("\n" + "=" * 70)
        test_results = test_memory_usage(model, batch_sizes=[1, 2, 4, 8, 16, 32])
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
