"""
Evaluate model perplexity on validation/test sets
"""
import argparse
import numpy as np
import mlx.core as mx
from pathlib import Path
import time

from src.model import create_model
from src.data import create_datasets
from src.utils import load_checkpoint
from src import config as cfg


def calculate_perplexity(model, dataset, batch_size=8, max_batches=None):
    """
    Calculate perplexity on a dataset
    
    Perplexity = exp(average negative log-likelihood)
    Lower perplexity = better model
    """
    total_loss = 0.0
    total_tokens = 0
    num_batches = 0
    
    print(f"Calculating perplexity on {len(dataset.tokens):,} tokens...")
    
    # Determine number of batches
    if max_batches is None:
        max_batches = min(1000, len(dataset.tokens) // (batch_size * dataset.context_length))
    
    start_time = time.time()
    
    for i in range(max_batches):
        x, y = dataset.get_batch(batch_size)
        
        # Forward pass
        logits, loss = model(x, y)
        
        # Accumulate loss
        total_loss += float(loss) * x.size
        total_tokens += x.size
        num_batches += 1
        
        if (i + 1) % 100 == 0:
            print(f"  Batch {i+1}/{max_batches} - Avg loss: {total_loss/total_tokens:.4f}")
    
    elapsed = time.time() - start_time
    
    # Calculate perplexity
    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)
    
    print(f"\nResults:")
    print(f"  Total batches: {num_batches}")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Average loss: {avg_loss:.4f}")
    print(f"  Perplexity: {perplexity:.2f}")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Tokens/sec: {total_tokens/elapsed:.0f}")
    
    return {
        'perplexity': float(perplexity),
        'loss': float(avg_loss),
        'tokens': total_tokens,
        'batches': num_batches,
        'time': elapsed
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate model perplexity")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--dataset", type=str, default=None,
                       choices=["tinystories", "finewebedu"],
                       help="Dataset to use (default: use dataset from config.py)")
    parser.add_argument("--split", type=str, default="validation",
                       choices=["train", "validation", "test"],
                       help="Dataset split to evaluate")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size for evaluation")
    parser.add_argument("--max_batches", type=int, default=None,
                       help="Maximum number of batches (None = all)")
    
    args = parser.parse_args()
    
    # Load config
    config = {k: v for k, v in vars(cfg).items() if not k.startswith('_')}
    
    # Determine which dataset to use (override if specified)
    if args.dataset is not None:
        config['dataset_name'] = args.dataset
        dataset_name = args.dataset
    else:
        dataset_name = config.get('dataset_name', 'tinystories')
    
    print("=" * 70)
    print("Model Perplexity Evaluation")
    print("=" * 70)
    print(f"Dataset: {dataset_name}")
    print(f"Split: {args.split}")
    
    # Create model
    print("\nLoading model...")
    model = create_model(config)
    load_checkpoint(args.checkpoint, model)
    
    # Load dataset
    print(f"\nLoading dataset...")
    train_dataset, val_dataset = create_datasets(config)
    dataset = val_dataset if args.split == "validation" else train_dataset
    
    # Calculate perplexity
    print("\n" + "=" * 70)
    results = calculate_perplexity(
        model, 
        dataset, 
        batch_size=args.batch_size,
        max_batches=args.max_batches
    )
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    main()
