"""
Training script for GPT on MLX
Optimized for 16GB M2 Pro
"""
import os
import sys
import time
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from tqdm import tqdm

from . import config as cfg
from .model import create_model
from .data import create_datasets, estimate_loss
from .utils import count_parameters, save_checkpoint, load_checkpoint, get_lr


def loss_fn(model, x, y):
    """Loss function for training"""
    _, loss = model(x, y)
    return loss


def train_step(model, x, y, optimizer, loss_and_grad_fn):
    """Single training step - optimized"""
    # Forward and backward pass
    loss, grads = loss_and_grad_fn(model, x, y)
    
    # Update parameters
    optimizer.update(model, grads)
    
    # Force evaluation
    mx.eval(loss, model.parameters())
    
    return loss


def train(resume_from=None):
    """Main training loop
    
    Args:
        resume_from: Path to checkpoint file to resume from (e.g., 'checkpoints/checkpoint_9000.npz')
    """
    # Convert config module to dict
    config = {k: v for k, v in vars(cfg).items() if not k.startswith('_')}
    
    # Set random seed
    mx.random.seed(config['seed'])
    
    print("=" * 70)
    print("GPT Training on MLX")
    print("=" * 70)
    
    # Create datasets
    print("\n[1/4] Loading data...")
    train_dataset, val_dataset = create_datasets(config)
    
    # Create model
    print("\n[2/4] Initializing model...")
    model = create_model(config)
    mx.eval(model.parameters())
    
    print("\nModel architecture:")
    count_parameters(model)
    
    # Create optimizer
    print("\n[3/4] Setting up optimizer...")
    optimizer = optim.AdamW(
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Resume from checkpoint if specified
    start_iteration = 0
    best_val_loss = float('inf')
    last_loss = float('inf')
    
    if resume_from:
        print(f"\n[*] Resuming from checkpoint: {resume_from}")
        checkpoint_iter, checkpoint_loss = load_checkpoint(resume_from, model, optimizer)
        start_iteration = checkpoint_iter + 1  # Start from next iteration
        best_val_loss = checkpoint_loss
        last_loss = checkpoint_loss
        print(f"    Resuming from iteration {start_iteration}")
        print(f"    Previous loss: {best_val_loss:.4f}")
        
        # Check if already at max_iters
        if start_iteration >= config['max_iters']:
            print(f"\n⚠️  Already at max_iters ({config['max_iters']})!")
            print(f"    Current checkpoint: iteration {checkpoint_iter}")
            print(f"\n💡 To continue training, update src/config.py:")
            print(f"    Change: max_iters = {config['max_iters']}")
            print(f"    To:     max_iters = {config['max_iters'] + 5000}  (or any higher value)")
            print()
            return
    
    # Value and grad function
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    
    # Training loop
    print("\n[4/4] Starting training...")
    print("=" * 70)
    
    start_time = time.time()
    
    # Create progress bar with better formatting
    pbar = tqdm(range(start_iteration, config['max_iters']), desc="Training", unit="iter", 
                miniters=1, mininterval=0.5, initial=start_iteration, total=config['max_iters'])
    
    for iteration in pbar:
        # Update learning rate
        lr = float(get_lr(iteration, config))
        optimizer.learning_rate = lr
        
        # Get batch
        x, y = train_dataset.get_batch(config['batch_size'])
        
        # Training step (optimized)
        loss = train_step(model, x, y, optimizer, loss_and_grad_fn)
        last_loss = float(loss)  # Track last loss for final save
        
        # Update progress bar every iteration
        elapsed = time.time() - start_time
        tokens_per_sec = (iteration - start_iteration + 1) * config['batch_size'] * config['context_length'] / elapsed
        pbar.set_postfix({
            'loss': f'{float(loss):.4f}',
            'lr': f'{lr:.2e}',
            'tok/s': f'{tokens_per_sec:.0f}'
        }, refresh=True)
        
        # Detailed logging every 100 iterations
        if iteration % 100 == 0 or iteration == config['max_iters'] - 1:
            tqdm.write(f"\niter {iteration:5d} | loss {float(loss):.4f} | lr {lr:.2e} | "
                      f"{tokens_per_sec:.0f} tok/s")
        
        # Evaluation (skip at iteration 0, evaluate at intervals and at end)
        should_eval = (iteration > 0 and iteration % config['eval_interval'] == 0) or iteration == config['max_iters'] - 1
        
        if should_eval:
            pbar.write("\n" + "-" * 70)
            pbar.write("Evaluating...")
            
            train_loss = estimate_loss(model, train_dataset, config)
            val_loss = estimate_loss(model, val_dataset, config)
            
            pbar.write(f"iter {iteration:5d} | train loss {train_loss:.4f} | val loss {val_loss:.4f}")
            pbar.write("-" * 70 + "\n")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(model, optimizer, iteration, val_loss, config, 
                              config['checkpoint_dir'])
        
        # Save checkpoint periodically
        if iteration > 0 and iteration % config['save_interval'] == 0:
            save_checkpoint(model, optimizer, iteration, float(loss), config,
                          config['checkpoint_dir'])
    
    # Close progress bar
    pbar.close()
    
    # Final save (only if we actually trained)
    if start_iteration < config['max_iters']:
        print("\n" + "=" * 70)
        print("Training complete!")
        save_checkpoint(model, optimizer, config['max_iters'], last_loss, config,
                       config['checkpoint_dir'])
        
        total_time = time.time() - start_time
        print(f"Total training time: {total_time/60:.2f} minutes")
        print(f"Best validation loss: {best_val_loss:.4f}")
        print("=" * 70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train GPT model on MLX')
    parser.add_argument('--resume', type=str, default=None,
                      help='Path to checkpoint to resume from (e.g., checkpoints/checkpoint_9000.npz)')
    
    args = parser.parse_args()
    
    train(resume_from=args.resume)
