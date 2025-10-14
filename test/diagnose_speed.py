"""
Diagnose training speed issues
"""
import time
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from src.model import create_model
from src.data import create_datasets
from src import config as cfg

print("=" * 70)
print("Training Speed Diagnostics")
print("=" * 70)

# Config
config = {k: v for k, v in vars(cfg).items() if not k.startswith('_')}
mx.random.seed(42)

print("\n[1/3] Loading data...")
train_dataset, val_dataset = create_datasets(config)
print(f"✓ Train tokens: {len(train_dataset.tokens):,}")

print("\n[2/3] Creating model...")
model = create_model(config)
mx.eval(model.parameters())
print(f"✓ Model created")

print("\n[3/3] Testing training speed...")
print("-" * 70)

optimizer = optim.AdamW(learning_rate=3e-4, weight_decay=0.1)

def loss_fn(model, x, y):
    _, loss = model(x, y)
    return loss

loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

# Test different scenarios
scenarios = [
    ("Batch preparation", lambda: train_dataset.get_batch(config['batch_size'])),
    ("Forward pass only", lambda: model(x)),
    ("Forward + loss", lambda: loss_fn(model, x, y)),
    ("Forward + backward", lambda: loss_and_grad_fn(model, x, y)),
]

x, y = train_dataset.get_batch(config['batch_size'])
print(f"Batch shape: x={x.shape}, y={y.shape}")
print(f"Batch size: {config['batch_size']}, Context: {config['context_length']}")
print(f"Tokens per batch: {config['batch_size'] * config['context_length']:,}")
print()

for name, fn in scenarios:
    # Warmup
    result = fn()
    if isinstance(result, tuple):
        mx.eval(result)
    else:
        mx.eval(result)
    
    # Timed runs
    times = []
    for _ in range(5):
        start = time.time()
        result = fn()
        if isinstance(result, tuple):
            mx.eval(result)
        else:
            mx.eval(result)
        times.append(time.time() - start)
    
    avg_time = sum(times) / len(times)
    print(f"{name:25s}: {avg_time*1000:6.1f} ms/iter")

print()
print("-" * 70)
print("Full training step test (10 iterations):")
print()

start = time.time()
for i in range(10):
    iter_start = time.time()
    
    # Get batch
    x, y = train_dataset.get_batch(config['batch_size'])
    
    # Forward and backward
    loss, grads = loss_and_grad_fn(model, x, y)
    
    # Update
    optimizer.update(model, grads)
    
    # Evaluate
    mx.eval(loss, model.parameters())
    
    iter_time = time.time() - iter_start
    print(f"Iter {i}: {iter_time:.3f}s ({float(loss):.4f})")

total_time = time.time() - start
avg_iter_time = total_time / 10

print()
print("-" * 70)
print(f"Average iteration time: {avg_iter_time:.3f}s")
print(f"Tokens per second: {config['batch_size'] * config['context_length'] / avg_iter_time:.0f}")
print()

if avg_iter_time > 5:
    print("⚠️  WARNING: Very slow! (>5s per iteration)")
    print("Expected: 0.5-2s per iteration on M2 Pro")
    print()
    print("Possible issues:")
    print("1. MLX not using GPU acceleration")
    print("2. Model too large for memory (causing swapping)")
    print("3. Need to reduce batch_size or context_length")
elif avg_iter_time > 2:
    print("⚠️  Slower than expected (>2s per iteration)")
    print("Expected: 0.5-2s per iteration on M2 Pro")
else:
    print("✓ Speed looks reasonable!")

print("=" * 70)
