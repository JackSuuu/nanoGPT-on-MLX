"""
Quick test to verify 10M model speed
"""
import time
import mlx.core as mx
import mlx.nn as nn
from src.model import create_model
from src import config as cfg
from src.utils import count_parameters

print("Testing 10M parameter model speed...")
print("=" * 70)

# Create config dict
config = {k: v for k, v in vars(cfg).items() if not k.startswith('_')}

print(f"\nModel configuration:")
print(f"  Layers: {config['n_layers']}")
print(f"  Model dimension: {config['d_model']}")
print(f"  Heads: {config['n_heads']}")
print(f"  Context length: {config['context_length']}")
print(f"  Batch size: {config['batch_size']}")

# Create model
print("\nCreating model...")
model = create_model(config)
mx.eval(model.parameters())

print("\nParameter count:")
count_parameters(model)

# Test forward pass speed
print("\n" + "=" * 70)
print("Speed test: 10 training iterations")
print("=" * 70)

batch_size = config['batch_size']
context_length = config['context_length']
vocab_size = config['vocab_size']

# Create optimizer
import mlx.optimizers as optim
optimizer = optim.AdamW(learning_rate=3e-4)

# Loss function
def loss_fn(model, x, y):
    _, loss = model(x, y)
    return loss

loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

# Warmup
print("\nWarming up...")
for _ in range(3):
    x = mx.random.randint(0, vocab_size, (batch_size, context_length))
    y = mx.random.randint(0, vocab_size, (batch_size, context_length))
    loss, grads = loss_and_grad_fn(model, x, y)
    optimizer.update(model, grads)
    mx.eval(model.parameters(), loss)

# Actual test
print("Running 10 iterations...")
start = time.time()
losses = []

for i in range(10):
    iter_start = time.time()
    
    x = mx.random.randint(0, vocab_size, (batch_size, context_length))
    y = mx.random.randint(0, vocab_size, (batch_size, context_length))
    
    loss, grads = loss_and_grad_fn(model, x, y)
    optimizer.update(model, grads)
    mx.eval(model.parameters(), loss)
    
    iter_time = time.time() - iter_start
    tokens_per_sec = (batch_size * context_length) / iter_time
    
    losses.append(float(loss))
    print(f"  Iter {i+1}/10: {iter_time:.3f}s | loss: {float(loss):.4f} | {tokens_per_sec:.0f} tok/s")

total_time = time.time() - start
avg_time_per_iter = total_time / 10
tokens_per_sec = (10 * batch_size * context_length) / total_time

print("\n" + "=" * 70)
print("Results:")
print(f"  Total time: {total_time:.2f}s")
print(f"  Average per iteration: {avg_time_per_iter:.3f}s")
print(f"  Throughput: {tokens_per_sec:.0f} tokens/sec")
print(f"  Expected time for 10K iters: {(avg_time_per_iter * 10000) / 3600:.1f} hours")
print("=" * 70)

if avg_time_per_iter < 5:
    print("\n✅ GOOD! Training should be fast (~14 hours for 10K iterations)")
elif avg_time_per_iter < 15:
    print("\n⚠️  MODERATE: Training will take ~40 hours for 10K iterations")
else:
    print("\n❌ SLOW: Training will take >40 hours. Consider reducing model size further.")
