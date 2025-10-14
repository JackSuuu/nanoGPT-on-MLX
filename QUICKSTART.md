# Quick Reference Guide

## Project Commands

### Setup
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Test setup
python test_setup.py
```

### Training
```bash
# Test data loading
python -m src.data

# Start fresh training
python -m src.train

# Resume from checkpoint (if interrupted)
python -m src.train --resume checkpoints/checkpoint_5000.npz

# Extend training (train more iterations):
# 1. Edit src/config.py: max_iters = 15000
# 2. Resume from last checkpoint
python -m src.train --resume checkpoints/checkpoint_10000.npz

# Training features:
# - Real-time progress bar (loss, lr, throughput)
# - Auto-saves every 1000 iterations
# - Evaluates every 500 iterations
# - Tracks best validation loss
# - Resume capability from any checkpoint
```

### Knowledge Distillation (Optional)
```bash
# Setup: Create .env file with Groq API key
cp .env.example .env
# Edit .env and add: GROQ_API_KEY=your_key_here
# Get free key at: https://console.groq.com/

# Run distillation from trained checkpoint
python -m src.distill --resume checkpoints/checkpoint_10000.npz

# Benefits:
# - 60% quality improvement vs base model
# - Uses GPT-OSS-20B (20B) as teacher
# - 5K-10K iterations (~1-2 hours)
# - Much better text generation
```

### Generation
```bash
# Interactive mode (uses latest checkpoint)
python -m src.generate --interactive

# Specify checkpoint
python -m src.generate \
    --checkpoint checkpoints/checkpoint_10000.npz \
    --prompt "Once upon a time" \
    --max_tokens 100 \
    --temperature 0.8 \
    --top_k 50
```

### Evaluation

#### Comprehensive Benchmark
```bash
python -m eval.benchmark --checkpoint checkpoints/checkpoint_10000.npz
```

#### Individual Evaluations
```bash
# Perplexity
python -m eval.perplexity \
    --checkpoint checkpoints/checkpoint_10000.npz \
    --split validation

# Throughput
python -m eval.throughput \
    --checkpoint checkpoints/checkpoint_10000.npz \
    --batch_size 8 \
    --generation

# Memory Analysis
python -m eval.memory \
    --test_batches \
    --available_memory 16
```

## File Structure

```
nanoGPT-on-MLX/
├── src/                    # Core training code
│   ├── config.py          # Model configuration
│   ├── model.py           # GPT architecture
│   ├── data.py            # Data loading
│   ├── train.py           # Training loop
│   ├── generate.py        # Text generation
│   └── utils.py           # Utilities
├── eval/                   # Evaluation tools
│   ├── benchmark.py       # Full benchmark suite
│   ├── perplexity.py      # Model quality
│   ├── throughput.py      # Performance
│   └── memory.py          # Memory profiling
├── data/                   # Cached datasets (auto-created)
├── checkpoints/            # Saved models (auto-created)
└── test_setup.py          # Setup verification
```

## Configuration

Edit `src/config.py` to adjust:

```python
# Model size
n_layers = 12           # Number of transformer layers
n_heads = 12            # Number of attention heads
d_model = 768           # Embedding dimension
d_ff = 3072             # Feedforward dimension

# Training
batch_size = 8          # Adjust for your memory
learning_rate = 3e-4
max_iters = 10000       # Training iterations
max_tokens = 1_000_000  # Dataset size (1M tokens)

# Context
context_length = 1024   # Maximum sequence length
```

## Memory Management

**For 16GB M2 Pro:**
- Batch size 8: ~5-6GB memory usage
- Batch size 4: ~3-4GB memory usage
- Batch size 2: ~2-3GB memory usage

Use `python -m eval.memory --test_batches` to find optimal batch size.

## Expected Performance

**M2 Pro 16GB:**
- Training speed: 10,000-15,000 tokens/sec
- Generation speed: 30-50 tokens/sec
- Perplexity (10K iters): ~40-60 (validation)

**Training time (10K iterations):**
- ~2-4 hours with batch_size=8
- ~4-6 hours with batch_size=4

## Common Issues

### Out of Memory
```bash
# Reduce batch size in src/config.py
batch_size = 4  # or 2

# Reduce context length
context_length = 512  # or 256
```

### Slow Training
```bash
# Check MLX is using GPU
python -c "import mlx.core as mx; print(mx.metal.is_available())"

# Should print: True
```

### Import Errors
```bash
# Make sure you're using module syntax
python -m src.train  # ✓ Correct
python src/train.py  # ✗ Won't work with relative imports
```

## Tips

1. **Start small**: Test with `max_iters=1000` first
2. **Monitor memory**: Use Activity Monitor during training
3. **Save frequently**: Checkpoints saved every 1000 iterations
4. **Use benchmarks**: Run `eval.benchmark` after training
5. **Experiment**: Try different temperatures (0.7-1.0) for generation
