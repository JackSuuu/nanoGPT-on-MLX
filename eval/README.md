# Evaluation Scripts

This directory contains comprehensive evaluation tools for assessing model performance, throughput, and memory usage.

## Scripts

### 1. `benchmark.py` - Comprehensive Benchmark Suite

Runs all evaluations in one command and saves results to JSON.

**Usage:**
```bash
python -m eval.benchmark --checkpoint checkpoints/checkpoint_10000.npz
```

**Options:**
- `--checkpoint`: Path to model checkpoint (required)
- `--output`: Output JSON file (default: `benchmark_results.json`)
- `--batch_size`: Batch size for evaluation (default: 8)
- `--throughput_iters`: Iterations for throughput test (default: 100)
- `--perplexity_batches`: Batches for perplexity (default: 500)
- `--eval_splits`: Dataset splits to evaluate (default: validation)
- `--skip_perplexity`: Skip perplexity evaluation
- `--skip_generation`: Skip generation throughput

**Example:**
```bash
python -m eval.benchmark \
    --checkpoint checkpoints/checkpoint_10000.npz \
    --eval_splits validation test \
    --output results.json
```

**Output:**
- Comprehensive JSON with all metrics
- Summary printed to console
- Perplexity, throughput, memory, and generation metrics

---

### 2. `perplexity.py` - Model Quality Evaluation

Calculates perplexity on train/validation/test splits.

**Usage:**
```bash
python -m eval.perplexity \
    --checkpoint checkpoints/checkpoint_10000.npz \
    --split validation
```

**Options:**
- `--checkpoint`: Path to model checkpoint (required)
- `--split`: Dataset split (train/validation/test, default: validation)
- `--batch_size`: Batch size (default: 8)
- `--max_batches`: Maximum batches to evaluate (default: all)

**Metrics:**
- **Perplexity**: exp(loss) - lower is better
- **Loss**: Cross-entropy loss
- **Tokens/sec**: Evaluation throughput

**Good perplexity values:**
- Excellent: < 20
- Good: 20-50
- Fair: 50-100
- Poor: > 100

---

### 3. `throughput.py` - Performance Benchmarking

Measures forward pass and generation throughput.

**Usage:**
```bash
python -m eval.throughput \
    --checkpoint checkpoints/checkpoint_10000.npz \
    --generation
```

**Options:**
- `--checkpoint`: Path to model checkpoint (required)
- `--batch_size`: Batch size for testing (default: 8)
- `--seq_length`: Sequence length (default: 1024)
- `--iterations`: Number of iterations (default: 100)
- `--warmup`: Warmup iterations (default: 10)
- `--generation`: Also measure generation throughput

**Metrics:**
- **Tokens/second**: Throughput
- **Batches/second**: Batch processing rate
- **Latency**: Average, P50, P95, P99 latencies
- **Generation speed**: Tokens/second for autoregressive generation

**Example output:**
```
Results:
  Average latency: 45.23ms (±2.15ms)
  P50 latency: 44.80ms
  P95 latency: 48.90ms
  P99 latency: 52.10ms
  Throughput: 18,067 tokens/sec
  Throughput: 2.21 batches/sec
```

---

### 4. `memory.py` - Memory Profiling

Analyzes model memory usage and provides recommendations.

**Usage:**
```bash
python -m eval.memory --test_batches --available_memory 16
```

**Options:**
- `--checkpoint`: Path to checkpoint (optional)
- `--test_batches`: Test different batch sizes
- `--available_memory`: Available memory in GB (default: 16)

**Metrics:**
- **Parameter memory**: Model weights in MB/GB
- **Activation memory**: Forward pass memory per batch
- **Backward pass memory**: Training memory estimate
- **Recommended batch size**: Based on available memory

**Example output:**
```
Model Memory Analysis
----------------------------------------------------------------------
  Total parameters: 124,439,808
  Memory (float32): 497.76 MB
  Memory (float16): 248.88 MB

Activation Memory Estimate
----------------------------------------------------------------------
  Batch size: 8
  Sequence length: 1024
  Per-layer activation: 76.55 MB
  Total layers activation: 918.60 MB
  Embedding memory: 25.17 MB
  Output logits memory: 1650.69 MB
  Total activation memory: 2594.46 MB
  Backward pass (est. 2x): 5188.92 MB

Batch Size Recommendation
----------------------------------------------------------------------
  Available memory: 16 GB
  Parameter memory: 0.47 GB
  Available for activations: 14.53 GB
  Estimated memory per sample: 324.31 MB
  Recommended batch size: 31
```

---

## Quick Start

### Run all evaluations:
```bash
# Complete benchmark
python -m eval.benchmark --checkpoint checkpoints/checkpoint_10000.npz

# With custom settings
python -m eval.benchmark \
    --checkpoint checkpoints/checkpoint_10000.npz \
    --eval_splits validation test \
    --batch_size 4 \
    --output my_results.json
```

### Run individual evaluations:
```bash
# Perplexity only
python -m eval.perplexity --checkpoint checkpoints/checkpoint_10000.npz

# Throughput only
python -m eval.throughput --checkpoint checkpoints/checkpoint_10000.npz --generation

# Memory analysis only
python -m eval.memory --test_batches
```

---

## Understanding the Results

### Perplexity
- **What it measures**: Model's ability to predict next token
- **Lower is better**: Perfect model has perplexity = 1
- **Typical range**: 20-100 for well-trained models
- **Use case**: Compare model quality, detect overfitting

### Throughput
- **What it measures**: Processing speed
- **Higher is better**: More tokens/second = faster
- **M2 Pro expected**: 10,000-20,000 tokens/sec (forward pass)
- **Use case**: Optimize batch size, deployment planning

### Memory
- **What it measures**: RAM requirements
- **Parameters**: Fixed per model size
- **Activations**: Varies with batch size and sequence length
- **Use case**: Choose optimal batch size, hardware planning

---

## Tips

1. **For training**: Use `memory.py` first to find optimal batch size
2. **For deployment**: Use `throughput.py` to estimate serving capacity
3. **For model quality**: Use `perplexity.py` on validation set
4. **For full report**: Use `benchmark.py` with all options

---

## Saving Results

All evaluation scripts support JSON output for tracking experiments:

```bash
# Benchmark saves automatically
python -m eval.benchmark --checkpoint model.npz --output results.json

# For others, redirect output
python -m eval.perplexity --checkpoint model.npz > perplexity.txt
python -m eval.throughput --checkpoint model.npz > throughput.txt
```

Parse JSON results:
```python
import json

with open('benchmark_results.json') as f:
    results = json.load(f)
    
print(f"Perplexity: {results['perplexity_validation']['perplexity']}")
print(f"Throughput: {results['throughput']['tokens_per_second']} tok/s")
```
