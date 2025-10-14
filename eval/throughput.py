"""
Measure model throughput (tokens/second, batches/second)
"""
import argparse
import time
import numpy as np
import mlx.core as mx
from pathlib import Path

from src.model import create_model
from src.utils import load_checkpoint
from src import config as cfg


def measure_throughput(model, batch_size, seq_length, num_iterations=100, warmup=10):
    """
    Measure model throughput
    
    Returns:
        - tokens_per_second: Forward pass throughput
        - batches_per_second: Batch throughput
        - avg_latency_ms: Average latency per batch
    """
    print(f"Measuring throughput:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_length}")
    print(f"  Iterations: {num_iterations}")
    
    # Create random input
    vocab_size = model.vocab_size
    
    # Warmup
    print(f"\nWarming up ({warmup} iterations)...")
    for i in range(warmup):
        x = mx.random.randint(0, vocab_size, (batch_size, seq_length))
        logits, _ = model(x)
        mx.eval(logits)
    
    # Benchmark
    print(f"\nBenchmarking ({num_iterations} iterations)...")
    latencies = []
    
    for i in range(num_iterations):
        x = mx.random.randint(0, vocab_size, (batch_size, seq_length))
        
        start = time.time()
        logits, _ = model(x)
        mx.eval(logits)
        elapsed = time.time() - start
        
        latencies.append(elapsed)
        
        if (i + 1) % 20 == 0:
            avg_latency = np.mean(latencies[-20:])
            print(f"  Iter {i+1}/{num_iterations} - Latency: {avg_latency*1000:.2f}ms")
    
    # Calculate metrics
    latencies = np.array(latencies)
    avg_latency = np.mean(latencies)
    std_latency = np.std(latencies)
    p50_latency = np.percentile(latencies, 50)
    p95_latency = np.percentile(latencies, 95)
    p99_latency = np.percentile(latencies, 99)
    
    tokens_per_batch = batch_size * seq_length
    tokens_per_second = tokens_per_batch / avg_latency
    batches_per_second = 1.0 / avg_latency
    
    print(f"\nResults:")
    print(f"  Average latency: {avg_latency*1000:.2f}ms (±{std_latency*1000:.2f}ms)")
    print(f"  P50 latency: {p50_latency*1000:.2f}ms")
    print(f"  P95 latency: {p95_latency*1000:.2f}ms")
    print(f"  P99 latency: {p99_latency*1000:.2f}ms")
    print(f"  Throughput: {tokens_per_second:.0f} tokens/sec")
    print(f"  Throughput: {batches_per_second:.2f} batches/sec")
    
    return {
        'avg_latency_ms': float(avg_latency * 1000),
        'std_latency_ms': float(std_latency * 1000),
        'p50_latency_ms': float(p50_latency * 1000),
        'p95_latency_ms': float(p95_latency * 1000),
        'p99_latency_ms': float(p99_latency * 1000),
        'tokens_per_second': float(tokens_per_second),
        'batches_per_second': float(batches_per_second)
    }


def measure_generation_throughput(model, prompt_length=50, max_new_tokens=100, 
                                   num_iterations=10, warmup=2):
    """Measure generation throughput"""
    print(f"\nMeasuring generation throughput:")
    print(f"  Prompt length: {prompt_length}")
    print(f"  Max new tokens: {max_new_tokens}")
    print(f"  Iterations: {num_iterations}")
    
    vocab_size = model.vocab_size
    
    # Warmup
    print(f"\nWarming up ({warmup} iterations)...")
    for i in range(warmup):
        prompt = mx.random.randint(0, vocab_size, (1, prompt_length))
        output = model.generate(prompt, max_new_tokens=max_new_tokens)
        mx.eval(output)
    
    # Benchmark
    print(f"\nBenchmarking ({num_iterations} iterations)...")
    latencies = []
    
    for i in range(num_iterations):
        prompt = mx.random.randint(0, vocab_size, (1, prompt_length))
        
        start = time.time()
        output = model.generate(prompt, max_new_tokens=max_new_tokens)
        mx.eval(output)
        elapsed = time.time() - start
        
        latencies.append(elapsed)
        tokens_generated = output.shape[1] - prompt_length
        print(f"  Iter {i+1}/{num_iterations} - {elapsed:.2f}s ({tokens_generated/elapsed:.1f} tok/s)")
    
    # Calculate metrics
    latencies = np.array(latencies)
    avg_latency = np.mean(latencies)
    tokens_per_second = max_new_tokens / avg_latency
    
    print(f"\nGeneration Results:")
    print(f"  Average latency: {avg_latency:.2f}s")
    print(f"  Throughput: {tokens_per_second:.1f} tokens/sec")
    
    return {
        'avg_generation_latency_s': float(avg_latency),
        'generation_tokens_per_second': float(tokens_per_second)
    }


def main():
    parser = argparse.ArgumentParser(description="Measure model throughput")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size for throughput test")
    parser.add_argument("--seq_length", type=int, default=1024,
                       help="Sequence length for throughput test")
    parser.add_argument("--iterations", type=int, default=100,
                       help="Number of iterations")
    parser.add_argument("--warmup", type=int, default=10,
                       help="Number of warmup iterations")
    parser.add_argument("--generation", action="store_true",
                       help="Also measure generation throughput")
    
    args = parser.parse_args()
    
    # Load config
    config = {k: v for k, v in vars(cfg).items() if not k.startswith('_')}
    
    print("=" * 70)
    print("Model Throughput Evaluation")
    print("=" * 70)
    
    # Create model
    print("\nLoading model...")
    model = create_model(config)
    load_checkpoint(args.checkpoint, model)
    
    # Measure forward pass throughput
    print("\n" + "=" * 70)
    print("Forward Pass Throughput")
    print("=" * 70)
    forward_results = measure_throughput(
        model,
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        num_iterations=args.iterations,
        warmup=args.warmup
    )
    
    # Measure generation throughput
    if args.generation:
        print("\n" + "=" * 70)
        print("Generation Throughput")
        print("=" * 70)
        gen_results = measure_generation_throughput(model)
        forward_results.update(gen_results)
    
    print("\n" + "=" * 70)
    
    return forward_results


if __name__ == "__main__":
    main()
