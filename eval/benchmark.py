"""
Comprehensive benchmark suite for model evaluation
Combines perplexity, throughput, and memory analysis
"""
import argparse
import json
import time
from pathlib import Path
from datetime import datetime

from src.model import create_model
from src.data import TinyStoriesDataset
from src.utils import load_checkpoint
from src import config as cfg

# Import eval modules
import sys
sys.path.append(str(Path(__file__).parent))
from perplexity import calculate_perplexity
from throughput import measure_throughput, measure_generation_throughput
from memory import analyze_model_memory, estimate_activation_memory


def run_full_benchmark(checkpoint_path, config, args):
    """Run complete benchmark suite"""
    results = {
        'timestamp': datetime.now().isoformat(),
        'checkpoint': str(checkpoint_path),
        'config': config
    }
    
    # Load model
    print("\n" + "=" * 70)
    print("Loading Model")
    print("=" * 70)
    model = create_model(config)
    iteration, loss = load_checkpoint(checkpoint_path, model)
    results['checkpoint_iteration'] = iteration
    results['checkpoint_loss'] = loss
    
    # Memory Analysis
    print("\n" + "=" * 70)
    print("Memory Analysis")
    print("=" * 70)
    memory_results = analyze_model_memory(model)
    activation_results = estimate_activation_memory(config, batch_size=args.batch_size)
    results['memory'] = {**memory_results, **activation_results}
    
    # Throughput Measurement
    print("\n" + "=" * 70)
    print("Throughput Measurement")
    print("=" * 70)
    throughput_results = measure_throughput(
        model,
        batch_size=args.batch_size,
        seq_length=config['context_length'],
        num_iterations=args.throughput_iters,
        warmup=args.warmup
    )
    results['throughput'] = throughput_results
    
    # Generation Throughput
    if args.include_generation:
        print("\n" + "=" * 70)
        print("Generation Throughput")
        print("=" * 70)
        gen_results = measure_generation_throughput(
            model,
            prompt_length=50,
            max_new_tokens=100,
            num_iterations=10
        )
        results['generation'] = gen_results
    
    # Perplexity Evaluation
    if args.include_perplexity:
        for split in args.eval_splits:
            print("\n" + "=" * 70)
            print(f"Perplexity Evaluation - {split.upper()}")
            print("=" * 70)
            
            dataset = TinyStoriesDataset(config, split=split)
            perplexity_results = calculate_perplexity(
                model,
                dataset,
                batch_size=args.batch_size,
                max_batches=args.perplexity_batches
            )
            results[f'perplexity_{split}'] = perplexity_results
    
    return results


def print_summary(results):
    """Print benchmark summary"""
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    
    print(f"\nCheckpoint: {results['checkpoint']}")
    print(f"Iteration: {results['checkpoint_iteration']}")
    print(f"Training Loss: {results['checkpoint_loss']:.4f}")
    
    print(f"\n📊 Model Size:")
    mem = results['memory']
    print(f"  Parameters: {mem['total_parameters']:,} ({mem['total_parameters']/1e6:.1f}M)")
    print(f"  Memory (fp32): {mem['param_memory_mb_fp32']:.1f} MB")
    print(f"  Memory (fp16): {mem['param_memory_mb_fp16']:.1f} MB")
    
    print(f"\n⚡ Throughput:")
    tp = results['throughput']
    print(f"  Forward pass: {tp['tokens_per_second']:.0f} tokens/sec")
    print(f"  Latency (avg): {tp['avg_latency_ms']:.2f}ms")
    print(f"  Latency (p95): {tp['p95_latency_ms']:.2f}ms")
    
    if 'generation' in results:
        gen = results['generation']
        print(f"\n🎯 Generation:")
        print(f"  Throughput: {gen['generation_tokens_per_second']:.1f} tokens/sec")
        print(f"  Latency: {gen['avg_generation_latency_s']:.2f}s per 100 tokens")
    
    if 'perplexity_validation' in results:
        ppl = results['perplexity_validation']
        print(f"\n📈 Perplexity (Validation):")
        print(f"  Perplexity: {ppl['perplexity']:.2f}")
        print(f"  Loss: {ppl['loss']:.4f}")
    
    if 'perplexity_test' in results:
        ppl = results['perplexity_test']
        print(f"\n📈 Perplexity (Test):")
        print(f"  Perplexity: {ppl['perplexity']:.2f}")
        print(f"  Loss: {ppl['loss']:.4f}")
    
    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Comprehensive model benchmark")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--output", type=str, default="benchmark_results.json",
                       help="Output file for results")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size for evaluation")
    parser.add_argument("--throughput_iters", type=int, default=100,
                       help="Iterations for throughput test")
    parser.add_argument("--perplexity_batches", type=int, default=500,
                       help="Batches for perplexity evaluation")
    parser.add_argument("--warmup", type=int, default=10,
                       help="Warmup iterations")
    parser.add_argument("--eval_splits", nargs="+", default=["validation"],
                       choices=["train", "validation", "test"],
                       help="Dataset splits to evaluate")
    parser.add_argument("--skip_perplexity", action="store_true",
                       help="Skip perplexity evaluation")
    parser.add_argument("--skip_generation", action="store_true",
                       help="Skip generation throughput")
    
    args = parser.parse_args()
    args.include_perplexity = not args.skip_perplexity
    args.include_generation = not args.skip_generation
    
    # Load config
    config = {k: v for k, v in vars(cfg).items() if not k.startswith('_')}
    
    print("=" * 70)
    print("GPT MODEL BENCHMARK")
    print("=" * 70)
    print(f"\nCheckpoint: {args.checkpoint}")
    print(f"Output: {args.output}")
    print(f"Batch size: {args.batch_size}")
    
    # Run benchmark
    start_time = time.time()
    results = run_full_benchmark(args.checkpoint, config, args)
    total_time = time.time() - start_time
    results['benchmark_time_seconds'] = total_time
    
    # Print summary
    print_summary(results)
    
    # Save results
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {output_path}")
    print(f"✓ Total benchmark time: {total_time:.1f}s")


if __name__ == "__main__":
    main()
