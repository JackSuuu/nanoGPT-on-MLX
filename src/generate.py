"""
Text generation script for trained GPT model
"""
import argparse
import mlx.core as mx
import tiktoken
from pathlib import Path

from .model import create_model
from .utils import load_checkpoint
from . import config as cfg


def generate_text(model, tokenizer, prompt, max_tokens=100, temperature=1.0, top_k=50):
    """
    Generate text from a prompt
    
    Args:
        model: Trained GPT model
        tokenizer: Tokenizer
        prompt: Input text prompt
        max_tokens: Number of tokens to generate
        temperature: Sampling temperature (higher = more random)
        top_k: Sample from top-k tokens (None = sample from all)
    """
    # Encode prompt
    tokens = tokenizer.encode(prompt)
    tokens = mx.array([tokens])  # Add batch dimension
    
    print(f"Prompt: {prompt}")
    print(f"\nGenerating {max_tokens} tokens...")
    print("-" * 70)
    
    # Generate
    output_tokens = model.generate(
        tokens, 
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k
    )
    
    # Decode
    generated_tokens = output_tokens[0].tolist()
    generated_text = tokenizer.decode(generated_tokens)
    
    print(generated_text)
    print("-" * 70)
    
    return generated_text


def main():
    parser = argparse.ArgumentParser(description="Generate text with trained GPT model")
    parser.add_argument("--checkpoint", type=str, required=True, 
                       help="Path to model checkpoint")
    parser.add_argument("--prompt", type=str, default="The meaning of life is",
                       help="Text prompt for generation")
    parser.add_argument("--max_tokens", type=int, default=100,
                       help="Number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8,
                       help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=50,
                       help="Top-k sampling parameter")
    
    args = parser.parse_args()
    
    # Load config
    config = {k: v for k, v in vars(cfg).items() if not k.startswith('_')}
    
    print("=" * 70)
    print("GPT Text Generation")
    print("=" * 70)
    
    # Create model
    print("\nInitializing model...")
    model = create_model(config)
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    load_checkpoint(args.checkpoint, model)
    
    # Initialize tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")
    
    # Generate text
    print("\n" + "=" * 70)
    generate_text(
        model, 
        tokenizer, 
        args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k
    )
    print("=" * 70)


def interactive_mode():
    """Interactive generation mode"""
    # Load config
    config = {k: v for k, v in vars(cfg).items() if not k.startswith('_')}
    
    # Find latest checkpoint
    checkpoint_dir = Path(config['checkpoint_dir'])
    if not checkpoint_dir.exists():
        print(f"No checkpoints found in {checkpoint_dir}")
        return
    
    checkpoints = sorted(checkpoint_dir.glob("checkpoint_*.npz"))
    if not checkpoints:
        print(f"No checkpoints found in {checkpoint_dir}")
        return
    
    latest_checkpoint = checkpoints[-1]
    
    print("=" * 70)
    print("GPT Interactive Generation")
    print("=" * 70)
    
    # Create model
    print("\nInitializing model...")
    model = create_model(config)
    
    # Load checkpoint
    print(f"Loading checkpoint from {latest_checkpoint}...")
    load_checkpoint(str(latest_checkpoint), model)
    
    # Initialize tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")
    
    print("\n" + "=" * 70)
    print("Interactive mode (type 'quit' to exit)")
    print("=" * 70 + "\n")
    
    while True:
        prompt = input("Prompt: ").strip()
        
        if prompt.lower() in ['quit', 'exit', 'q']:
            break
        
        if not prompt:
            continue
        
        print()
        generate_text(model, tokenizer, prompt, max_tokens=100, temperature=0.8, top_k=50)
        print()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) == 1 or sys.argv[1] == "--interactive":
        interactive_mode()
    else:
        main()
