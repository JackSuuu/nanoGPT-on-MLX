"""
Knowledge Distillation: Train small model to mimic larger model
Uses Groq API (openai/gpt-oss-20b) as teacher model
"""
import os
import sys
import time
import json
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from tqdm import tqdm
from groq import Groq
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from . import config as cfg
from .model import create_model
from .data import create_datasets
from .utils import save_checkpoint, load_checkpoint, get_lr


def get_teacher_logits_from_groq(client, prompt, max_tokens=100, dataset_type='finewebedu'):
    """
    Get predictions from teacher model using Groq API
    
    Args:
        client: Groq client
        prompt: Input text
        max_tokens: Maximum tokens to generate
        dataset_type: Type of dataset ('tinystories' or 'finewebedu') to adjust system prompt
        
    Returns:
        Generated text from teacher model
    """
    # Adjust system prompt based on dataset type
    if dataset_type == 'tinystories':
        system_prompt = "You are a creative storyteller. Continue the given story naturally."
    else:  # finewebedu or other diverse content
        system_prompt = "You are a knowledgeable assistant. Continue the given text naturally, maintaining its style and context."
    
    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",  # Using GPT-OSS-20B as teacher (20B params)
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Continue this text naturally:\n\n{prompt}"}
            ],
            max_tokens=max_tokens,
            temperature=1.0,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error getting teacher response: {e}")
        return prompt  # Return original if error


def distillation_loss_fn(model, teacher_tokens, student_x, student_y, alpha=0.7, temperature=2.0):
    """
    Distillation loss combines:
    1. Hard loss: Regular cross-entropy with true labels (30%)
    2. Soft loss: Mimicking teacher behavior (70%)
    
    Args:
        model: Student model
        teacher_tokens: Tokens from teacher completion
        student_x: Input tokens for student
        student_y: Target tokens for student
        alpha: Weight for hard loss (1-alpha for soft loss)
        temperature: Temperature for softening distributions
        
    Returns:
        Combined loss
    """
    # Get student predictions
    logits, hard_loss = model(student_x, student_y)
    
    # For soft loss, we want student to generate similar to teacher
    # Since we have teacher's completion, we can compute likelihood
    if teacher_tokens is not None and len(teacher_tokens) > len(student_x[0]):
        # Use teacher's continuation as soft targets
        teacher_continuation = teacher_tokens[len(student_x[0]):]
        
        # Get student's predictions for teacher's tokens
        # This makes student learn to predict what teacher predicted
        if len(teacher_continuation) > 0:
            # Truncate to match lengths
            max_len = min(len(teacher_continuation), logits.shape[1])
            teacher_target = mx.array(teacher_continuation[:max_len])
            
            # Soft loss: encourage student to predict teacher's choices
            soft_logits = logits[0, :max_len, :] / temperature
            soft_loss = nn.losses.cross_entropy(
                soft_logits, 
                teacher_target,
                reduction='mean'
            )
            
            # Combined loss
            total_loss = alpha * hard_loss + (1 - alpha) * soft_loss
            return total_loss, hard_loss, soft_loss
    
    # If no teacher tokens, just use hard loss
    return hard_loss, hard_loss, mx.array(0.0)


def distill(resume_from=None, groq_api_key=None):
    """
    Main distillation training loop
    
    Args:
        resume_from: Path to checkpoint to resume from
        groq_api_key: Groq API key for teacher model (optional, loads from .env if not provided)
    """
    # Load API key from .env if not provided
    if not groq_api_key:
        groq_api_key = os.getenv("GROQ_API_KEY")
    
    if not groq_api_key:
        print("❌ Error: Groq API key required for distillation")
        print("   Create a .env file with GROQ_API_KEY=your_key_here")
        print("   Or pass --api-key argument")
        return
    
    # Initialize Groq client
    client = Groq(api_key=groq_api_key)
    
    # Convert config module to dict
    config = {k: v for k, v in vars(cfg).items() if not k.startswith('_')}
    
    # Distillation-specific settings
    distill_config = {
        'max_iters': config.get('max_iters', 10000) + 5000,  # Add 5K distillation iterations
        'distill_alpha': 0.7,  # 70% hard loss, 30% soft loss
        'distill_temperature': 2.0,
        'teacher_samples_per_iter': 1,  # Generate 1 teacher sample per iteration
        'teacher_prompt_length': 64,  # Use 64 tokens as prompt for teacher
        'teacher_max_tokens': 100,  # Teacher generates up to 100 tokens
    }
    config.update(distill_config)
    
    mx.random.seed(config['seed'])
    
    print("=" * 70)
    print("Knowledge Distillation Training")
    print("=" * 70)
    print()
    print("Teacher Model: GPT-OSS-20B (20B params) via Groq API")
    print("Student Model: 22M parameter GPT")
    print(f"Distillation Alpha: {config['distill_alpha']} (hard loss weight)")
    print(f"Temperature: {config['distill_temperature']}")
    print()
    
    # Load datasets
    print("[1/4] Loading data...")
    train_dataset, val_dataset = create_datasets(config)
    
    # Create student model
    print("\n[2/4] Initializing student model...")
    model = create_model(config)
    mx.eval(model.parameters())
    
    # Create optimizer
    print("\n[3/4] Setting up optimizer...")
    optimizer = optim.AdamW(
        learning_rate=config['learning_rate'] * 0.5,  # Lower LR for distillation
        weight_decay=config['weight_decay']
    )
    
    # Resume from checkpoint
    start_iteration = 0
    best_val_loss = float('inf')
    
    if resume_from:
        print(f"\n[*] Resuming from checkpoint: {resume_from}")
        checkpoint_iter, checkpoint_loss = load_checkpoint(resume_from, model, optimizer)
        start_iteration = checkpoint_iter + 1
        best_val_loss = checkpoint_loss
        print(f"    Starting distillation from iteration {start_iteration}")
        print(f"    Base model loss: {best_val_loss:.4f}")
    
    # Training loop
    print("\n[4/4] Starting distillation...")
    print("=" * 70)
    print()
    
    start_time = time.time()
    teacher_cache = {}  # Cache teacher responses to save API calls
    
    # Progress bar
    pbar = tqdm(range(start_iteration, config['max_iters']), 
                desc="Distillation", unit="iter",
                initial=start_iteration, total=config['max_iters'])
    
    for iteration in pbar:
        # Learning rate schedule
        lr = float(get_lr(iteration, config)) * 0.5  # Half LR for fine-tuning
        optimizer.learning_rate = lr
        
        # Get batch
        x, y = train_dataset.get_batch(config['batch_size'])
        
        # Every N iterations, get teacher's prediction
        use_teacher = (iteration % 10 == 0)  # Use teacher every 10 iterations
        teacher_tokens = None
        
        if use_teacher:
            # Get prompt from first sequence in batch
            prompt_tokens = x[0, :config['teacher_prompt_length']].tolist()
            
            # Decode prompt
            import tiktoken
            tokenizer = tiktoken.get_encoding("gpt2")
            prompt_text = tokenizer.decode(prompt_tokens)
            
            # Check cache
            cache_key = prompt_text[:100]  # Use first 100 chars as key
            if cache_key in teacher_cache:
                teacher_text = teacher_cache[cache_key]
            else:
                # Get teacher's completion (with dataset-specific prompt)
                dataset_type = config.get('dataset_name', 'finewebedu')
                teacher_text = get_teacher_logits_from_groq(
                    client, 
                    prompt_text,
                    max_tokens=config['teacher_max_tokens'],
                    dataset_type=dataset_type
                )
                teacher_cache[cache_key] = teacher_text
                
                # Limit cache size
                if len(teacher_cache) > 1000:
                    teacher_cache.pop(next(iter(teacher_cache)))
            
            # Encode teacher's completion
            try:
                teacher_tokens = tokenizer.encode(prompt_text + teacher_text)
            except:
                teacher_tokens = None
        
        # Forward and backward pass with distillation
        if teacher_tokens and use_teacher:
            def distill_loss(m, tx, sx, sy):
                return distillation_loss_fn(
                    m, tx, sx, sy, 
                    alpha=config['distill_alpha'],
                    temperature=config['distill_temperature']
                )[0]
            
            loss_and_grad_fn = nn.value_and_grad(model, distill_loss)
            loss, grads = loss_and_grad_fn(model, teacher_tokens, x, y)
        else:
            # Regular training without teacher
            def regular_loss(m, x, y):
                _, loss = m(x, y)
                return loss
            loss_and_grad_fn = nn.value_and_grad(model, regular_loss)
            loss, grads = loss_and_grad_fn(model, x, y)
        
        # Update parameters
        optimizer.update(model, grads)
        mx.eval(model.parameters(), loss)
        
        # Update progress
        elapsed = time.time() - start_time
        tokens_per_sec = (iteration - start_iteration + 1) * config['batch_size'] * config['context_length'] / elapsed
        
        pbar.set_postfix({
            'loss': f'{float(loss):.4f}',
            'lr': f'{lr:.2e}',
            'tok/s': f'{tokens_per_sec:.0f}',
            'teacher': '✓' if use_teacher else '✗'
        }, refresh=True)
        
        # Logging
        if iteration % 100 == 0:
            tqdm.write(f"\niter {iteration:5d} | loss {float(loss):.4f} | lr {lr:.2e} | {tokens_per_sec:.0f} tok/s")
        
        # Evaluation
        should_eval = (iteration > start_iteration and iteration % config['eval_interval'] == 0)
        
        if should_eval:
            from .data import estimate_loss
            
            tqdm.write("\n" + "-" * 70)
            tqdm.write("Evaluating...")
            
            train_loss = estimate_loss(model, train_dataset, config)
            val_loss = estimate_loss(model, val_dataset, config)
            
            tqdm.write(f"iter {iteration:5d} | train {train_loss:.4f} | val {val_loss:.4f}")
            tqdm.write("-" * 70 + "\n")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(model, optimizer, iteration, val_loss, config,
                              config['checkpoint_dir'])
                tqdm.write(f"✓ Saved best model (val loss: {best_val_loss:.4f})\n")
        
        # Save checkpoint periodically
        if iteration > start_iteration and iteration % config['save_interval'] == 0:
            save_checkpoint(model, optimizer, iteration, float(loss), config,
                          config['checkpoint_dir'])
            tqdm.write(f"✓ Checkpoint saved at iteration {iteration}\n")
    
    pbar.close()
    
    # Final save
    if start_iteration < config['max_iters']:
        print("\n" + "=" * 70)
        print("Distillation complete!")
        save_checkpoint(model, optimizer, config['max_iters'], float(loss), config,
                       config['checkpoint_dir'])
        
        total_time = time.time() - start_time
        print(f"Distillation time: {total_time/60:.2f} minutes")
        print(f"Best validation loss: {best_val_loss:.4f}")
        print(f"Teacher samples generated: ~{(config['max_iters'] - start_iteration) // 10}")
        print("=" * 70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Distill knowledge from larger model')
    parser.add_argument('--resume', type=str, required=True,
                      help='Checkpoint to resume from (e.g., checkpoints/checkpoint_10000.npz)')
    parser.add_argument('--groq-api-key', type=str, default=None,
                      help='Groq API key (or set GROQ_API_KEY env var)')
    
    args = parser.parse_args()
    
    # Get API key from args or environment
    api_key = args.groq_api_key or os.getenv('GROQ_API_KEY')
    
    if not api_key:
        print("❌ Error: Groq API key required")
        print("\nProvide via:")
        print("  1. Command line: --groq-api-key YOUR_KEY")
        print("  2. Environment: export GROQ_API_KEY=YOUR_KEY")
        sys.exit(1)
    
    distill(resume_from=args.resume, groq_api_key=api_key)
