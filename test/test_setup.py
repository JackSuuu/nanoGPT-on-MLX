"""
Quick test script to verify model setup
"""
import mlx.core as mx

print("=" * 70)
print("Testing nanoGPT on MLX Setup")
print("=" * 70)

# Test 1: MLX installation
print("\n[1/3] Testing MLX installation...")
try:
    a = mx.array([1, 2, 3])
    b = mx.array([4, 5, 6])
    c = a + b
    print(f"✓ MLX working: {a} + {b} = {c}")
except Exception as e:
    print(f"✗ MLX error: {e}")
    exit(1)

# Test 2: Model creation
print("\n[2/3] Testing model creation...")
try:
    from src import config as cfg
    from src.model import create_model
    from src.utils import count_parameters
    
    config = {k: v for k, v in vars(cfg).items() if not k.startswith('_')}
    model = create_model(config)
    mx.eval(model.parameters())
    
    print("✓ Model created successfully")
    print("\nModel configuration:")
    print(f"  - Layers: {config['n_layers']}")
    print(f"  - Heads: {config['n_heads']}")
    print(f"  - Embedding dim: {config['d_model']}")
    print(f"  - FFN dim: {config['d_ff']}")
    print(f"  - Context length: {config['context_length']}")
    print(f"  - Vocab size: {config['vocab_size']}")
    
    print("\nParameter count:")
    total_params = count_parameters(model)
    
    # Account for weight tying (embedding weights shared with output layer)
    embedding_params = config['vocab_size'] * config['d_model']
    unique_params = total_params - embedding_params
    
    print(f"\nWith weight tying (embedding = output layer):")
    print(f"  Unique parameters: {unique_params:,} ({unique_params/1e6:.2f}M)")
    
    if 120_000_000 <= unique_params <= 130_000_000:
        print(f"\n✓ Unique parameter count within target range (120M-130M)")
    else:
        print(f"\n⚠ Unique parameter count: {unique_params/1e6:.1f}M (target: 120-130M)")
        
except Exception as e:
    print(f"✗ Model creation error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 3: Forward pass
print("\n[3/3] Testing forward pass...")
try:
    batch_size = 2
    seq_len = 64
    
    # Random input
    x = mx.random.randint(0, config['vocab_size'], (batch_size, seq_len))
    
    # Forward pass
    logits, _ = model(x)
    
    print(f"✓ Forward pass successful")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {logits.shape}")
    print(f"  Expected: ({batch_size}, {seq_len}, {config['vocab_size']})")
    
    assert logits.shape == (batch_size, seq_len, config['vocab_size']), "Output shape mismatch"
    print("✓ Output shape correct")
    
except Exception as e:
    print(f"✗ Forward pass error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "=" * 70)
print("All tests passed! ✓")
print("=" * 70)
print("\nNext steps:")
print("1. Test data loading: python -m src.data")
print("2. Start training: python -m src.train")
print("3. Generate text: python -m src.generate --interactive")
print("4. Run benchmarks: python -m eval.benchmark --checkpoint <path>")
print("=" * 70)
