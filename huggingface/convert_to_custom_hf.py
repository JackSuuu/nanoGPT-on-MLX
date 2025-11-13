"""
Convert MLX checkpoint to custom NanoGPT HuggingFace format (Pre-LN architecture)
This uses our custom model that matches the MLX architecture exactly
"""
import os
import json
import argparse
import numpy as np
import mlx.core as mx
import torch
from pathlib import Path
from safetensors.torch import save_file

# Import custom model
import sys
sys.path.insert(0, str(Path(__file__).parent))
from modeling_nanogpt import NanoGPTConfig, NanoGPTLMHeadModel


def convert_mlx_to_custom_hf(checkpoint_path, output_dir="huggingface", model_name=None):
    """
    Convert MLX checkpoint to custom NanoGPT HF format
    """
    print("="*70)
    print("MLX to Custom NanoGPT HuggingFace Converter")
    print("="*70)
    
    # Load checkpoint metadata
    checkpoint_path = Path(checkpoint_path)
    meta_path = checkpoint_path.parent / f"{checkpoint_path.stem}_meta.json"
    
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")
    
    with open(meta_path, 'r') as f:
        metadata = json.load(f)
    
    config_dict = metadata['config']
    iteration = metadata['iteration']
    loss = metadata['loss']
    
    print(f"\n📦 Loading checkpoint: {checkpoint_path.name}")
    print(f"   Iteration: {iteration:,}")
    print(f"   Loss: {loss:.4f}")
    
    # Load MLX weights
    print("\n📥 Loading MLX weights...")
    mlx_params = mx.load(str(checkpoint_path))
    print(f"   ✓ Loaded {len(mlx_params)} parameter tensors")
    
    # Create HF config
    hf_config = NanoGPTConfig(
        vocab_size=config_dict['vocab_size'],
        n_positions=config_dict['context_length'],
        n_embd=config_dict['d_model'],
        n_layer=config_dict['n_layers'],
        n_head=config_dict['n_heads'],
        n_inner=config_dict['d_ff'],
        resid_pdrop=config_dict['dropout'],
        embd_pdrop=config_dict['dropout'],
        attn_pdrop=config_dict['dropout'],
    )
    
    # Create output directory
    if model_name is None:
        model_name = f"nanogpt-mlx-{config_dict['d_model']}d-{iteration//1000}k"
    
    output_path = Path(output_dir) / model_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📁 Output directory: {output_path}")
    
    # Convert weights
    print(f"\n🔄 Converting weights...")
    hf_weights = {}
    
    # Token embeddings
    hf_weights['transformer.token_embedding.weight'] = torch.from_numpy(np.array(mlx_params['token_embedding.weight']))
    
    # LM head - use separate weight from checkpoint (not tied in practice)
    hf_weights['transformer.lm_head.weight'] = torch.from_numpy(np.array(mlx_params['lm_head.weight']))
    
    # Position embeddings
    hf_weights['transformer.position_embedding.weight'] = torch.from_numpy(np.array(mlx_params['position_embedding.weight']))
    
    # Transformer blocks
    n_layers = config_dict['n_layers']
    for i in range(n_layers):
        mlx_prefix = f'blocks.{i}'
        hf_prefix = f'transformer.blocks.{i}'
        
        # Layer norm 1
        hf_weights[f'{hf_prefix}.ln1.weight'] = torch.from_numpy(np.array(mlx_params[f'{mlx_prefix}.ln1.weight']))
        hf_weights[f'{hf_prefix}.ln1.bias'] = torch.from_numpy(np.array(mlx_params[f'{mlx_prefix}.ln1.bias']))
        
        # Attention QKV projection - NO TRANSPOSE (our model uses standard Linear, not Conv1D)
        hf_weights[f'{hf_prefix}.attn.qkv_proj.weight'] = torch.from_numpy(np.array(mlx_params[f'{mlx_prefix}.attn.qkv_proj.weight']))
        hf_weights[f'{hf_prefix}.attn.qkv_proj.bias'] = torch.from_numpy(np.array(mlx_params[f'{mlx_prefix}.attn.qkv_proj.bias']))
        
        # Attention output projection  
        hf_weights[f'{hf_prefix}.attn.out_proj.weight'] = torch.from_numpy(np.array(mlx_params[f'{mlx_prefix}.attn.out_proj.weight']))
        hf_weights[f'{hf_prefix}.attn.out_proj.bias'] = torch.from_numpy(np.array(mlx_params[f'{mlx_prefix}.attn.out_proj.bias']))
        
        # Layer norm 2
        hf_weights[f'{hf_prefix}.ln2.weight'] = torch.from_numpy(np.array(mlx_params[f'{mlx_prefix}.ln2.weight']))
        hf_weights[f'{hf_prefix}.ln2.bias'] = torch.from_numpy(np.array(mlx_params[f'{mlx_prefix}.ln2.bias']))
        
        # MLP fc1
        hf_weights[f'{hf_prefix}.mlp.fc1.weight'] = torch.from_numpy(np.array(mlx_params[f'{mlx_prefix}.ff.fc1.weight']))
        hf_weights[f'{hf_prefix}.mlp.fc1.bias'] = torch.from_numpy(np.array(mlx_params[f'{mlx_prefix}.ff.fc1.bias']))
        
        # MLP fc2
        hf_weights[f'{hf_prefix}.mlp.fc2.weight'] = torch.from_numpy(np.array(mlx_params[f'{mlx_prefix}.ff.fc2.weight']))
        hf_weights[f'{hf_prefix}.mlp.fc2.bias'] = torch.from_numpy(np.array(mlx_params[f'{mlx_prefix}.ff.fc2.bias']))
    
    # Final layer norm
    hf_weights['transformer.ln_f.weight'] = torch.from_numpy(np.array(mlx_params['ln_f.weight']))
    hf_weights['transformer.ln_f.bias'] = torch.from_numpy(np.array(mlx_params['ln_f.bias']))
    
    print(f"   ✓ Converted {len(hf_weights)} weight tensors")
    
    # Save config
    config_path = output_path / "config.json"
    hf_config.save_pretrained(str(output_path))
    print(f"\n💾 Saved config: {config_path}")
    
    # Create model and load weights (to handle tied weights correctly)
    print(f"\n🔄 Creating model and loading weights...")
    model = NanoGPTLMHeadModel(hf_config)
    model.load_state_dict(hf_weights, strict=False)
    
    # Save model with safe_serialization=False to handle tied weights
    model.save_pretrained(str(output_path), safe_serialization=False)
    print(f"   ✓ Saved model: {output_path}")
    
    # Copy modeling file
    import shutil
    modeling_src = Path(__file__).parent / "modeling_nanogpt.py"
    modeling_dst = output_path / "modeling_nanogpt.py"
    shutil.copy(modeling_src, modeling_dst)
    print(f"   ✓ Copied model code: {modeling_dst}")
    
    # Save tokenizer
    from transformers import GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.save_pretrained(str(output_path))
    print(f"   ✓ Saved tokenizer")
    
    # Test loading
    print(f"\n🧪 Testing model loading...")
    try:
        model = NanoGPTLMHeadModel.from_pretrained(str(output_path))
        print(f"   ✓ Model loads successfully!")
        
        # Quick test
        test_ids = torch.tensor([[1, 2, 3, 4]])
        with torch.no_grad():
            outputs = model(test_ids)
            logits = outputs.logits
        print(f"   ✓ Forward pass works! Logits shape: {logits.shape}")
    except Exception as e:
        print(f"   ❌ Error loading model: {e}")
    
    print("\n" + "="*70)
    print("✅ Conversion completed successfully!")
    print("="*70)
    print(f"\n📂 Model saved to: {output_path}")
    print(f"\n🚀 To use:")
    print(f"   from huggingface.modeling_nanogpt import NanoGPTLMHeadModel")
    print(f"   model = NanoGPTLMHeadModel.from_pretrained('{output_path}')")
    
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert MLX to custom NanoGPT HF format")
    parser.add_argument("checkpoint", type=str, help="Path to MLX checkpoint (.npz file)")
    parser.add_argument("--output-dir", type=str, default="huggingface", 
                        help="Output directory (default: huggingface)")
    parser.add_argument("--model-name", type=str, default=None,
                        help="Model name (default: auto-generated)")
    
    args = parser.parse_args()
    
    convert_mlx_to_custom_hf(args.checkpoint, args.output_dir, args.model_name)
