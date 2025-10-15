"""
Convert MLX model (.npz) to HuggingFace format
This script converts your trained nanoGPT model to HuggingFace GPT-2 compatible format
"""
import os
import json
import argparse
import numpy as np
import mlx.core as mx
from pathlib import Path
from src.model import create_model
from src.utils import load_checkpoint


def convert_mlx_to_hf(checkpoint_path, output_dir="huggingface", model_name=None):
    """
    Convert MLX checkpoint to HuggingFace format
    
    Args:
        checkpoint_path: Path to .npz checkpoint file
        output_dir: Output directory for HuggingFace model
        model_name: Optional model name (defaults to checkpoint name)
    """
    print("="*70)
    print("MLX to HuggingFace Model Converter")
    print("="*70)
    
    # Load checkpoint metadata
    checkpoint_path = Path(checkpoint_path)
    meta_path = checkpoint_path.parent / f"{checkpoint_path.stem}_meta.json"
    
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")
    
    with open(meta_path, 'r') as f:
        metadata = json.load(f)
    
    config = metadata['config']
    iteration = metadata['iteration']
    loss = metadata['loss']
    
    print(f"\n📦 Loading checkpoint: {checkpoint_path.name}")
    print(f"   Iteration: {iteration:,}")
    print(f"   Loss: {loss:.4f}")
    print(f"   Model: {config['d_model']}d, {config['n_layers']} layers, {config['n_heads']} heads")
    
    # Create MLX model
    print("\n🔨 Creating MLX model...")
    model = create_model(config)
    
    # Load weights
    print("📥 Loading weights...")
    model.load_weights(str(checkpoint_path))
    mx.eval(model.parameters())
    
    # Get model parameters
    params = model.parameters()
    
    # Create output directory
    if model_name is None:
        model_name = f"nanogpt-mlx-{config['d_model']}d-{iteration//1000}k"
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📁 Output directory: {output_path}")
    
    # Convert to HuggingFace config format
    hf_config = {
        "architectures": ["GPT2LMHeadModel"],
        "model_type": "gpt2",
        "vocab_size": config['vocab_size'],
        "n_positions": config['context_length'],
        "n_embd": config['d_model'],
        "n_layer": config['n_layers'],
        "n_head": config['n_heads'],
        "n_inner": config['d_ff'],
        "activation_function": "gelu_new",
        "resid_pdrop": config['dropout'],
        "embd_pdrop": config['dropout'],
        "attn_pdrop": config['dropout'],
        "layer_norm_epsilon": 1e-5,
        "initializer_range": 0.02,
        "bos_token_id": 50256,
        "eos_token_id": 50256,
        "tie_word_embeddings": True,
        "torch_dtype": "float32",
        "transformers_version": "4.35.0",
        # Custom metadata
        "mlx_training": {
            "framework": "MLX",
            "iterations": iteration,
            "final_loss": loss,
            "dataset": config.get('dataset_name', 'tinystories'),
            "max_tokens": config.get('max_tokens', 2_000_000),
        }
    }
    
    # Save config.json
    config_path = output_path / "config.json"
    print(f"\n💾 Saving config.json...")
    with open(config_path, 'w') as f:
        json.dump(hf_config, f, indent=2)
    print(f"   ✓ {config_path}")
    
    # Convert weights to HuggingFace format
    print(f"\n🔄 Converting weights to HuggingFace format...")
    hf_weights = convert_weights_mlx_to_hf(params, config)
    
    # Save as safetensors (recommended) or pytorch_model.bin
    try:
        from safetensors.numpy import save_file
        weights_path = output_path / "model.safetensors"
        save_file(hf_weights, weights_path)
        print(f"   ✓ Saved as SafeTensors: {weights_path}")
    except ImportError:
        print("   ⚠ safetensors not installed, saving as numpy format")
        weights_path = output_path / "model.npz"
        np.savez(weights_path, **hf_weights)
        print(f"   ✓ Saved as NPZ: {weights_path}")
    
    # Calculate total parameters
    def count_params(params_dict):
        """Recursively count parameters in nested dict"""
        total = 0
        for v in params_dict.values():
            if isinstance(v, dict):
                total += count_params(v)
            elif hasattr(v, 'size'):
                total += v.size
        return total
    
    total_params = count_params(params)
    
    # Save training metadata
    metadata_path = output_path / "training_metadata.json"
    training_metadata = {
        "model_name": model_name,
        "architecture": "GPT-2",
        "parameters": f"{total_params:,}",
        "training": {
            "iterations": iteration,
            "final_loss": loss,
            "dataset": config.get('dataset_name', 'tinystories'),
            "tokens_trained": config.get('max_tokens', 2_000_000),
            "batch_size": config['batch_size'],
            "learning_rate": config['learning_rate'],
            "context_length": config['context_length'],
        },
        "model_config": {
            "d_model": config['d_model'],
            "n_layers": config['n_layers'],
            "n_heads": config['n_heads'],
            "d_ff": config['d_ff'],
            "vocab_size": config['vocab_size'],
        }
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(training_metadata, f, indent=2)
    print(f"   ✓ Training metadata: {metadata_path}")
    
    # Create generation config
    generation_config = {
        "bos_token_id": 50256,
        "eos_token_id": 50256,
        "max_length": config['context_length'],
        "temperature": 1.0,
        "top_k": 50,
        "top_p": 0.95,
        "do_sample": True,
    }
    
    gen_config_path = output_path / "generation_config.json"
    with open(gen_config_path, 'w') as f:
        json.dump(generation_config, f, indent=2)
    print(f"   ✓ Generation config: {gen_config_path}")
    
    print("\n" + "="*70)
    print("✅ Conversion completed successfully!")
    print("="*70)
    print(f"\n📂 HuggingFace model saved to: {output_path}")
    print(f"\n🚀 Next steps:")
    print(f"   1. Review README.md in {output_path}")
    print(f"   2. Test loading: python huggingface/test_model.py")
    print(f"   3. Upload: python huggingface/upload_to_hf.py --model-dir {output_path}")
    
    return output_path


def convert_weights_mlx_to_hf(mlx_params, config):
    """
    Convert MLX parameter names to HuggingFace GPT-2 format
    
    MLX structure:
        embedding.weight
        layers[i].attention.qkv_proj.weight/bias
        layers[i].attention.out_proj.weight/bias
        layers[i].ln1.weight/bias
        layers[i].ffn.fc1.weight/bias
        layers[i].ffn.fc2.weight/bias
        layers[i].ln2.weight/bias
        ln_f.weight/bias
        lm_head.weight (tied with embedding)
    
    HF GPT-2 structure:
        transformer.wte.weight (word embeddings)
        transformer.wpe.weight (position embeddings)
        transformer.h.{i}.ln_1.weight/bias
        transformer.h.{i}.attn.c_attn.weight/bias (combined QKV)
        transformer.h.{i}.attn.c_proj.weight/bias
        transformer.h.{i}.ln_2.weight/bias
        transformer.h.{i}.mlp.c_fc.weight/bias
        transformer.h.{i}.mlp.c_proj.weight/bias
        transformer.ln_f.weight/bias
        lm_head.weight
    """
    hf_weights = {}
    
    # Convert MLX arrays to numpy
    def to_numpy(x):
        return np.array(x)
    
    # Word embeddings
    if 'embedding' in mlx_params and 'weight' in mlx_params['embedding']:
        hf_weights['transformer.wte.weight'] = to_numpy(mlx_params['embedding']['weight'])
    
    # Create position embeddings (initialize with small random values)
    n_positions = config['context_length']
    d_model = config['d_model']
    hf_weights['transformer.wpe.weight'] = np.random.randn(n_positions, d_model).astype(np.float32) * 0.02
    
    # Convert each transformer layer
    if 'layers' in mlx_params:
        for i, layer in enumerate(mlx_params['layers']):
            prefix = f'transformer.h.{i}'
            
            # Layer norm 1
            if 'ln1' in layer:
                hf_weights[f'{prefix}.ln_1.weight'] = to_numpy(layer['ln1']['weight'])
                hf_weights[f'{prefix}.ln_1.bias'] = to_numpy(layer['ln1']['bias'])
            
            # Attention
            if 'attention' in layer:
                attn = layer['attention']
                
                # Combined QKV projection -> c_attn
                if 'qkv_proj' in attn:
                    hf_weights[f'{prefix}.attn.c_attn.weight'] = to_numpy(attn['qkv_proj']['weight'])
                    hf_weights[f'{prefix}.attn.c_attn.bias'] = to_numpy(attn['qkv_proj']['bias'])
                
                # Output projection -> c_proj
                if 'out_proj' in attn:
                    hf_weights[f'{prefix}.attn.c_proj.weight'] = to_numpy(attn['out_proj']['weight'])
                    hf_weights[f'{prefix}.attn.c_proj.bias'] = to_numpy(attn['out_proj']['bias'])
            
            # Layer norm 2
            if 'ln2' in layer:
                hf_weights[f'{prefix}.ln_2.weight'] = to_numpy(layer['ln2']['weight'])
                hf_weights[f'{prefix}.ln_2.bias'] = to_numpy(layer['ln2']['bias'])
            
            # MLP/FFN
            if 'ffn' in layer:
                ffn = layer['ffn']
                
                # fc1 -> c_fc
                if 'fc1' in ffn:
                    hf_weights[f'{prefix}.mlp.c_fc.weight'] = to_numpy(ffn['fc1']['weight'])
                    hf_weights[f'{prefix}.mlp.c_fc.bias'] = to_numpy(ffn['fc1']['bias'])
                
                # fc2 -> c_proj
                if 'fc2' in ffn:
                    hf_weights[f'{prefix}.mlp.c_proj.weight'] = to_numpy(ffn['fc2']['weight'])
                    hf_weights[f'{prefix}.mlp.c_proj.bias'] = to_numpy(ffn['fc2']['bias'])
    
    # Final layer norm
    if 'ln_f' in mlx_params:
        hf_weights['transformer.ln_f.weight'] = to_numpy(mlx_params['ln_f']['weight'])
        hf_weights['transformer.ln_f.bias'] = to_numpy(mlx_params['ln_f']['bias'])
    
    # LM head (tied with embeddings in GPT-2)
    # HuggingFace will automatically tie these if tie_word_embeddings=True
    if 'lm_head' in mlx_params and 'weight' in mlx_params['lm_head']:
        hf_weights['lm_head.weight'] = to_numpy(mlx_params['lm_head']['weight'])
    
    print(f"   ✓ Converted {len(hf_weights)} weight tensors")
    
    return hf_weights


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert MLX model to HuggingFace format")
    parser.add_argument("checkpoint", type=str, help="Path to MLX checkpoint (.npz file)")
    parser.add_argument("--output-dir", type=str, default="huggingface", 
                        help="Output directory (default: huggingface)")
    parser.add_argument("--model-name", type=str, default=None,
                        help="Model name (default: auto-generated)")
    
    args = parser.parse_args()
    
    convert_mlx_to_hf(args.checkpoint, args.output_dir, args.model_name)
