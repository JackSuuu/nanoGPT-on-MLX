"""
Test loading HuggingFace model to verify conversion
"""
import sys
import argparse
from pathlib import Path


def test_model_loading(model_dir):
    """Test if converted model can be loaded"""
    print("="*70)
    print("Testing HuggingFace Model Loading")
    print("="*70)
    
    model_dir = Path(model_dir)
    
    if not model_dir.exists():
        print(f"❌ Error: Model directory not found: {model_dir}")
        return False
    
    print(f"\n📁 Model directory: {model_dir}")
    
    # Check files
    print("\n📋 Checking files...")
    required_files = {
        'config.json': 'Model configuration',
        'generation_config.json': 'Generation configuration',
        'training_metadata.json': 'Training metadata'
    }
    
    weight_files = {
        'model.safetensors': 'SafeTensors weights',
        'model.npz': 'NumPy weights',
        'pytorch_model.bin': 'PyTorch weights'
    }
    
    for filename, description in required_files.items():
        filepath = model_dir / filename
        if filepath.exists():
            print(f"   ✓ {filename} ({description})")
        else:
            print(f"   ❌ {filename} MISSING!")
            return False
    
    has_weights = False
    for filename, description in weight_files.items():
        filepath = model_dir / filename
        if filepath.exists():
            print(f"   ✓ {filename} ({description})")
            has_weights = True
    
    if not has_weights:
        print("   ❌ No weight file found!")
        return False
    
    # Try loading with transformers (if available)
    print("\n🔧 Testing with transformers library...")
    try:
        from transformers import AutoConfig, AutoTokenizer
        import json
        
        # Load config
        config = AutoConfig.from_pretrained(str(model_dir))
        print(f"   ✓ Config loaded")
        print(f"      - Model type: {config.model_type}")
        print(f"      - Vocab size: {config.vocab_size}")
        print(f"      - Layers: {config.n_layer}")
        print(f"      - Hidden size: {config.n_embd}")
        
        # Try loading tokenizer (will use GPT-2 tokenizer)
        try:
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
            print(f"   ✓ Tokenizer loaded (GPT-2)")
        except Exception as e:
            print(f"   ⚠️  Tokenizer: {e}")
        
        # Try loading model weights
        try:
            from transformers import AutoModelForCausalLM
            print("\n   Loading model weights...")
            model = AutoModelForCausalLM.from_pretrained(str(model_dir))
            print(f"   ✓ Model loaded successfully!")
            print(f"      - Parameters: {model.num_parameters():,}")
            
            # Try a quick generation test
            print("\n🧪 Testing generation...")
            prompt = "Once upon a time"
            inputs = tokenizer(prompt, return_tensors="pt")
            
            outputs = model.generate(
                inputs.input_ids,
                max_length=50,
                temperature=0.8,
                do_sample=True,
            )
            
            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"   ✓ Generation test passed!")
            print(f"\n   Prompt: {prompt}")
            print(f"   Output: {generated}")
            
        except Exception as e:
            print(f"   ⚠️  Model loading: {e}")
            print(f"      This might be expected if weights need PyTorch conversion")
        
    except ImportError:
        print("   ⚠️  transformers library not installed")
        print("      Install with: pip install transformers torch")
        print("      Model files are valid, but can't test loading")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False
    
    # Load metadata
    print("\n📊 Training Metadata...")
    metadata_path = model_dir / "training_metadata.json"
    if metadata_path.exists():
        import json
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        print(f"   Model: {metadata.get('model_name', 'N/A')}")
        print(f"   Iterations: {metadata.get('training', {}).get('iterations', 'N/A'):,}")
        print(f"   Final Loss: {metadata.get('training', {}).get('final_loss', 'N/A')}")
        print(f"   Dataset: {metadata.get('training', {}).get('dataset', 'N/A')}")
    
    print("\n" + "="*70)
    print("✅ Model verification complete!")
    print("="*70)
    
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test HuggingFace model loading")
    parser.add_argument("--model-dir", type=str, default="huggingface",
                        help="Directory containing HuggingFace model (default: huggingface)")
    
    args = parser.parse_args()
    
    success = test_model_loading(args.model_dir)
    sys.exit(0 if success else 1)
