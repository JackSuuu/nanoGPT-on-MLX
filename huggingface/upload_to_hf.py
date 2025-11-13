"""
Upload HuggingFace model to HuggingFace Hub
Requires: huggingface_hub library and authentication
"""
import os
import json
import argparse
from pathlib import Path


def upload_to_huggingface(model_dir, repo_name, private=False, commit_message=None):
    """
    Upload model to HuggingFace Hub
    
    Args:
        model_dir: Directory containing HuggingFace model files
        repo_name: Repository name (username/model-name)
        private: Whether to make the model private
        commit_message: Custom commit message
    """
    try:
        from huggingface_hub import HfApi, create_repo, login
    except ImportError:
        print("❌ Error: huggingface_hub not installed")
        print("\n📦 Install with: pip install huggingface_hub")
        return False
    
    print("="*70)
    print("HuggingFace Model Upload")
    print("="*70)
    
    model_dir = Path(model_dir)
    
    # Check if model directory exists
    if not model_dir.exists():
        print(f"❌ Error: Model directory not found: {model_dir}")
        return False
    
    # Check required files
    required_files = ['config.json']
    model_files = ['model.safetensors', 'model.npz', 'pytorch_model.bin']
    
    has_weights = False
    for f in required_files:
        if not (model_dir / f).exists():
            print(f"❌ Error: Required file missing: {f}")
            return False
    
    for f in model_files:
        if (model_dir / f).exists():
            has_weights = True
            break
    
    if not has_weights:
        print("❌ Error: No model weights file found (model.safetensors, model.npz, or pytorch_model.bin)")
        return False
    
    print(f"\n📁 Model directory: {model_dir}")
    print(f"📦 Repository: {repo_name}")
    print(f"🔒 Private: {private}")
    
    # Authenticate
    print("\n🔐 Authenticating with HuggingFace...")
    print("   Note: You'll need a HuggingFace token with write access")
    print("   Get one at: https://huggingface.co/settings/tokens")
    
    try:
        # Try to login (will use cached token if available)
        api = HfApi()
        whoami = api.whoami()
        username = whoami['name']
        print(f"   ✓ Authenticated as: {username}")
    except Exception as e:
        print(f"\n❌ Authentication failed: {e}")
        print("\n🔑 Please login:")
        print("   1. Get your token from: https://huggingface.co/settings/tokens")
        print("   2. Run: huggingface-cli login")
        print("   3. Or set HF_TOKEN environment variable")
        return False
    
    # Validate repo_name format
    if '/' not in repo_name:
        repo_name = f"{username}/{repo_name}"
        print(f"\n📝 Using full repo name: {repo_name}")
    
    # Create repository
    print(f"\n🏗️  Creating repository...")
    try:
        repo_url = create_repo(
            repo_id=repo_name,
            repo_type="model",
            private=private,
            exist_ok=True  # Don't error if repo already exists
        )
        print(f"   ✓ Repository ready: {repo_url}")
    except Exception as e:
        print(f"   ⚠️  Note: {e}")
        print(f"   Continuing with upload...")
    
    # Prepare commit message
    if commit_message is None:
        # Load metadata for auto-generated message
        metadata_path = model_dir / "training_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            iterations = metadata.get('training', {}).get('iterations', 'unknown')
            loss = metadata.get('training', {}).get('final_loss', 'unknown')
            commit_message = f"Upload model - {iterations} iterations, loss: {loss:.4f}"
        else:
            commit_message = "Upload model checkpoint"
    
    # Upload files
    print(f"\n📤 Uploading files...")
    try:
        from huggingface_hub import upload_folder
        
        api.upload_folder(
            folder_path=str(model_dir),
            repo_id=repo_name,
            repo_type="model",
            commit_message=commit_message,
        )
        
        print(f"   ✓ All files uploaded successfully!")
        
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        return False
    
    # Success!
    repo_url = f"https://huggingface.co/{repo_name}"
    print("\n" + "="*70)
    print("✅ Upload completed successfully!")
    print("="*70)
    print(f"\n🌐 Model URL: {repo_url}")
    print(f"\n📝 Next steps:")
    print(f"   1. Visit {repo_url} to view your model")
    print(f"   2. Update the model card (README.md) if needed")
    print(f"   3. Test loading: ")
    print(f"      from transformers import AutoModelForCausalLM")
    print(f"      model = AutoModelForCausalLM.from_pretrained('{repo_name}')")
    
    return True


def check_setup():
    """Check if all requirements are installed"""
    print("Checking setup...")
    
    try:
        import huggingface_hub
        print("✓ huggingface_hub installed")
    except ImportError:
        print("❌ huggingface_hub not installed")
        print("   Install: pip install huggingface_hub")
        return False
    
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        whoami = api.whoami()
        print(f"✓ Authenticated as: {whoami['name']}")
    except Exception:
        print("❌ Not authenticated with HuggingFace")
        print("   Login: huggingface-cli login")
        return False
    
    print("\n✅ Setup complete!")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload model to HuggingFace Hub")
    parser.add_argument("--model-dir", type=str, default="huggingface",
                        help="Directory containing HuggingFace model files")
    parser.add_argument("--repo-name", type=str, 
                        help="Repository name (username/model-name or just model-name)")
    parser.add_argument("--private", action="store_true",
                        help="Make repository private")
    parser.add_argument("--commit-message", type=str, default=None,
                        help="Custom commit message")
    parser.add_argument("--check", action="store_true",
                        help="Just check setup and authentication")
    
    args = parser.parse_args()
    
    if args.check:
        check_setup()
    else:
        if not args.repo_name:
            print("❌ Error: --repo-name is required")
            print("Example: --repo-name my-username/my-model-name")
            exit(1)
        
        success = upload_to_huggingface(
            args.model_dir,
            args.repo_name,
            args.private,
            args.commit_message
        )
        
        exit(0 if success else 1)
