"""
Add GPT-2 tokenizer files to HuggingFace model repository
This fixes the "tokenizer.json not found" error when loading from HuggingFace
"""
import argparse
from huggingface_hub import HfApi, login
from transformers import AutoTokenizer
import tempfile
import shutil
from pathlib import Path


def add_tokenizer_to_repo(repo_name, model_dir="huggingface"):
    """
    Download GPT-2 tokenizer and upload to your model repo
    
    Args:
        repo_name: HuggingFace repo name (username/model-name)
        model_dir: Local model directory
    """
    print("=" * 70)
    print("Add GPT-2 Tokenizer to HuggingFace Model")
    print("=" * 70)
    
    print(f"\n📦 Repository: {repo_name}")
    
    # Authenticate
    print("\n🔐 Authenticating with HuggingFace...")
    try:
        api = HfApi()
        user_info = api.whoami()
        print(f"   ✓ Authenticated as: {user_info['name']}")
    except Exception as e:
        print(f"   ❌ Authentication failed: {e}")
        print("\n💡 Run: huggingface-cli login")
        return False
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Download GPT-2 tokenizer
        print("\n📥 Downloading GPT-2 tokenizer files...")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        
        # Save to temp directory
        print("💾 Saving tokenizer files...")
        tokenizer.save_pretrained(temp_path)
        
        # List files to upload
        tokenizer_files = [
            "tokenizer.json",
            "vocab.json", 
            "merges.txt",
            "tokenizer_config.json"
        ]
        
        print(f"\n📤 Uploading tokenizer files to {repo_name}...")
        for filename in tokenizer_files:
            file_path = temp_path / filename
            if file_path.exists():
                try:
                    api.upload_file(
                        path_or_fileobj=str(file_path),
                        path_in_repo=filename,
                        repo_id=repo_name,
                        repo_type="model",
                    )
                    print(f"   ✓ Uploaded: {filename}")
                except Exception as e:
                    print(f"   ⚠️  Failed to upload {filename}: {e}")
            else:
                print(f"   ⚠️  File not found: {filename}")
        
        # Also save to local model directory
        if model_dir:
            model_path = Path(model_dir)
            if model_path.exists():
                print(f"\n💾 Saving tokenizer to local directory: {model_dir}")
                tokenizer.save_pretrained(model_path)
                print("   ✓ Tokenizer files saved locally")
    
    print("\n" + "=" * 70)
    print("✅ Tokenizer added successfully!")
    print("=" * 70)
    
    print(f"\n🧪 Test loading your model:")
    print(f"""
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("{repo_name}")
model = AutoModelForCausalLM.from_pretrained("{repo_name}")

# Generate text
inputs = tokenizer("Once upon a time", return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
text = tokenizer.decode(outputs[0])
print(text)
""")
    
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add GPT-2 tokenizer to HuggingFace model")
    parser.add_argument("--repo-name", type=str, required=True,
                       help="HuggingFace repo name (username/model-name)")
    parser.add_argument("--model-dir", type=str, default="huggingface",
                       help="Local model directory (default: huggingface)")
    
    args = parser.parse_args()
    
    success = add_tokenizer_to_repo(args.repo_name, args.model_dir)
    
    if not success:
        exit(1)
