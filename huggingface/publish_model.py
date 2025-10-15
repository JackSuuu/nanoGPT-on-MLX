"""
Unified workflow: Convert MLX model to HuggingFace and upload
One-stop script for the entire process
"""
import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from huggingface.convert_to_hf import convert_mlx_to_hf
from huggingface.upload_to_hf import upload_to_huggingface, check_setup


def main():
    parser = argparse.ArgumentParser(
        description="Convert MLX model and upload to HuggingFace Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert only
  python huggingface/publish_model.py checkpoints/checkpoint_10000.npz --convert-only
  
  # Convert and upload
  python huggingface/publish_model.py checkpoints/checkpoint_10000.npz \\
    --repo-name username/my-model
  
  # Full workflow with custom name
  python huggingface/publish_model.py checkpoints/checkpoint_20000.npz \\
    --repo-name username/nanogpt-20k \\
    --model-name nanogpt-mlx-20k \\
    --private
        """
    )
    
    parser.add_argument("checkpoint", type=str,
                        help="Path to MLX checkpoint (.npz file)")
    parser.add_argument("--output-dir", type=str, default="huggingface",
                        help="Output directory for HuggingFace files (default: huggingface)")
    parser.add_argument("--model-name", type=str, default=None,
                        help="Model name for local files (default: auto-generated)")
    parser.add_argument("--repo-name", type=str, default=None,
                        help="HuggingFace repo name (username/model-name)")
    parser.add_argument("--private", action="store_true",
                        help="Make HuggingFace repository private")
    parser.add_argument("--convert-only", action="store_true",
                        help="Only convert, don't upload")
    parser.add_argument("--upload-only", action="store_true",
                        help="Only upload (assumes already converted)")
    parser.add_argument("--check-setup", action="store_true",
                        help="Check if HuggingFace authentication is setup")
    
    args = parser.parse_args()
    
    # Check setup if requested
    if args.check_setup:
        check_setup()
        return
    
    # Validate arguments
    if not args.convert_only and not args.upload_only and not args.repo_name:
        print("❌ Error: --repo-name is required for upload")
        print("   Use --convert-only to skip upload")
        print("   Example: --repo-name username/my-model")
        sys.exit(1)
    
    # Step 1: Convert (unless upload-only)
    if not args.upload_only:
        print("\n" + "🔄 STEP 1: Converting MLX model to HuggingFace format")
        print("="*70)
        
        try:
            output_path = convert_mlx_to_hf(
                args.checkpoint,
                args.output_dir,
                args.model_name
            )
            print(f"\n✅ Conversion successful!")
        except Exception as e:
            print(f"\n❌ Conversion failed: {e}")
            sys.exit(1)
    else:
        output_path = Path(args.output_dir)
        if not output_path.exists():
            print(f"❌ Error: Output directory not found: {output_path}")
            sys.exit(1)
    
    # Step 2: Upload (unless convert-only)
    if not args.convert_only:
        print("\n\n" + "📤 STEP 2: Uploading to HuggingFace Hub")
        print("="*70)
        
        try:
            success = upload_to_huggingface(
                str(output_path),
                args.repo_name,
                args.private
            )
            
            if success:
                print(f"\n\n{'='*70}")
                print("🎉 SUCCESS! Model published to HuggingFace!")
                print("="*70)
                print(f"\n🌐 View your model: https://huggingface.co/{args.repo_name}")
            else:
                print("\n❌ Upload failed")
                sys.exit(1)
                
        except Exception as e:
            print(f"\n❌ Upload failed: {e}")
            sys.exit(1)
    
    # Done!
    print("\n" + "="*70)
    print("✅ All done!")
    print("="*70)
    
    if args.convert_only:
        print(f"\n📁 Converted model saved to: {output_path}")
        print(f"\n📝 Next steps:")
        print(f"   1. Review the model files in {output_path}")
        print(f"   2. Upload with: python huggingface/upload_to_hf.py --repo-name username/model-name")
    else:
        print(f"\n🎉 Your model is now live on HuggingFace!")
        print(f"\n📝 Next steps:")
        print(f"   1. Visit https://huggingface.co/{args.repo_name}")
        print(f"   2. Customize the model card (README.md)")
        print(f"   3. Test loading:")
        print(f"      from transformers import AutoModelForCausalLM")
        print(f"      model = AutoModelForCausalLM.from_pretrained('{args.repo_name}')")


if __name__ == "__main__":
    main()
