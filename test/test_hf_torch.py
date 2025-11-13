"""
Quick test to verify the model loads from HuggingFace
"""
from transformers import AutoTokenizer, AutoModelForCausalLM

print("🔄 Loading model from HuggingFace...")
print("   Repository: jacksuuuu/nanogpt-mlx-53m-finewebedu")

try:
    # Load tokenizer
    print("\n📥 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("jacksuuuu/nanogpt-mlx-53m-finewebedu")
    print("   ✓ Tokenizer loaded successfully")
    
    # Load model
    print("\n📥 Loading model...")
    model = AutoModelForCausalLM.from_pretrained("jacksuuuu/nanogpt-mlx-53m-finewebedu")
    print("   ✓ Model loaded successfully")
    
    # Check model size
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\n📊 Model parameters: {num_params:,} ({num_params/1e6:.1f}M)")
    
    # Generate text
    print("\n🎨 Generating text...")
    prompt = "Once upon a time in a small village"
    inputs = tokenizer(prompt, return_tensors="pt")
    
    outputs = model.generate(
        **inputs,
        max_length=100,
        temperature=0.8,
        do_sample=True,
        top_k=50,
        top_p=0.95,
    )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print(f"\n📝 Prompt: {prompt}")
    print(f"📝 Generated:\n{generated_text}")
    
    print("\n" + "=" * 70)
    print("✅ SUCCESS! Model works perfectly on HuggingFace!")
    print("=" * 70)
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    print("\n💡 Make sure you have transformers and torch installed:")
    print("   pip install transformers torch")
