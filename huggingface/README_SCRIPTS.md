# HuggingFace Model Publishing Scripts

Scripts to convert your MLX-trained nanoGPT model to HuggingFace format and publish to HuggingFace Hub.

## 📁 Files

| File | Purpose |
|------|---------|
| `publish_model.py` | **⭐ Main script** - Convert & upload in one command |
| `convert_to_hf.py` | Convert MLX `.npz` to HuggingFace format |
| `upload_to_hf.py` | Upload model to HuggingFace Hub |
| `test_model.py` | Test if converted model loads correctly |
| `README.md` | Model card template (will be published) |
| `GUIDE.md` | Detailed usage guide |
| `requirements.txt` | Python dependencies |

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install huggingface-hub safetensors
```

### 2. Authenticate with HuggingFace

```bash
huggingface-cli login
```

Get your token at: https://huggingface.co/settings/tokens

### 3. Publish Your Model

```bash
python huggingface/publish_model.py checkpoints/checkpoint_10000.npz \
  --repo-name your-username/your-model-name
```

That's it! Your model is now on HuggingFace! 🎉

## 📖 Usage Examples

### Example 1: Full Workflow (Convert + Upload)

```bash
python huggingface/publish_model.py checkpoints/checkpoint_20000.npz \
  --repo-name jacksu/nanogpt-20k \
  --model-name nanogpt-mlx-20k
```

### Example 2: Convert Only (No Upload)

```bash
python huggingface/publish_model.py checkpoints/checkpoint_10000.npz \
  --convert-only
```

This creates the HuggingFace files in the `huggingface/` directory without uploading.

### Example 3: Private Model

```bash
python huggingface/publish_model.py checkpoints/checkpoint_30000.npz \
  --repo-name jacksu/my-private-model \
  --private
```

### Example 4: Separate Steps

```bash
# Step 1: Convert
python huggingface/convert_to_hf.py checkpoints/checkpoint_10000.npz

# Step 2: Edit model card
vim huggingface/README.md

# Step 3: Test
python huggingface/test_model.py

# Step 4: Upload
python huggingface/upload_to_hf.py --repo-name jacksu/my-model
```

## 🔧 Individual Scripts

### Convert to HuggingFace Format

```bash
python huggingface/convert_to_hf.py <checkpoint.npz> \
  --output-dir huggingface \
  --model-name my-model-name
```

**Creates:**
- `config.json` - Model configuration
- `model.safetensors` - Model weights
- `generation_config.json` - Generation settings
- `training_metadata.json` - Training details
- `README.md` - Model card (from template)

### Test Converted Model

```bash
python huggingface/test_model.py --model-dir huggingface
```

Verifies:
- All required files present
- Model loads with transformers
- Generation works

### Upload to HuggingFace Hub

```bash
python huggingface/upload_to_hf.py \
  --model-dir huggingface \
  --repo-name username/model-name \
  [--private]
```

## 📝 Customizing Your Model Card

Before uploading, edit `huggingface/README.md` to:

1. **Replace placeholders:**
   - `YOUR_NAME` → Your name
   - `YOUR_USERNAME` → Your username
   - Performance metrics
   - Training details

2. **Add examples:**
   - Sample generations
   - Use cases
   - Limitations

3. **Update metadata:**
   - Training iterations
   - Final loss
   - Dataset information

## 🧪 Testing Your Model

After uploading, test it works:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("username/model-name")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

text = tokenizer.decode(
    model.generate(
        tokenizer("Once upon a time", return_tensors="pt").input_ids,
        max_length=100
    )[0]
)
print(text)
```

## 📦 What Gets Uploaded

Your HuggingFace repository will contain:

```
username/model-name/
├── config.json                  # Model architecture config
├── model.safetensors            # Model weights (recommended format)
├── generation_config.json       # Default generation parameters
├── training_metadata.json       # Training information
└── README.md                    # Model card
```

## 🔑 Authentication Options

### Method 1: CLI Login (Recommended)

```bash
huggingface-cli login
```

### Method 2: Environment Variable

```bash
export HF_TOKEN=your_token_here
python huggingface/upload_to_hf.py ...
```

### Method 3: Python Script

```python
from huggingface_hub import login
login(token="your_token_here")
```

## ⚙️ Command Line Options

### publish_model.py

```
--output-dir DIR      Output directory (default: huggingface)
--model-name NAME     Local model name (auto-generated if omitted)
--repo-name NAME      HuggingFace repo (username/model-name)
--private             Make repository private
--convert-only        Only convert, don't upload
--upload-only         Only upload (skip conversion)
--check-setup         Check HuggingFace authentication
```

### convert_to_hf.py

```
checkpoint            Path to .npz checkpoint file (required)
--output-dir DIR      Output directory (default: huggingface)
--model-name NAME     Model name (auto-generated if omitted)
```

### upload_to_hf.py

```
--model-dir DIR       Model directory (default: huggingface)
--repo-name NAME      Repository name (required)
--private             Make repository private
--commit-message MSG  Custom commit message
--check               Check setup only
```

## 🐛 Troubleshooting

### "Not authenticated with HuggingFace"

```bash
huggingface-cli login
```

### "safetensors not installed"

```bash
pip install safetensors
```

Model will be saved as `.npz` format as fallback.

### "Model won't load in transformers"

Install PyTorch:
```bash
pip install torch transformers
```

### "Repository already exists"

The script will update existing repo. Use `--private` if you want it private.

## 📚 Documentation

- **Detailed Guide**: See `GUIDE.md`
- **Model Card Template**: See `README.md`
- **HuggingFace Docs**: https://huggingface.co/docs/hub

## 🎯 Workflow Summary

```
Your MLX Model (.npz)
        ↓
[convert_to_hf.py] → HuggingFace files
        ↓
[test_model.py] → Verify conversion
        ↓
[upload_to_hf.py] → HuggingFace Hub
        ↓
Your Published Model! 🎉
```

## 💡 Tips

1. **Test locally first** with `test_model.py`
2. **Use SafeTensors format** (install `safetensors`)
3. **Write good model cards** (edit `README.md`)
4. **Include checkpoint iteration** in model name
5. **Make it private** while testing, public when ready
6. **Tag appropriately** in the README frontmatter

## 📞 Support

For issues or questions:
- Check `GUIDE.md` for detailed instructions
- Review error messages carefully
- Ensure authentication is setup
- Test conversion before upload

---

Made with ❤️ for the MLX community
