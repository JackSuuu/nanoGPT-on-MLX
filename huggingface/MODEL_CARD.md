---
language:
- en
license: mit
tags:
- text-generation
- pytorch
- gpt
- transformers
- pre-ln
- causal-lm
datasets:
- HuggingFaceFW/fineweb-edu
library_name: transformers
pipeline_tag: text-generation
metrics:
- perplexity
widget:
- text: "Once upon a time"
  example_title: "Story Beginning"
- text: "The capital of France is"
  example_title: "Factual Question"
- text: "In the field of machine learning,"
  example_title: "Technical Topic"
---

# NanoGPT 53M - Pre-LN Transformer

A 53-million parameter GPT model trained from scratch on FineWebEdu educational content. This model implements a **Pre-LayerNorm (Pre-LN) transformer architecture**, compatible with HuggingFace Transformers library.

> **Model Format:** PyTorch (cross-platform compatible)  
> **Training Framework:** Apple MLX (exported to PyTorch for universal compatibility)

## Model Details

### Architecture
- **Model Type:** GPT (Decoder-only Transformer)
- **Parameters:** 53M (52,990,464 total, 43M unique with weight tying)
- **Architecture Pattern:** Pre-LayerNorm (Pre-LN)
- **Layers:** 8 transformer blocks
- **Hidden Size:** 384
- **Attention Heads:** 8
- **Feedforward Dimension:** 1536
- **Context Length:** 512 tokens
- **Vocabulary Size:** 50,257 (GPT-2 tokenizer)

### Training
- **Framework:** Apple MLX (training), PyTorch (export)
- **Dataset:** FineWebEdu - 10M tokens of educational web content
- **Training Hardware:** Apple M2 Pro (16GB unified memory)
- **Checkpoint:** 35000 iterations
- **Training Method:** Base pretraining (20K iters) + Knowledge Distillation (15K iters)
- **Teacher Model:** GPT-OSS-20B (via Groq API)

### Architecture Highlights

This model uses **Pre-LayerNorm** architecture, different from standard GPT-2's Post-LN:

```python
# Pre-LN (this model)
x = x + attn(ln(x))
x = x + ff(ln(x))

# vs Post-LN (standard GPT-2)
x = ln(x + attn(x))
x = ln(x + ff(x))
```

Pre-LN provides better training stability and is used in modern transformers (GPT-3, PaLM, LLaMA).

## Training Details

- **Dataset:** FineWebEdu (diverse educational web content)
- **Training Tokens:** 10M
- **Base Training:** 20,000 iterations (loss 0.758)
- **Knowledge Distillation:** 15,000 additional iterations with GPT-OSS-20B as teacher
- **Total Iterations:** 35,000
- **Batch Size:** 12
- **Learning Rate:** 3e-4 with cosine decay (base), 3e-5 (distillation)
- **Final Training Loss:** 3.46
- **Distillation Method:** 50% hard loss (ground truth) + 50% soft loss (teacher)

### Performance Benchmarks

Measured on Apple M2 Pro (16GB unified memory):

| Metric | Value |
|--------|-------|
| **Model Size** | 53.0M parameters |
| **Memory (fp32)** | 202.1 MB |
| **Memory (fp16)** | 101.1 MB |
| **Training Throughput** | 27,355 tokens/sec |
| **Batch Processing** | 13.36 batches/sec (batch=4, seq=512) |
| **Inference Speed** | 169.9 tokens/sec |
| **Generation Latency** | ~0.59s per 100 tokens |
| **Activation Memory** | 843 MB (batch=4, seq=512) |

> **Note:** Benchmarks measured at checkpoint 20000. This release (checkpoint 35000) includes additional knowledge distillation training.

## Usage

### Basic Text Generation

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model and tokenizer (requires trust_remote_code for custom architecture)
tokenizer = AutoTokenizer.from_pretrained("jacksuuuu/nanogpt-mlx-53m-finewebedu")
model = AutoModelForCausalLM.from_pretrained(
    "jacksuuuu/nanogpt-mlx-53m-finewebedu",
    trust_remote_code=True
)

# Generate text
prompt = "Once upon a time"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(
    **inputs,
    max_length=100,
    temperature=0.8,
    top_k=50,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id
)

text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(text)
```

### Example Output

**Prompt:** "Once upon a time"

**Generated (Checkpoint 35000 with distillation):**
```
Once upon a time: "the)." as in KDE, set by an article of the U and 
updated to the existing of a network. For requirements of the application 
to an individual to the data above above above above...
```

**Note:** This checkpoint shows characteristics of knowledge distillation training. The model has learned broader patterns from the teacher model (GPT-OSS-20B), though generation quality varies. For more coherent story generation, consider fine-tuning on your specific use case.

## Model Architecture

```python
NanoGPTLMHeadModel(
  (transformer): NanoGPTModel(
    (token_embedding): Embedding(50257, 384)
    (position_embedding): Embedding(512, 384)
    (blocks): ModuleList(
      (0-7): 8 x NanoGPTBlock(
        (ln1): LayerNorm((384,), eps=1e-05)
        (attn): NanoGPTAttention(
          (qkv_proj): Linear(384, 1152)
          (out_proj): Linear(384, 384)
        )
        (ln2): LayerNorm((384,), eps=1e-05)
        (ff): FeedForward(
          (fc1): Linear(384, 1536)
          (fc2): Linear(1536, 384)
        )
      )
    )
    (ln_f): LayerNorm((384,), eps=1e-05)
  )
  (lm_head): Linear(384, 50257)
)
```

**Note:** `token_embedding` and `lm_head` weights are tied (shared), reducing effective parameters from 53M to 43M unique weights.

## Training Configuration

```python
{
  "vocab_size": 50257,
  "d_model": 384,
  "n_layers": 8,
  "n_heads": 8,
  "d_ff": 1536,
  "context_length": 512,
  "dropout": 0.1,
  "batch_size": 12,
  "learning_rate": 3e-4,
  "weight_decay": 0.1,
  "max_iters": 20000
}
```

## Limitations

- **Context length:** Limited to 512 tokens
- **Domain:** Trained on educational web content (FineWebEdu)
- **Size:** 53M parameters is relatively small compared to modern LLMs
- **Generation:** Best for short-form content (stories, paragraphs)
- **No instruction tuning:** This is a base language model, not instruction-tuned

## Intended Use

**Primary use cases:**
- Educational demonstrations of transformer training
- Resource-constrained inference on Apple Silicon
- Base model for fine-tuning on specific domains
- Research and experimentation with Pre-LN architectures

**Not recommended for:**
- Production applications requiring factual accuracy
- Long-form content generation (>512 tokens)
- Instruction following or chat applications (not instruction-tuned)

## Ethical Considerations

This model was trained on FineWebEdu, which contains diverse web content. Users should:
- Be aware of potential biases in generated content
- Validate outputs for factual accuracy
- Not use for applications requiring high reliability
- Consider fine-tuning on domain-specific data for production use

## Citation

If you use this model, please cite:

```bibtex
@software{nanogpt_mlx_2025,
  author = {JackSu},
  title = {NanoGPT MLX: 53M Parameter Pre-LN Transformer},
  year = {2025},
  url = {https://huggingface.co/jacksuuuu/nanogpt-mlx-53m-finewebedu}
}
```

## Additional Resources

- **GitHub Repository:** [JackSuuu/nanoGPT-on-MLX](https://github.com/JackSuuu/nanoGPT-on-MLX)
- **MLX Framework:** [ml-explore/mlx](https://github.com/ml-explore/mlx)
- **Training Dataset:** [HuggingFaceFW/fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)

## License

MIT License - See repository for details.
