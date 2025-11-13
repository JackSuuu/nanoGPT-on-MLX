---
language:
- en
license: mit
tags:
- text-generation
- mlx
- gpt
- pre-ln
datasets:
- HuggingFaceFW/fineweb-edu
metrics:
- perplexity
model-index:
- name: nanogpt-mlx-53m-finewebedu
  results:
  - task:
      type: text-generation
      name: Text Generation
    dataset:
      name: FineWebEdu
      type: HuggingFaceFW/fineweb-edu
    metrics:
    - type: perplexity
      value: 690728
      name: Validation Perplexity
    - type: loss
      value: 0.758
      name: Training Loss
---

# NanoGPT MLX 53M (FineWebEdu)

A 53-million parameter GPT model trained on FineWebEdu using Apple's MLX framework. This model features a **Pre-LayerNorm (Pre-LN) transformer architecture** optimized for Apple Silicon.

## Model Details

- **Parameters:** 53M (52,990,464 total)
- **Architecture:** Pre-LN Transformer (8 layers, 384d model, 8 attention heads)
- **Context Length:** 512 tokens
- **Vocabulary:** 50,257 tokens (GPT-2 tokenizer)
- **Training Data:** FineWebEdu (10M tokens, educational web content)
- **Training Framework:** MLX (Apple Silicon optimized)
- **Hardware:** M2 Pro with 16GB memory
- **Checkpoint:** 35000 (includes knowledge distillation from GPT-OSS-20B)

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

Training and inference on M2 Pro (measured at checkpoint 20000):

```
📊 Model Size:      53.0M parameters
                   202.1 MB (fp32), 101.1 MB (fp16)

⚡ Training:        27,355 tokens/sec (forward pass)
                   13.36 batches/sec (batch=4, seq=512)

🎯 Inference:       169.9 tokens/sec
                   ~0.59s per 100 tokens

💾 Memory:          843 MB activations (batch=4, seq=512)
```

**Note:** This checkpoint (35000) includes additional training with knowledge distillation.

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
