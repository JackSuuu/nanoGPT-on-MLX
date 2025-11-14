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
- roneneldan/TinyStories
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

A 53-million parameter GPT model trained from scratch on TinyStories dataset. This model implements a **Pre-LayerNorm (Pre-LN) transformer architecture** and serves as a demonstration of efficient training on Apple Silicon using the MLX framework.

> **Model Format:** PyTorch (cross-platform compatible)  
> **Training Framework:** Apple MLX (exported to PyTorch for universal compatibility)  
> **Best for:** Educational demonstrations, research, and fine-tuning on specific domains

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
- **Dataset:** TinyStories - Simple children's stories for language learning
- **Training Hardware:** Apple M2 Pro (16GB unified memory)
- **Checkpoint:** 20000 iterations
- **Training Method:** Base pretraining from scratch

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

- **Dataset:** TinyStories (simple children's stories)
- **Training Tokens:** ~2M training tokens
- **Total Iterations:** 20,000
- **Batch Size:** 12 sequences/batch
- **Sequence Length:** 512 tokens
- **Learning Rate:** 3e-4 with cosine decay schedule
- **Optimizer:** AdamW (β1=0.9, β2=0.95, weight_decay=0.1)
- **Final Training Loss:** 0.7583
- **Training Time:** ~4 hours on Apple M2 Pro
- **Gradient Accumulation:** None (direct updates)

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

> **Note:** All benchmarks measured at checkpoint 20000 (this release).

## Usage

### Basic Text Generation

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model and tokenizer (requires trust_remote_code for custom architecture)
tokenizer = AutoTokenizer.from_pretrained("jacksuuuu/tinystories")
model = AutoModelForCausalLM.from_pretrained(
    "jacksuuuu/tinystories",
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

**Generated:**
```
Once upon a time, the boy named Lily and his dog named Max went for a walk. 
They ran and ran, but they kept each and got very tired. Suddenly the way, 
Max saw something shiny on the ground. He pointed the shiny to his owner and 
explained, "What does this?"

Max meowed and said, "I don't sign, Max. The sign is too small and it's 
important to learn."
```

**Note:** This model generates coherent short stories and educational content. While grammatically imperfect due to its small size (53M params), it demonstrates good narrative flow and vocabulary learned from FineWebEdu dataset.

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

- **Context length:** Limited to 512 tokens (can't process longer documents)
- **Domain:** Trained primarily on educational web content (FineWebEdu)
- **Model size:** 53M parameters - significantly smaller than modern LLMs (1B+)
- **Generation quality:** Produces coherent narratives but with occasional grammatical errors
- **Factual accuracy:** Limited by small model size and training data
- **No instruction tuning:** Base language model - cannot follow instructions or engage in dialogue
- **Training data:** Only 10M tokens (modern models use trillions)

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
  url = {https://huggingface.co/jacksuuuu/tinystories}
}
```

## Additional Resources

- **GitHub Repository:** [JackSuuu/nanoGPT-on-MLX](https://github.com/JackSuuu/nanoGPT-on-MLX)
- **MLX Framework:** [ml-explore/mlx](https://github.com/ml-explore/mlx)
- **Training Dataset:** [roneneldan/TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)

## License

MIT License - See repository for details.
