"""
Custom HuggingFace-compatible GPT model with Pre-LN architecture
Matches the MLX nanoGPT implementation exactly
"""
import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
import math


class NanoGPTConfig(PretrainedConfig):
    """Configuration for NanoGPT model"""
    model_type = "nanogpt"
    
    # Add attribute mapping for HuggingFace compatibility
    attribute_map = {
        "hidden_size": "n_embd",
        "num_hidden_layers": "n_layer",
        "num_attention_heads": "n_head",
        "intermediate_size": "n_inner",
        "max_position_embeddings": "n_positions",
    }
    
    def __init__(
        self,
        vocab_size=50257,
        n_positions=512,
        n_embd=384,
        n_layer=8,
        n_head=8,
        n_inner=1536,
        activation_function="gelu",
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        bos_token_id=50256,
        eos_token_id=50256,
        **kwargs
    ):
        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)
        self.vocab_size = vocab_size
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_inner = n_inner
        self.activation_function = activation_function
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        
        # Add standard HuggingFace attributes for compatibility
        self.hidden_size = n_embd
        self.num_hidden_layers = n_layer
        self.num_attention_heads = n_head
        self.intermediate_size = n_inner
        self.max_position_embeddings = n_positions


class NanoGPTAttention(nn.Module):
    """Multi-head self-attention with Pre-LN"""
    
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        self.scale = math.sqrt(self.head_dim)
        
        # Combined QKV projection (standard Linear, not Conv1D)
        self.qkv_proj = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        
        # Causal mask
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.n_positions, config.n_positions)).view(
                1, 1, config.n_positions, config.n_positions
            ),
        )
        
    def forward(self, x):
        B, T, C = x.size()  # batch, sequence length, embedding dim
        
        # Project and split into Q, K, V
        qkv = self.qkv_proj(x)  # (B, T, 3*C)
        qkv = qkv.view(B, T, 3, self.n_head, self.head_dim)  # (B, T, 3, n_head, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, n_head, T, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        scores = (q @ k.transpose(-2, -1)) / self.scale  # (B, n_head, T, T)
        
        # Apply causal mask
        scores = scores.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Combine heads
        out = attn_weights @ v  # (B, n_head, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)
        
        return self.resid_dropout(self.out_proj(out))


class NanoGPTMLP(nn.Module):
    """Feed-forward network"""
    
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.n_embd, config.n_inner)
        self.fc2 = nn.Linear(config.n_inner, config.n_embd)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(config.resid_pdrop)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return self.dropout(x)


class NanoGPTBlock(nn.Module):
    """Transformer block with Pre-LN architecture"""
    
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = NanoGPTAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.mlp = NanoGPTMLP(config)
        
    def forward(self, x):
        # Pre-norm architecture (LayerNorm before attention/MLP)
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class NanoGPTModel(PreTrainedModel):
    """NanoGPT model with Pre-LN architecture"""
    config_class = NanoGPTConfig
    
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        
        # Embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([NanoGPTBlock(config) for _ in range(config.n_layer)])
        
        # Final layer norm
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        
        # LM head (tied with token_embedding)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Tie weights
        self.lm_head.weight = self.token_embedding.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
    
    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        labels=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        if input_ids is not None:
            batch_size, seq_length = input_ids.size()
        else:
            batch_size, seq_length = inputs_embeds.size()[:-1]
        
        if position_ids is None:
            if input_ids is not None:
                device = input_ids.device
            elif inputs_embeds is not None:
                device = inputs_embeds.device
            else:
                device = next(self.parameters()).device
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        # Embeddings
        if inputs_embeds is None:
            inputs_embeds = self.token_embedding(input_ids)
        
        position_embeds = self.position_embedding(position_ids)
        hidden_states = self.drop(inputs_embeds + position_embeds)
        
        # Transformer blocks
        for block in self.blocks:
            hidden_states = block(hidden_states)
        
        # Final layer norm
        hidden_states = self.ln_f(hidden_states)
        
        # LM head
        lm_logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        if not return_dict:
            output = (lm_logits,)
            return ((loss,) + output) if loss is not None else output
        
        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )


class NanoGPTLMHeadModel(PreTrainedModel):
    """Causal language model wrapper"""
    config_class = NanoGPTConfig
    
    def __init__(self, config):
        super().__init__(config)
        self.transformer = NanoGPTModel(config)
        
    def forward(self, *args, **kwargs):
        return self.transformer(*args, **kwargs)
    
    def generate(self, *args, **kwargs):
        """Use HF's generate method"""
        # Remove unused kwargs that transformers might pass
        kwargs.pop("attention_mask", None)
        kwargs.pop("token_type_ids", None)
        return super().generate(*args, **kwargs)
    
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        # Remove unused kwargs
        kwargs.pop("attention_mask", None)
        kwargs.pop("token_type_ids", None)
        
        # Our model doesn't support KV caching, so we need to pass the full sequence
        return {
            "input_ids": input_ids,
        }
    
    def can_generate(self):
        """Indicate this model can generate"""
        return True
