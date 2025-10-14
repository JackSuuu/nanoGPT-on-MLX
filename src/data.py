"""
Data loading and preprocessing for TinyStories
"""
import os
import numpy as np
import mlx.core as mx
import tiktoken
from datasets import load_dataset
from tqdm import tqdm


class TinyStoriesDataset:
    """TinyStories dataset with tokenization - perfect for small models!"""
    
    def __init__(self, config, split='train'):
        self.config = config
        self.split = split
        self.context_length = config['context_length']
        
        # Initialize GPT-2 tokenizer
        self.tokenizer = tiktoken.get_encoding("gpt2")
        
        # Load and prepare data
        self.tokens = self._load_and_tokenize()
        print(f"{split} dataset: {len(self.tokens):,} tokens")
        
    def _load_and_tokenize(self):
        """Load TinyStories and tokenize"""
        cache_file = f"data/tinystories_{self.split}_tokens.npy"
        
        # Check if cached tokens exist
        if os.path.exists(cache_file):
            print(f"Loading cached tokens from {cache_file}")
            tokens = np.load(cache_file)
            return tokens
        
        print(f"Loading TinyStories {self.split} split...")
        # TinyStories is much simpler than WikiText - perfect for 22M model!
        dataset = load_dataset("roneneldan/TinyStories", split=self.split)
        
        # Concatenate all stories (TinyStories has 'text' field)
        print(f"Processing {len(dataset):,} stories...")
        text = "\n\n".join([item['text'] for item in tqdm(dataset, desc="Loading stories")])
        
        print("Tokenizing...")
        tokens = self.tokenizer.encode(text)
        
        # Subset to max_tokens for faster training
        if self.split == 'train':
            max_tokens = self.config.get('max_tokens', 2_000_000)  # TinyStories is simpler, use more tokens
            tokens = tokens[:max_tokens]
        
        tokens = np.array(tokens, dtype=np.int32)
        
        # Cache tokens
        os.makedirs("data", exist_ok=True)
        np.save(cache_file, tokens)
        print(f"Cached tokens to {cache_file}")
        
        return tokens
    
    def get_batch(self, batch_size):
        """Get a random batch of data"""
        # Random starting indices
        max_start = len(self.tokens) - self.context_length - 1
        starts = np.random.randint(0, max_start, size=batch_size)
        
        # Extract sequences
        x = np.stack([self.tokens[i:i+self.context_length] for i in starts])
        y = np.stack([self.tokens[i+1:i+self.context_length+1] for i in starts])
        
        # Convert to MLX arrays
        x = mx.array(x)
        y = mx.array(y)
        
        return x, y


def create_datasets(config):
    """Create train and validation datasets"""
    # For TinyStories, we use the predefined train/validation splits
    train_dataset = TinyStoriesDataset(config, split='train')
    val_dataset = TinyStoriesDataset(config, split='validation')
    
    return train_dataset, val_dataset


def estimate_loss(model, dataset, config):
    """Estimate loss over multiple batches"""
    losses = []
    eval_iters = config['eval_iters']
    
    # Use tqdm with leave=False and disable if running in background
    for _ in tqdm(range(eval_iters), desc=f"Eval ({dataset.split})", leave=False, ncols=80):
        x, y = dataset.get_batch(config['batch_size'])
        _, loss = model(x, y)
        mx.eval(loss)  # Force evaluation
        losses.append(float(loss))
    return np.mean(losses)


if __name__ == "__main__":
    # Test data loading
    from . import config as cfg
    
    config_dict = {k: v for k, v in vars(cfg).items() if not k.startswith('_')}
    
    print("Testing data loading...")
    train_dataset, val_dataset = create_datasets(config_dict)
    
    print(f"\nTrain tokens: {len(train_dataset.tokens):,}")
    print(f"Val tokens: {len(val_dataset.tokens):,}")
    
    # Test batch generation
    x, y = train_dataset.get_batch(4)
    print(f"\nBatch shapes: x={x.shape}, y={y.shape}")
    print(f"Sample tokens: {x[0, :10]}")
    
    # Decode sample
    sample_text = train_dataset.tokenizer.decode(x[0, :50].tolist())
    print(f"\nSample text:\n{sample_text}")
