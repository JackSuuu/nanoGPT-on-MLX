"""
Test script for FineWebEdu dataset loading
Run this before training to verify the dataset works correctly
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config as cfg
from src.data import create_datasets


def test_finewebedu_loading():
    """Test FineWebEdu dataset loading and batch generation"""
    
    # Create config dict with FineWebEdu
    config_dict = {k: v for k, v in vars(cfg).items() if not k.startswith('_')}
    config_dict['dataset_name'] = 'finewebedu'
    
    # Reduce tokens for faster testing
    print("Testing FineWebEdu with reduced token count (1M for testing)...")
    config_dict['max_tokens'] = 1_000_000  # 1M tokens for quick test
    
    print("\n" + "="*70)
    print("TESTING FINEWEBEDU DATASET LOADING")
    print("="*70 + "\n")
    
    # Load datasets
    print("Loading datasets...")
    train_dataset, val_dataset = create_datasets(config_dict)
    
    print(f"\n✓ Train tokens: {len(train_dataset.tokens):,}")
    print(f"✓ Val tokens: {len(val_dataset.tokens):,}")
    
    # Test batch generation
    print("\n" + "-"*70)
    print("Testing batch generation...")
    print("-"*70 + "\n")
    
    x, y = train_dataset.get_batch(4)
    print(f"✓ Batch shapes: x={x.shape}, y={y.shape}")
    print(f"✓ Sample token IDs: {x[0, :10]}")
    
    # Decode samples from multiple batches
    print("\n" + "-"*70)
    print("Sample texts from FineWebEdu:")
    print("-"*70 + "\n")
    
    for i in range(3):
        x, y = train_dataset.get_batch(1)
        sample_text = train_dataset.tokenizer.decode(x[0, :200].tolist())
        print(f"Sample {i+1}:")
        print(sample_text)
        print("\n" + "."*70 + "\n")
    
    # Validation set test
    print("-"*70)
    print("Testing validation set:")
    print("-"*70 + "\n")
    
    x_val, y_val = val_dataset.get_batch(2)
    print(f"✓ Val batch shapes: x={x_val.shape}, y={y_val.shape}")
    
    val_text = val_dataset.tokenizer.decode(x_val[0, :150].tolist())
    print(f"\nValidation sample:")
    print(val_text)
    
    print("\n" + "="*70)
    print("✅ ALL TESTS PASSED!")
    print("="*70)
    print("\nFineWebEdu is ready for training!")
    print(f"Train: {len(train_dataset.tokens):,} tokens")
    print(f"Val: {len(val_dataset.tokens):,} tokens")


if __name__ == "__main__":
    test_finewebedu_loading()
