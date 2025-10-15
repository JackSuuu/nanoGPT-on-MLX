#!/usr/bin/env python3
"""
Quick dataset switcher - easily switch between TinyStories and FineWebEdu
Usage: python scripts/switch_dataset.py [tinystories|finewebedu]
"""
import sys
import os


def switch_dataset(dataset_name):
    """Switch the dataset in config.py"""
    config_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'config.py')
    
    # Read current config
    with open(config_path, 'r') as f:
        lines = f.readlines()
    
    # Update dataset_name line
    updated = False
    for i, line in enumerate(lines):
        if line.strip().startswith('dataset_name ='):
            lines[i] = f'dataset_name = "{dataset_name}"  # Options: "tinystories", "finewebedu"\n'
            updated = True
            print(f"✓ Updated config.py: dataset_name = '{dataset_name}'")
            break
    
    if not updated:
        print("Error: Could not find 'dataset_name' in config.py")
        return False
    
    # Write updated config
    with open(config_path, 'w') as f:
        f.writelines(lines)
    
    # Show recommendations
    print("\nDataset switched successfully! 🎉\n")
    
    if dataset_name == 'tinystories':
        print("TinyStories selected:")
        print("  - Simple children's stories")
        print("  - Good for: Initial training, small models")
        print("  - Recommended: 2M tokens")
        print("\nNext steps:")
        print("  1. python -m src.train")
        print("  2. python -m src.generate")
    else:
        print("FineWebEdu selected:")
        print("  - Diverse web content (science, tech, culture, etc.)")
        print("  - Good for: Distillation, general knowledge")
        print("  - Recommended: 10M+ tokens")
        print("\nNext steps:")
        print("  1. python test/test_finewebedu.py  # Test first")
        print("  2. python -m src.distill  # Distillation recommended")
        print("  3. python -m src.generate")
    
    return True


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/switch_dataset.py [tinystories|finewebedu]")
        print("\nCurrent datasets available:")
        print("  - tinystories: Simple children's stories")
        print("  - finewebedu: Diverse web content")
        sys.exit(1)
    
    dataset = sys.argv[1].lower()
    
    if dataset not in ['tinystories', 'finewebedu']:
        print(f"Error: Unknown dataset '{dataset}'")
        print("Choose from: tinystories, finewebedu")
        sys.exit(1)
    
    switch_dataset(dataset)
