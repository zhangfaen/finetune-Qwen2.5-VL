#!/usr/bin/env python3
"""
Script to run LoRA and full fine-tuning comparison with efficiency measurement.
This script will run both training methods and generate comprehensive comparison plots.
"""

import os
import json
import datetime
from pathlib import Path

# Import training function from the main training script
import sys
sys.path.append('.')
from finetune_distributed import train
from finetune_distributed import compare_training_methods

def run_training_with_lora():
    """Run training with LoRA enabled."""
    print("Starting LoRA training...")
    
    try:
        # Call training function directly with LoRA enabled
        train(use_lora=True)
        print("LoRA training completed successfully")
        return True
    except Exception as e:
        print(f"LoRA training failed: {e}")
        return False

def run_training_without_lora():
    """Run training without LoRA (full fine-tuning)."""
    print("Starting full fine-tuning...")
    
    try:
        # Call training function directly with LoRA disabled
        train(use_lora=False)
        print("Full fine-tuning completed successfully")
        return True
    except Exception as e:
        print(f"Full fine-tuning failed: {e}")
        return False

def find_latest_output_dir():
    """Find the most recent output directory."""
    train_output_dir = Path('train_output')
    if not train_output_dir.exists():
        return None
    
    # Get all subdirectories sorted by creation time
    dirs = [d for d in train_output_dir.iterdir() if d.is_dir()]
    if not dirs:
        return None
    
    # Sort by modification time (newest first)
    dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return dirs[0]

def load_loss_data(output_dir):
    """Load loss data from output directory."""
    lora_data = None
    full_data = None
    
    # Look for LoRA loss data
    lora_json = output_dir / 'lora_loss_data.json'
    if lora_json.exists():
        with open(lora_json, 'r') as f:
            lora_data = json.load(f)
    
    # Look for full fine-tuning loss data
    full_json = output_dir / 'full_loss_data.json'
    if full_json.exists():
        with open(full_json, 'r') as f:
            full_data = json.load(f)
    
    return lora_data, full_data

def main():
    """Main function to run comparison."""
    print("Starting LoRA vs Full Fine-tuning Comparison")
    print("=" * 50)
    
    # Create comparison output directory
    comparison_dir = f'comparison_output/{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}/'
    os.makedirs(comparison_dir, exist_ok=True)
    
    # Run LoRA training
    print("\n1. Running LoRA training...")
    lora_success = run_training_with_lora()
    
    if lora_success:
        lora_output_dir = find_latest_output_dir()
        if lora_output_dir:
            print(f"LoRA training output saved to: {lora_output_dir}")
    
    # Run full fine-tuning
    print("\n2. Running full fine-tuning...")
    full_success = run_training_without_lora()
    
    if full_success:
        full_output_dir = find_latest_output_dir()
        if full_output_dir:
            print(f"Full fine-tuning output saved to: {full_output_dir}")
    
    # Load and compare results
    if lora_success and full_success:
        print("\n3. Loading training results...")
        
        # Load LoRA data
        lora_data, _ = load_loss_data(lora_output_dir)
        
        # Load full fine-tuning data
        _, full_data = load_loss_data(full_output_dir)
        
        if lora_data and full_data:
            print("4. Generating comparison plots...")
            
            # Import the comparison function
            import sys
            sys.path.append('.')
            
            # Create a simple logger for the comparison
            import logging
            logger = logging.getLogger('comparison')
            logger.setLevel(logging.INFO)
            
            # Generate comparison
            compare_training_methods(lora_data, full_data, comparison_dir)
            
            print(f"Comparison results saved to: {comparison_dir}")
        else:
            print("Warning: Could not load loss data for comparison")
    else:
        print("Warning: One or both training runs failed")
    
    print("\nComparison completed!")

if __name__ == "__main__":
    main() 