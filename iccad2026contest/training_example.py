#!/usr/bin/env python3
"""
ICCAD 2026 FloorSet Challenge - Training Data Example

Shows how to use the 1M training samples with the contest cost function.
Run: python training_example.py

For neural network / RL approaches, use this pattern in your training loop.
"""

import math
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

from iccad2026_evaluate import (
    get_training_dataloader,
    compute_training_loss,
    compute_training_loss_batch
)


def training_example():
    """
    Example showing how to train a model using the 1M training samples
    with the SAME cost function used for final evaluation.
    """
    print("="*70)
    print("TRAINING EXAMPLE - Using 1M samples with contest cost function")
    print("="*70)
    
    # =========================================================================
    # STEP 1: Load training data
    # =========================================================================
    print("\n[1] Loading training data...")
    dataloader = get_training_dataloader(
        batch_size=1,           # Use larger batch for real training
        num_samples=5,          # Use None for all 1M samples
        shuffle=False
    )
    print(f"    Loaded dataloader with {len(dataloader)} batches")
    
    # =========================================================================
    # STEP 2: Training loop with cost function
    # =========================================================================
    print("\n[2] Training loop example:")
    print("-"*70)
    
    for batch_idx, (inputs, labels) in enumerate(dataloader):
        # Unpack inputs
        area_target, b2b_conn, p2b_conn, pins_pos, constraints = inputs
        polygons, metrics = labels  # Ground truth (for supervised learning)
        
        block_count = int((area_target != -1).sum().item())
        print(f"\n    Sample {batch_idx}: {block_count} blocks")
        
        # YOUR MODEL would predict positions here
        # For demo, we use random positions
        predicted_positions = []
        canvas = math.sqrt(sum(float(a) for a in area_target[:block_count] if a > 0)) * 1.5
        for i in range(block_count):
            area = float(area_target[i]) if area_target[i] > 0 else 1.0
            w = h = math.sqrt(area)
            x = random.uniform(0, canvas - w)
            y = random.uniform(0, canvas - h)
            predicted_positions.append((x, y, w, h))
        
        # =====================================================================
        # COMPUTE LOSS - Same cost function as final evaluation!
        # =====================================================================
        loss_dict = compute_training_loss(
            predicted_positions,
            b2b_conn, p2b_conn, pins_pos, area_target,
            return_components=True
        )
        
        print(f"    Loss breakdown:")
        print(f"      HPWL (b2b):       {loss_dict['hpwl_b2b']:.2f}")
        print(f"      HPWL (p2b):       {loss_dict['hpwl_p2b']:.2f}")
        print(f"      HPWL (total):     {loss_dict['hpwl_total']:.2f}")
        print(f"      Bounding box:     {loss_dict['bbox_area']:.2f}")
        print(f"      Overlaps:         {loss_dict['overlap_count']}")
        print(f"      Area violations:  {loss_dict['area_violations']}")
        print(f"      ─────────────────────────────────")
        print(f"      TOTAL LOSS:       {loss_dict['total']:.2f}")
        
        # In real training:
        # loss = loss_dict['total']
        # loss.backward()  # If using PyTorch autograd
        # optimizer.step()
    
    print("\n" + "="*70)
    print("Training example complete!")
    print("In your real code:")
    print("  1. Replace random positions with your model's predictions")
    print("  2. Use loss_dict['total'] for backpropagation")
    print("  3. The loss uses the SAME cost function as final evaluation")
    print("="*70)


if __name__ == '__main__':
    training_example()
