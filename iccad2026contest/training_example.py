#!/usr/bin/env python3
"""
ICCAD 2026 FloorSet Challenge - Training Data Example

Shows how to train a neural network using the DIFFERENTIABLE contest cost function.
Run: python iccad2026contest/training_example.py

The loss is the SAME formula as contest evaluation:
  Cost = (1 + α·(HPWL_gap + Area_gap)) × exp(β·V_soft)

But implemented with differentiable operations for .backward()
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from iccad2026contest.iccad2026_evaluate import (
    get_training_dataloader,
    compute_training_loss_differentiable,
)


def main():
    print("="*70)
    print("ICCAD 2026 FloorSet Challenge - Training Example")
    print("Using DIFFERENTIABLE contest cost function")
    print("="*70)
    
    # =========================================================================
    # Load training data (1M samples)
    # =========================================================================
    print("\nLoading training data...")
    dataloader = get_training_dataloader(
        batch_size=1,
        num_samples=3,  # Use None for all 1M samples
        shuffle=False
    )
    print(f"Loaded {len(dataloader)} samples\n")
    
    # =========================================================================
    # Training loop
    # =========================================================================
    for batch_idx, batch in enumerate(dataloader):
        # Unpack batch - 8 tensors
        area_target, b2b_conn, p2b_conn, pins_pos, constraints, tree_sol, fp_sol, metrics = batch
        
        # Remove batch dimension (batch_size=1)
        area_target = area_target.squeeze(0)
        b2b_conn = b2b_conn.squeeze(0)
        p2b_conn = p2b_conn.squeeze(0)
        pins_pos = pins_pos.squeeze(0)
        metrics = metrics.squeeze(0)
        fp_sol = fp_sol.squeeze(0)
        
        # Count valid blocks (non-padded)
        block_count = int((area_target != -1).sum().item())
        
        print(f"Sample {batch_idx}: {block_count} blocks")
        
        # =================================================================
        # YOUR NEURAL NETWORK HERE
        # =================================================================
        # Example: positions = model(area_target, b2b_conn, p2b_conn, ...)
        #
        # Output shape: [block_count, 4] = (x, y, w, h) per block
        #
        # For demo, use ground truth with noise (to show gradients work)
        ground_truth = fp_sol[:block_count]  # [w, h, x, y]
        # Reorder to [x, y, w, h] for our loss function
        positions = torch.zeros(block_count, 4, requires_grad=True)
        positions = torch.stack([
            ground_truth[:, 2] + torch.randn(block_count) * 5,  # x + noise
            ground_truth[:, 3] + torch.randn(block_count) * 5,  # y + noise
            ground_truth[:, 0],  # w
            ground_truth[:, 1],  # h
        ], dim=1)
        positions = positions.clone().detach().requires_grad_(True)
        
        # =================================================================
        # DIFFERENTIABLE CONTEST COST FUNCTION
        # Same formula as actual evaluation!
        # Cost = (1 + α·(HPWL_gap + Area_gap)) × exp(β·V_soft)
        # =================================================================
        loss = compute_training_loss_differentiable(
            positions,
            b2b_conn,
            p2b_conn,
            pins_pos,
            area_target[:block_count],
            metrics
        )
        
        print(f"  Contest Cost (differentiable): {loss.item():.4f}")
        
        # Verify gradients flow
        loss.backward()
        print(f"  Gradient exists: {positions.grad is not None}")
        print(f"  Gradient norm: {positions.grad.norm().item():.4f}")
        print()
        
        # In real training:
        # optimizer.zero_grad()
        # positions = model(inputs)
        # loss = compute_training_loss_differentiable(positions, ...)
        # loss.backward()
        # optimizer.step()
    
    print("="*70)
    print("Training data format:")
    print("  - positions: [N, 4] tensor of (x, y, w, h) per block")
    print("  - Loss = contest cost formula, fully differentiable")
    print("")
    print("The loss includes:")
    print("  - HPWL gap vs ground truth baseline")
    print("  - Area gap vs ground truth baseline")
    print("  - Overlap violation (soft, differentiable)")
    print("  - Area tolerance violation (soft, differentiable)")
    print("")
    print("Final evaluation: python iccad2026_evaluate.py --evaluate your_optimizer.py")
    print("="*70)


if __name__ == '__main__':
    main()
