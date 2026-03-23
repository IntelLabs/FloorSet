#!/usr/bin/env python3
"""
ICCAD 2026 FloorSet Challenge - Optimizer Template

HOW TO USE:
1. Copy this file: cp optimizer_template.py my_optimizer.py
2. Implement your algorithm in the MyOptimizer.solve() method
3. Test: python contest.py --evaluate my_optimizer.py
4. Validate: python contest.py --validate my_optimizer.py

TRAINING DATA (1M samples) + COST FUNCTION:
  # Explore data
  python contest.py --training
  
  # In your training code:
  from contest import get_training_dataloader, compute_training_loss
  
  dataloader = get_training_dataloader(batch_size=64, num_samples=10000)
  for inputs, labels in dataloader:
      area_target, b2b_conn, p2b_conn, pins_pos, constraints = inputs
      polygons, metrics = labels  # Ground truth solutions
      
      # Your model predicts positions
      predicted = my_model(inputs)
      
      # Compute loss (SAME cost function as final evaluation)
      loss = compute_training_loss(
          predicted, b2b_conn, p2b_conn, pins_pos, area_target
      )['total']
      
      loss.backward()  # For gradient-based methods

Your optimizer will receive:
  - block_count: Number of blocks to place  
  - area_targets: Target area per block [n_blocks]
  - b2b_connectivity: Block-to-block edges [n_edges x 3] (i, j, weight)
  - p2b_connectivity: Pin-to-block edges [n_edges x 3] (pin_idx, block_idx, weight)
  - pins_pos: Pin positions [n_pins x 2]
  - constraints: Constraint flags [n_blocks x 5]
    - col 0: Fixed (shape must match ground truth)
    - col 1: Preplaced (shape AND position must match)
    - col 2: MIB group ID (>0 means blocks must have same shape)
    - col 3: Cluster group ID (>0 means blocks must be contiguous)
    - col 4: Boundary (bitfield: LEFT=1, RIGHT=2, TOP=4, BOTTOM=8)

Your optimizer must return:
  - List of (x, y, width, height) tuples, one per block
  - MUST return exactly `block_count` tuples

HARD CONSTRAINTS (violation = infeasible, Cost = 10.0):
  - NO OVERLAPS: Blocks cannot overlap (touching edges OK)
  - AREA TOLERANCE: Block area (w × h) must be within 1% of area_targets[i]
"""

import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch

# Add contest directory for imports
sys.path.insert(0, str(Path(__file__).parent))

# Import cost function for querying during optimization
from contest import (
    FloorplanOptimizer,
    calculate_hpwl_b2b,
    calculate_hpwl_p2b,
    calculate_bbox_area,
    check_overlap,
    get_training_dataloader,      # For loading 1M training samples
    compute_training_loss,        # For computing loss during training
    compute_training_loss_batch   # For batch loss computation
)


class MyOptimizer(FloorplanOptimizer):
    """
    YOUR OPTIMIZER IMPLEMENTATION
    
    Replace this with your algorithm (Simulated Annealing, Genetic Algorithm,
    Reinforcement Learning, Sequence Pair, B*-Tree, etc.)
    """
    
    def __init__(self, verbose: bool = False):
        super().__init__(verbose)
        # Add your initialization here
        self.max_iterations = 10000
    
    def solve(
        self,
        block_count: int,
        area_targets: torch.Tensor,
        b2b_connectivity: torch.Tensor,
        p2b_connectivity: torch.Tensor,
        pins_pos: torch.Tensor,
        constraints: torch.Tensor
    ) -> List[Tuple[float, float, float, float]]:
        """
        YOUR ALGORITHM GOES HERE
        
        Args:
            block_count: Number of blocks (21-120)
            area_targets: [n_blocks] target area for each block
            b2b_connectivity: [n_edges, 3] block-to-block edges (i, j, weight)
            p2b_connectivity: [n_edges, 3] pin-to-block edges (pin, block, weight)
            pins_pos: [n_pins, 2] pin (x, y) positions
            constraints: [n_blocks, 5] constraint flags per block
        
        Returns:
            List of exactly `block_count` tuples: [(x, y, width, height), ...]
        
        HARD CONSTRAINTS (violation = Cost 10.0):
            - NO OVERLAPS between blocks (touching edges OK)
            - AREA: w*h must be within 1% of area_targets[i]
        """
        
        # =====================================================================
        # STEP 1: Initialize block dimensions from target areas
        # =====================================================================
        positions = []
        total_area = sum(float(area_targets[i]) for i in range(block_count) 
                        if area_targets[i] > 0)
        canvas_size = math.sqrt(total_area) * 1.5  # Allow some slack
        
        for i in range(block_count):
            area = float(area_targets[i]) if area_targets[i] > 0 else 1.0
            # Start with square blocks (you can optimize aspect ratios)
            w = h = math.sqrt(area)
            # Random initial placement
            x = random.uniform(0, max(0, canvas_size - w))
            y = random.uniform(0, max(0, canvas_size - h))
            positions.append([x, y, w, h])
        
        # =====================================================================
        # STEP 2: YOUR OPTIMIZATION ALGORITHM
        # =====================================================================
        
        # Example: Simple hill-climbing (REPLACE WITH YOUR ALGORITHM)
        current_cost = self._evaluate(positions, b2b_connectivity, 
                                      p2b_connectivity, pins_pos)
        best_positions = [tuple(p) for p in positions]
        best_cost = current_cost
        
        for iteration in range(self.max_iterations):
            # Pick a random block to move
            idx = random.randint(0, block_count - 1)
            old_pos = positions[idx].copy()
            
            # Random perturbation
            positions[idx][0] += random.gauss(0, canvas_size * 0.05)
            positions[idx][1] += random.gauss(0, canvas_size * 0.05)
            positions[idx][0] = max(0, positions[idx][0])
            positions[idx][1] = max(0, positions[idx][1])
            
            # Evaluate new solution
            new_cost = self._evaluate(positions, b2b_connectivity,
                                     p2b_connectivity, pins_pos)
            
            # Accept if better (or add SA acceptance criterion)
            if new_cost < current_cost:
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_positions = [tuple(p) for p in positions]
            else:
                # Reject - restore old position
                positions[idx] = old_pos
        
        # =====================================================================
        # STEP 3: Handle constraints (IMPLEMENT BASED ON YOUR APPROACH)
        # =====================================================================
        # TODO: Handle fixed, preplaced, MIB, cluster, boundary constraints
        # See contest.py for constraint column meanings
        
        return best_positions
    
    def _evaluate(self, positions, b2b_conn, p2b_conn, pins_pos) -> float:
        """
        Evaluate current solution quality.
        
        You can use the imported functions:
        - calculate_hpwl_b2b(positions, b2b_connectivity)
        - calculate_hpwl_p2b(positions, p2b_connectivity, pins_pos)  
        - calculate_bbox_area(positions)
        - check_overlap(positions)
        """
        pos_tuples = [tuple(p) for p in positions]
        
        hpwl_b2b = calculate_hpwl_b2b(pos_tuples, b2b_conn)
        hpwl_p2b = calculate_hpwl_p2b(pos_tuples, p2b_conn, pins_pos)
        area = calculate_bbox_area(pos_tuples)
        overlaps = check_overlap(pos_tuples)
        
        # Combine objectives (tune these weights for your approach)
        return hpwl_b2b + hpwl_p2b + area * 0.01 + overlaps * 10000


# =============================================================================
# You can also use the built-in baselines for comparison
# =============================================================================
# from contest import RandomOptimizer, SimulatedAnnealingOptimizer


# =============================================================================
# TRAINING EXAMPLE (for neural network / RL approaches)
# =============================================================================
def training_example():
    """
    Example showing how to train a model using the 1M training samples
    with the SAME cost function used for final evaluation.
    
    Run this with: python optimizer_template.py --train
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
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--train':
        # Show training example
        training_example()
    else:
        # Quick inference test
        print("Testing MyOptimizer...")
        print("(Run with --train flag to see training example)\n")
        
        optimizer = MyOptimizer(verbose=True)
        
        # Dummy test data
        block_count = 10
        area_targets = torch.tensor([100.0] * block_count)
        b2b_connectivity = torch.tensor([
            [0, 1, 1.0], [1, 2, 1.0], [2, 3, 1.0], [3, 4, 1.0],
            [4, 5, 1.0], [5, 6, 1.0], [6, 7, 1.0], [7, 8, 1.0], [8, 9, 1.0]
        ])
        p2b_connectivity = torch.tensor([[0, 0, 1.0], [1, 9, 1.0]])
        pins_pos = torch.tensor([[0.0, 0.0], [100.0, 100.0]])
        constraints = torch.zeros(block_count, 5)
        
        positions = optimizer.solve(
            block_count, area_targets, b2b_connectivity,
            p2b_connectivity, pins_pos, constraints
        )
        
        print(f"\nResult: {len(positions)} blocks placed")
        print(f"Sample positions: {positions[:3]}...")
        
        # Evaluate using contest cost function
        loss_dict = compute_training_loss(
            positions, b2b_connectivity, p2b_connectivity, 
            pins_pos, area_targets, return_components=True
        )
        
        print(f"\nMetrics (using contest cost function):")
        print(f"  HPWL:     {loss_dict['hpwl_total']:.2f}")
        print(f"  Area:     {loss_dict['bbox_area']:.2f}")
        print(f"  Overlaps: {loss_dict['overlap_count']}")
        print(f"  TOTAL:    {loss_dict['total']:.2f}")
        print("\nRun 'python contest.py --evaluate optimizer_template.py' for full evaluation.")
