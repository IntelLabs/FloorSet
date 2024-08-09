# Importing necessary libraries and modules
import copy
import numpy as np
import torch
from torch.utils.data import DataLoader
from shapely.affinity import rotate, translate
from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union

# Importing custom modules for dataset handling, visualization, and utility functions
from prime_dataset import FloorplanDataset, floorplan_collate
from visualize import get_hard_color, visualize_prime
from utils import (unpad_tensor, calculate_difference_with_tolerance,
                   normalize_polygon, polygons_have_same_shape, check_fixed_const,
                   check_preplaced_const, check_mib_const, check_boundary_const,
                   check_clust_const)
from cost import calculate_weighted_b2b_wirelength, calculate_weighted_p2b_wirelength, estimate_cost

from tqdm import tqdm  # Importing for progress tracking
import itertools

# Sample input for input solution tensor with 21 partitions
"""
The input tensor is a list of 21 tensors, each representing a polygon with points defined in torch.float16.
Example:
[
 tensor([[180.,  45.],
         [180.,  60.],
         [186.,  60.],
         [186.,  45.],
         [180.,  45.]], dtype=torch.float16),
 tensor([[180.,  75.],
         [180., 105.],
         [186., 105.],
         [186.,  75.],
         [180.,  75.]], dtype=torch.float16),
 ...
]
"""

# Define the root directory for the dataset
root = './'

# Create the FloorplanDataset
ds = FloorplanDataset(root)

# Initialize a variable to count placement violations
place_viol = 0

# Create a DataLoader for iterating over the dataset
dl = DataLoader(
    ds, 
    batch_size=100, 
    shuffle=False,
    collate_fn=floorplan_collate
)

# Iterate over the DataLoader to estimate the cost of training data
for batch_idx, batch in tqdm(enumerate(dl), total=len(dl), desc='Processing FloorSet-Prime Batches'):
    
    # Unpack the batch into respective components
    area_target, b2b_connectivity, p2b_connectivity, pins_pos, placement_constraints = batch[0]
    sol, metrics = batch[1]

    # Iterate over each solution in the batch
    for bind in range(sol.shape[0]):
        # Assign the test layout to the current solution
        test_layout = sol[bind]  # This will be modified by the predicted solution
        
        # Estimate the cost using the current test layout and related parameters
        result = estimate_cost(
            test_layout, 
            area_target[bind], 
            b2b_connectivity[bind], 
            p2b_connectivity[bind], 
            pins_pos[bind], 
            placement_constraints[bind], 
            sol[bind], 
            metrics[bind]
        )

        # Print the estimated result
        print(result)
        
        # Break after the first solution is processed
        break
    
    # Break after the first batch is processed
    break
