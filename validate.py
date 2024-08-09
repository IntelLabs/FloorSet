import copy
import numpy as np
import torch
from torch.utils.data import DataLoader
from shapely.affinity import rotate, translate
from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union

from prime_dataset import FloorplanDataset, floorplan_collate
from visualize import get_hard_color, visualize_prime

from utils import unpad_tensor, calculate_difference_with_tolerance
from utils import normalize_polygon, polygons_have_same_shape, check_fixed_const, check_preplaced_const, check_mib_const, check_boundary_const, check_clust_const
from cost import calculate_weighted_b2b_wirelength, calculate_weighted_p2b_wirelength, estimate_cost


##sample input for input solution tensor with 21-partitions
"""
[tensor([[180.,  45.],
         [180.,  60.],
         [186.,  60.],
         [186.,  45.],
         [180.,  45.]], dtype=torch.float16),
 tensor([[180.,  75.],
         [180., 105.],
         [186., 105.],
         [186.,  75.],
         [180.,  75.]], dtype=torch.float16),
 tensor([[150.,  30.],
         [150.,   0.],
         [120.,   0.],
         [120.,  30.],
         [150.,  30.]], dtype=torch.float16),
 tensor([[135.,  60.],
         [135.,  75.],
         [186.,  75.],
         [186.,  60.],
         [135.,  60.]], dtype=torch.float16),
 tensor([[ 60., 135.],
         [ 90., 135.],
         [ 90.,  75.],
         [ 60.,  75.],
         [ 60., 135.]], dtype=torch.float16),
 tensor([[120.,  60.],
         [120.,  75.],
         [135.,  75.],
         [135.,  45.],
         [120.,  45.],
         [120.,  60.]], dtype=torch.float16),
 tensor([[90., 60.],
         [90.,  0.],
         [60.,  0.],
         [60., 60.],
         [90., 60.]], dtype=torch.float16),
 tensor([[120.,  30.],
         [120.,   0.],
         [ 90.,   0.],
         [ 90.,  45.],
         [135.,  45.],
         [135.,  60.],
         [150.,  60.],
         [150.,  30.],
         [120.,  30.]], dtype=torch.float16),
 tensor([[186.,  45.],
         [186.,  30.],
         [150.,  30.],
         [150.,  60.],
         [180.,  60.],
         [180.,  45.],
         [186.,  45.]], dtype=torch.float16),
 tensor([[180.,  75.],
         [120.,  75.],
         [120., 105.],
         [180., 105.],
         [180.,  75.]], dtype=torch.float16),
 tensor([[135., 165.],
         [135., 135.],
         [150., 135.],
         [150., 105.],
         [120., 105.],
         [120., 165.],
         [135., 165.]], dtype=torch.float16),
 tensor([[  0., 186.],
         [ 30., 186.],
         [ 30., 135.],
         [  0., 135.],
         [  0., 186.]], dtype=torch.float16),
 tensor([[150., 186.],
         [150., 135.],
         [135., 135.],
         [135., 165.],
         [120., 165.],
         [120., 186.],
         [150., 186.]], dtype=torch.float16),
 tensor([[150.,  30.],
         [186.,  30.],
         [186.,   0.],
         [150.,   0.],
         [150.,  30.]], dtype=torch.float16),
 tensor([[ 60.,  75.],
         [ 30.,  75.],
         [ 30.,  60.],
         [  0.,  60.],
         [  0., 105.],
         [ 60., 105.],
         [ 60.,  75.]], dtype=torch.float16),
 tensor([[60.,  0.],
         [ 0.,  0.],
         [ 0., 30.],
         [60., 30.],
         [60.,  0.]], dtype=torch.float16),
 tensor([[45., 45.],
         [60., 45.],
         [60., 30.],
         [ 0., 30.],
         [ 0., 60.],
         [30., 60.],
         [30., 75.],
         [45., 75.],
         [45., 45.]], dtype=torch.float16),
 tensor([[60., 75.],
         [90., 75.],
         [90., 60.],
         [60., 60.],
         [60., 45.],
         [45., 45.],
         [45., 75.],
         [60., 75.]], dtype=torch.float16),
 tensor([[ 45., 105.],
         [  0., 105.],
         [  0., 135.],
         [ 30., 135.],
         [ 30., 186.],
         [ 45., 186.],
         [ 45., 105.]], dtype=torch.float16),
 tensor([[180., 186.],
         [186., 186.],
         [186., 105.],
         [150., 105.],
         [150., 186.],
         [180., 186.]], dtype=torch.float16),
 tensor([[ 45., 186.],
         [120., 186.],
         [120.,  45.],
         [ 90.,  45.],
         [ 90., 135.],
         [ 60., 135.],
         [ 60., 105.],
         [ 45., 105.],
         [ 45., 186.]], dtype=torch.float16)] 
"""



from tqdm import tqdm
import itertools
root = './'
ds = FloorplanDataset(root)
place_viol = 0

dl = DataLoader(
    ds, 
    batch_size=100, 
    shuffle=False,
    collate_fn=floorplan_collate
)

# example to estimate cost of training data, by iterating over the dataloader
for batch_idx, batch in tqdm(enumerate(dl), total=len(dl), desc='Processing FloorSet-Prime Batches'):
  
    area_target, b2b_connectivity, p2b_connectivity, pins_pos, placement_constraints = batch[0]
    sol, metrics = batch[1]

    for bind in range(sol.shape[0]):
        test_layout = sol[bind]##needs to be modified by predicted solution 
        
        result = estimate_cost(test_layout, area_target[bind], b2b_connectivity[bind], p2b_connectivity[bind], pins_pos[bind], 
                               placement_constraints[bind], sol[bind], metrics[bind])
        

        print(result)
        break
    break

