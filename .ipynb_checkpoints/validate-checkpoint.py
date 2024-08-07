import copy
import numpy as np
import torch
from torch.utils.data import DataLoader
from shapely.affinity import rotate, translate
from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union

from prime_dataset import FloorplanDataset, floorplan_collate
from visualize import get_hard_color, visualize_prime


from utils import normalize_polygon, polygons_have_same_shape, check_fixed_const, check_preplaced_const, check_mib_const, check_boundary_const, check_clust_const
from cost import calculate_weighted_b2b_wirelength, calculate_weighted_p2b_wirelength

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



def estimate_cost(bdata: list, layout_index: int):
    """
    Estimate the cost of a layout by evaluating area and wirelength violations.

    Args:
        bdata (list): The block data containing polygon coordinates.
        layout_index (int): The index of the layout to evaluate.
    """
    # Read baseline design
    root = './'
    ds = FloorplanDataset(root)
    target_data = ds.__getitem__(layout_index)

    target_area_budgets = target_data['input'][0]
    target_b2b_edges = target_data['input'][1]
    target_p2b_edges = target_data['input'][2]
    target_pins_pos = target_data['input'][3]
    target_constraints = target_data['input'][4]
    target_metrics = target_data['label'][1]
    target_poly = target_data['label'][0]

    target_b2b_wl = target_metrics[-2].item()
    target_p2b_wl = target_metrics[-1].item()
    target_layout_area = target_metrics[0].item()

    if len(bdata) != len(target_poly):
        print('ERROR: incorrect number of polygons in the solution')
        exit()

    # Estimate area and validate target area budgets
    target_sol = [Polygon(targ_poly.tolist()) for targ_poly in target_poly]

    sol_area_budgets = torch.zeros_like(target_area_budgets)
    fp_sol = []
    W, H = 0, 0
    min_x, min_y, max_x, max_y = float('inf'), float('inf'), 0, 0

    for ind, elem in enumerate(bdata):
        poly_elem = Polygon(elem.tolist())
        sol_area_budgets[ind] = poly_elem.area
        fp_sol.append(poly_elem)

        min_x, min_y, max_x, max_y = poly_elem.bounds
        W = max(W, max_x)
        H = max(H, max_y)

    sol_area_cost = float(W * H)
    delta_budgets = sol_area_budgets - target_area_budgets
    area_viol = torch.nonzero(delta_budgets < 0).squeeze()

    # Estimate wirelength
    centroids = torch.tensor(
        [list(Polygon(fp_sol[i]).centroid.coords[0]) for i in range(len(fp_sol))],
        dtype=torch.float32,
    )

    sol_b2b_wl = calculate_weighted_manhattan_distance(
        centroids, target_b2b_edges
    )

    sol_p2b_wl = calculate_pin_wirelength(
        centroids, target_p2b_edges, target_pins_pos
    )



    # Estimate constraint cost
    fixed_const = target_constraints[:, 0]
    preplaced_const = target_constraints[:, 1]
    mib_const = target_constraints[:, 2]
    clust_const = target_constraints[:, 3]
    bound_const = target_constraints[:, 4]

    # Prepare the output dictionary
    results = {
        'placement_constraints': {
            'fixed': 0,
            'preplaced': 0,
            'mib': 0,
            'cluster': 0,
            'boundary': 0
        },
        'wl_difference': {
            'b2b': sol_b2b_wl - target_b2b_wl,
            'p2b': sol_p2b_wl - target_p2b_wl
        },
        'layout_area_difference': sol_area_cost - target_layout_area,
        'partition_indices_with_area_violations': area_viol.tolist()
    }

    results['placement_constraints']['fixed'] = check_fixed_const(
        torch.nonzero(fixed_const), fp_sol, target_sol
    )
    results['placement_constraints']['preplaced'] = check_preplaced_const(
        torch.nonzero(preplaced_const), fp_sol, target_sol
    )
    results['placement_constraints']['mib'] = check_mib_const(
        mib_const, fp_sol, target_sol
    )
    results['placement_constraints']['cluster'] = check_clust_const(
        clust_const, fp_sol, target_sol
    )
    results['placement_constraints']['boundary'] = check_boundary_const(
        bound_const, fp_sol, target_sol, W, H
    )

    return results