import copy
import numpy as np
import torch
from torch.utils.data import DataLoader
from shapely.affinity import rotate, translate
from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union


def calculate_weighted_b2b_wirelength(
    centroids: torch.Tensor, b2b_edges: torch.Tensor
) -> float:
    """
    Calculate the weighted HPWL for block-to-block edges.

    Args:
        centroids (torch.Tensor): The centroids of polygons.
        b2b_edges (torch.Tensor): The block-to-block edges tensor.

    Returns:
        float: The weighted Manhattan distance.
    """
    b2b_indices_0 = b2b_edges[:, 0].long()
    b2b_indices_1 = b2b_edges[:, 1].long()

    b2b_weights = (
        b2b_edges[:, 2]
        if b2b_edges.shape[1] > 2
        else torch.ones(b2b_edges.shape[0])
    )

    diff_x_b2b = torch.abs(centroids[b2b_indices_1, 0] - centroids[b2b_indices_0, 0])
    diff_y_b2b = torch.abs(centroids[b2b_indices_1, 1] - centroids[b2b_indices_0, 1])

    return torch.sum((diff_x_b2b + diff_y_b2b) * b2b_weights).item()


def calculate_weighted_p2b_wirelength(
    centroids: torch.Tensor, p2b_edges: torch.Tensor, pins_pos: torch.Tensor
) -> float:
    """
    Calculate weighted HPWL for pin-to-block edges.

    Args:
        centroids (torch.Tensor): The centroids of polygons.
        p2b_edges (torch.Tensor): The pin-to-block edges tensor.
        pins_pos (torch.Tensor): The positions of pins.

    Returns:
        float: The weighted Manhattan distance for pin-to-block edges.
    """
    p2b_indices_0 = p2b_edges[:, 0].long()
    p2b_indices_1 = p2b_edges[:, 1].long()

    p2b_weights = (
        p2b_edges[:, 2]
        if p2b_edges.shape[1] > 2
        else torch.ones(p2b_edges.shape[0])
    )

    px_py = pins_pos[p2b_indices_0]
    px, py = px_py[:, 0], px_py[:, 1]

    diff_x_p2b = torch.abs(centroids[p2b_indices_1, 0] - px)
    diff_y_p2b = torch.abs(centroids[p2b_indices_1, 1] - py)

    return torch.sum((diff_x_p2b + diff_y_p2b) * p2b_weights).item()

def estimate_cost(bdata, target_area_budgets, target_b2b_edges, target_p2b_edges, target_pins_pos, target_constraints_pad, target_poly, target_metrics):
    """
    Estimate the cost of a layout by evaluating area and wirelength violations.

    Args:
        bdata (list): The block data containing polygon coordinates.
        layout_index (int): The index of the layout to evaluate.
    """


    target_b2b_wl = target_metrics[-2].item()
    target_p2b_wl = target_metrics[-1].item()
    target_layout_area = target_metrics[0].item()

    if len(bdata) != len(target_poly):
        print('ERROR: incorrect number of polygons in the solution')
        exit()

    # Estimate area and validate target area budgets
    target_sol = []
    for tpoly in target_poly.tolist():
        unpadded_poly = Polygon([point for point in tpoly if point != [-1.0, -1.0]])
        target_sol.append(unpadded_poly)
    
    ##print('Unpadded target solution::', target_sol)
    ##target_sol = [Polygon(targ_poly.tolist()) for targ_poly in target_poly]

    sol_area_budgets = torch.zeros_like(target_area_budgets)
    fp_sol = []
    W, H = 0, 0
    min_x, min_y, max_x, max_y = float('inf'), float('inf'), 0, 0

    for ind, elem in enumerate(bdata):
        poly_elem1 = elem.tolist()
        poly_elem2 = [point for point in poly_elem1 if point != [-1.0, -1.0]]
        poly_elem = Polygon(poly_elem2)
        sol_area_budgets[ind] = poly_elem.area
        fp_sol.append(poly_elem)

        min_x, min_y, max_x, max_y = poly_elem.bounds
        W = max(W, max_x)
        H = max(H, max_y)

    sol_area_cost = float(W * H)
    delta_budgets = sol_area_budgets - target_area_budgets
    area_viol = torch.nonzero(delta_budgets < 0).squeeze()


    ###print('Unpadded predicted sol::', fp_sol)
    # Estimate wirelength
    centroids = unpad_tensor(torch.tensor(
        [list(Polygon(fp_sol[i]).centroid.coords[0]) for i in range(len(fp_sol))],
        dtype=torch.float32,
    ))

    

    
    sol_b2b_wl = calculate_weighted_b2b_wirelength(
        centroids, unpad_tensor(target_b2b_edges)
    )

    sol_p2b_wl = calculate_weighted_p2b_wirelength(
        centroids, unpad_tensor(target_p2b_edges), unpad_tensor(target_pins_pos)
    )


    target_constraints = unpad_tensor(target_constraints_pad)
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
            'b2b': calculate_difference_with_tolerance(sol_b2b_wl, target_b2b_wl),
            'p2b': calculate_difference_with_tolerance(sol_p2b_wl, target_p2b_wl)
        },
        'layout_area_difference': calculate_difference_with_tolerance(sol_area_cost, target_layout_area),
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