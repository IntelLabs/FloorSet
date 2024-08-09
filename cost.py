# Import necessary libraries and modules
import torch
from torch.utils.data import DataLoader
from shapely.geometry import Polygon
import copy
import numpy as np

# Custom utility functions
from utils import (
    unpad_tensor,
    calculate_difference_with_tolerance,
    check_fixed_const,
    check_preplaced_const,
    check_mib_const,
    check_boundary_const,
    check_clust_const
)


def calculate_weighted_b2b_wirelength(centroids: torch.Tensor, b2b_edges: torch.Tensor) -> float:
    """
    Calculate the weighted Half-Perimeter Wire Length (HPWL) for block-to-block edges.

    Args:
        centroids (torch.Tensor): The centroids of polygons.
        b2b_edges (torch.Tensor): The block-to-block edges tensor.

    Returns:
        float: The weighted Manhattan distance.
    """
    # Extract indices and weights
    b2b_indices_0 = b2b_edges[:, 0].long()
    b2b_indices_1 = b2b_edges[:, 1].long()
    b2b_weights = b2b_edges[:, 2] if b2b_edges.shape[1] > 2 else torch.ones(b2b_edges.shape[0])

    # Calculate differences in x and y directions
    diff_x_b2b = torch.abs(centroids[b2b_indices_1, 0] - centroids[b2b_indices_0, 0])
    diff_y_b2b = torch.abs(centroids[b2b_indices_1, 1] - centroids[b2b_indices_0, 1])

    # Return the weighted sum of differences
    return torch.sum((diff_x_b2b + diff_y_b2b) * b2b_weights).item()


def calculate_weighted_p2b_wirelength(centroids: torch.Tensor, p2b_edges: torch.Tensor, pins_pos: torch.Tensor) -> float:
    """
    Calculate weighted Half-Perimeter Wire Length (HPWL) for pin-to-block edges.

    Args:
        centroids (torch.Tensor): The centroids of polygons.
        p2b_edges (torch.Tensor): The pin-to-block edges tensor.
        pins_pos (torch.Tensor): The positions of pins.

    Returns:
        float: The weighted Manhattan distance for pin-to-block edges.
    """
    # Extract indices and weights
    p2b_indices_0 = p2b_edges[:, 0].long()
    p2b_indices_1 = p2b_edges[:, 1].long()
    p2b_weights = p2b_edges[:, 2] if p2b_edges.shape[1] > 2 else torch.ones(p2b_edges.shape[0])

    # Extract pin positions
    px_py = pins_pos[p2b_indices_0]
    px, py = px_py[:, 0], px_py[:, 1]

    # Calculate differences in x and y directions
    diff_x_p2b = torch.abs(centroids[p2b_indices_1, 0] - px)
    diff_y_p2b = torch.abs(centroids[p2b_indices_1, 1] - py)

    # Return the weighted sum of differences
    return torch.sum((diff_x_p2b + diff_y_p2b) * p2b_weights).item()


def estimate_cost(bdata, target_area_budgets, target_b2b_edges, target_p2b_edges, target_pins_pos, target_constraints_pad, target_poly, target_metrics):
    """
    Estimate the cost of a layout by evaluating area and wire length violations.

    Args:
        bdata (list of tensors): The block data containing polygon coordinates from predicted floorplan solution.
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

        
        target_area_budgets (torch.Tensor): The target area budgets for each polygon.
        target_b2b_edges (torch.Tensor): The target block-to-block edges.
        target_p2b_edges (torch.Tensor): The target pin-to-block edges.
        target_pins_pos (torch.Tensor): The target pin positions.
        target_constraints_pad (torch.Tensor): The target constraints padded tensor.
        target_poly (torch.Tensor): The target polygons.
        target_metrics (torch.Tensor): The target metrics.

    Returns:
        dict: A dictionary containing results for placement constraints, wire length difference, layout area difference, and partition indices with area violations.
    """

    # Extract target metrics
    target_b2b_wl = target_metrics[-2].item()
    target_p2b_wl = target_metrics[-1].item()
    target_layout_area = target_metrics[0].item()

    # Validate the number of polygons
    if len(bdata) != len(target_poly):
        print('ERROR: incorrect number of polygons in the solution')
        exit()

    # Prepare the target solution by removing padding
    target_sol = [Polygon([point for point in tpoly if point != [-1.0, -1.0]]) for tpoly in target_poly.tolist()]

    # Initialize variables for area calculation
    sol_area_budgets = torch.zeros_like(target_area_budgets)
    fp_sol = []
    W, H = 0, 0

    # Iterate over block data to calculate area budgets and bounds
    for ind, elem in enumerate(bdata):
        poly_elem = Polygon([point for point in elem.tolist() if point != [-1.0, -1.0]])
        sol_area_budgets[ind] = poly_elem.area
        fp_sol.append(poly_elem)

        min_x, min_y, max_x, max_y = poly_elem.bounds
        W = max(W, max_x)
        H = max(H, max_y)

    # Calculate area cost and violations
    sol_area_cost = float(W * H)
    delta_budgets = sol_area_budgets - target_area_budgets
    area_viol = torch.nonzero(delta_budgets < 0).squeeze()

    # Calculate centroids of the solution polygons
    centroids = unpad_tensor(torch.tensor(
        [list(Polygon(fp_sol[i]).centroid.coords[0]) for i in range(len(fp_sol))],
        dtype=torch.float32,
    ))

    # Calculate wire lengths
    sol_b2b_wl = calculate_weighted_b2b_wirelength(centroids, unpad_tensor(target_b2b_edges))
    sol_p2b_wl = calculate_weighted_p2b_wirelength(centroids, unpad_tensor(target_p2b_edges), unpad_tensor(target_pins_pos))

    # Unpad target constraints for further processing
    target_constraints = unpad_tensor(target_constraints_pad)

    # Extract individual constraints
    fixed_const = target_constraints[:, 0]
    preplaced_const = target_constraints[:, 1]
    mib_const = target_constraints[:, 2]
    clust_const = target_constraints[:, 3]
    bound_const = target_constraints[:, 4]

    # Prepare the output dictionary with detailed constraint checks
    results = {
        'placement_constraints': {
            'fixed': check_fixed_const(torch.nonzero(fixed_const), fp_sol, target_sol),
            'preplaced': check_preplaced_const(torch.nonzero(preplaced_const), fp_sol, target_sol),
            'mib': check_mib_const(mib_const, fp_sol, target_sol),
            'cluster': check_clust_const(clust_const, fp_sol, target_sol),
            'boundary': check_boundary_const(bound_const, fp_sol, target_sol, W, H)
        },
        'wl_difference': {
            'b2b': calculate_difference_with_tolerance(sol_b2b_wl, target_b2b_wl),
            'p2b': calculate_difference_with_tolerance(sol_p2b_wl, target_p2b_wl)
        },
        'layout_area_difference': calculate_difference_with_tolerance(sol_area_cost, target_layout_area),
        'partition_indices_with_area_violations': area_viol.tolist()
    }

    return results