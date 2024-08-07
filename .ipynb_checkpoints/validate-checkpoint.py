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

    fixed_viol = check_fixed_const(torch.nonzero(fixed_const), fp_sol, target_sol)
    preplaced_viol = check_preplaced_const(torch.nonzero(preplaced_const), fp_sol, target_sol)
    mib_viol = check_mib_const(mib_const, fp_sol, target_sol)
    clust_viol = check_clust_const(clust_const, fp_sol, target_sol)
    boundary_viol = check_boundary_const(bound_const, fp_sol, target_sol, W, H)

    
    print('Placement constraints: viol count (fixed/preplaced/mib/cluster/boundary)::', fixed_viol, preplaced_viol, mib_viol, clust_viol, boundary_viol)
    print('WL difference (B2B and P2B):', sol_b2b_wl - target_b2b_wl, sol_p2b_wl - target_p2b_wl)
    print('Layout area difference:', sol_area_cost - target_layout_area)
    print('Partition indices with target-area-budget violations:', area_viol.tolist())
    

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
