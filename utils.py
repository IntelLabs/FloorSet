import copy
import numpy as np
import torch
from torch.utils.data import DataLoader
from shapely.affinity import rotate, translate
from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union

from prime_dataset import FloorplanDataset, floorplan_collate
from visualize import get_hard_color, visualize_prime

def unpad_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """
    Remove padding from a tensor, returning only valid rows.

    Args:
        tensor (torch.Tensor): Input tensor with potential padding.

    Returns:
        torch.Tensor: Unpadded tensor containing valid rows.
    """
    mask = tensor != -1.0
    valid_rows_mask = mask.all(dim=1)
    unpadded_tensor = tensor[valid_rows_mask]
    return unpadded_tensor

def calculate_difference_with_tolerance(sol: float, target: float, tol: float = 0.01) -> float:
    """
    Calculate the difference between two values with a specified tolerance.

    Args:
        sol (float): The solution value.
        target (float): The target value.
        tol (float): The tolerance level.

    Returns:
        float: Adjusted difference considering the tolerance.
    """
    difference = sol - target
    return difference if abs(difference) >= tol else 0

def normalize_polygon(polygon: Polygon) -> Polygon:
    """
    Normalize a polygon by translating its bounding box to the origin and rotating it to align with axes.

    Args:
        polygon (Polygon): The input Shapely polygon to be normalized.

    Returns:
        Polygon: The normalized polygon.
    """
    bbox = polygon.minimum_rotated_rectangle
    bbox_coords = list(bbox.exterior.coords)[:-1]
    min_x = min(coord[0] for coord in bbox_coords)
    min_y = min(coord[1] for coord in bbox_coords)
    translated_polygon = translate(polygon, xoff=-min_x, yoff=-min_y)
    
    # Get the oriented bounding box again after translation
    bbox = translated_polygon.minimum_rotated_rectangle
    bbox_coords = list(bbox.exterior.coords)[:-1]
    angle = np.arctan2(bbox_coords[1][1] - bbox_coords[0][1], bbox_coords[1][0] - bbox_coords[0][0])
    
    aligned_polygon = rotate(translated_polygon, -np.degrees(angle), origin='centroid')
    return aligned_polygon

def normalize_centroid_based(polygon: Polygon) -> Polygon:
    """
    Normalize a polygon by centering its centroid at the origin and aligning the longest edge with the x-axis.

    Args:
        polygon (Polygon): The input Shapely polygon to be normalized.

    Returns:
        Polygon: The normalized polygon with centroid at the origin and aligned with the x-axis.
    """
    centroid = polygon.centroid
    translated = translate(polygon, -centroid.x, -centroid.y)
    
    coords = list(translated.exterior.coords[:-1])
    longest_edge = max(
        [(coords[i], coords[i + 1]) for i in range(len(coords) - 1)],
        key=lambda edge: np.linalg.norm(np.array(edge[1]) - np.array(edge[0]))
    )

    p1, p2 = longest_edge
    dx, dy = np.array(p2) - np.array(p1)
    angle = np.arctan2(dy, dx)
    aligned = rotate(translated, -np.degrees(angle), origin=(0, 0))
    return aligned

def polygons_have_same_shape(poly1: Polygon, poly2: Polygon, tolerance: float = 1e-3) -> bool:
    """
    Determine if two polygons have the same shape, disregarding location.

    Args:
        poly1 (Polygon): The first polygon for comparison.
        poly2 (Polygon): The second polygon for comparison.
        tolerance (float): The tolerance for area and equality comparison.

    Returns:
        bool: True if polygons have the same shape, False otherwise.
    """
    if not np.isclose(poly1.area, poly2.area, atol=tolerance):
        return False

    norm_poly1 = normalize_polygon(poly1)
    norm_poly2 = normalize_polygon(poly2)

    return norm_poly1.equals_exact(norm_poly2, tolerance)

def check_fixed_const(indices: torch.Tensor, pred_sol: list[Polygon], target_sol: list[Polygon]) -> int:
    """
    Check for violations of the fixed constraint by comparing predicted and target polygons.

    Args:
        indices (torch.Tensor): Indices to check for fixed constraints.
        pred_sol (list): Predicted solutions list containing polygons.
        target_sol (list): Target solutions list containing polygons.

    Returns:
        int: The count of violations found.
    """
    viol_count = sum(
        not polygons_have_same_shape(pred_sol[index], target_sol[index])
        for index in indices
    )
    return viol_count

def check_preplaced_const(indices: torch.Tensor, pred_sol: list[Polygon], target_sol: list[Polygon], threshold: float = 0.001) -> int:
    """
    Check for violations of the preplaced constraint by evaluating intersection areas.

    Args:
        indices (torch.Tensor): Indices to check for preplaced constraints.
        pred_sol (list): Predicted solutions list containing polygons.
        target_sol (list): Target solutions list containing polygons.
        threshold (float): The threshold for intersection area comparison.

    Returns:
        int: The count of violations found.
    """
    viol_count = sum(
        polygon1.intersection(polygon2).area + threshold <= polygon1.area
        for index in indices
        for polygon1, polygon2 in [(pred_sol[index], target_sol[index])]
    )
    return viol_count

def check_mib_const(indices: torch.Tensor, pred_sol: list[Polygon], target_sol: list[Polygon]) -> int:
    """
    Check for violations of the MIB constraint by comparing polygons in groups.

    Args:
        indices (torch.Tensor): Indices representing MIB groups.
        pred_sol (list): Predicted solutions list containing polygons.
        target_sol (list): Target solutions list containing polygons.

    Returns:
        int: The count of violations found.
    """
    viol_count = 0
    mib_groups = int(max(indices).item())
    if mib_groups == 0:
        return viol_count

    for index in range(1, mib_groups + 1):
        shared_poly_ind = torch.where(indices == index)[0].tolist()
        polygon1 = pred_sol[shared_poly_ind[0]]
        viol_count += sum(
            not polygons_have_same_shape(polygon1, pred_sol[sind])
            for sind in shared_poly_ind[1:]
        )

    return viol_count

def check_clust_const(indices: torch.Tensor, pred_sol: list[Polygon], target_sol: list[Polygon]) -> int:
    """
    Check for violations of the clustering constraint by evaluating union of polygons.

    Args:
        indices (torch.Tensor): Indices representing clustering groups.
        pred_sol (list): Predicted solutions list containing polygons.
        target_sol (list): Target solutions list containing polygons.

    Returns:
        int: The count of violations found.
    """
    viol_count = 0
    clust_groups = int(max(indices).item())
    if clust_groups == 0:
        return viol_count

    for index in range(1, clust_groups + 1):
        shared_poly_ind = torch.where(indices == index)[0].tolist()
        clust_poly = [pred_sol[sind] for sind in shared_poly_ind]
        union = unary_union(clust_poly)

        viol_count += len(union.geoms) if union.geom_type == 'MultiPolygon' else 0

    return viol_count

def check_boundary_const(bound_const: torch.Tensor, pred_sol: list[Polygon], target_sol: list[Polygon], W: int, H: int) -> int:
    """
    Check for violations of the boundary constraint.

    Args:
        bound_const (torch.Tensor): Boundary constraint tensor.
        pred_sol (list): Predicted solutions list containing polygons.
        target_sol (list): Target solutions list containing polygons.
        W (int): Width of the bounding box.
        H (int): Height of the bounding box.

    Returns:
        int: The count of violations found.
    """
    nz_indices = torch.nonzero(bound_const).numpy().flatten().tolist()
    nz_values = bound_const.numpy().astype(np.int32).flatten().tolist()
    viol_count = 0

    edges = {
        5: [LineString([(0, H), (W, H)]), LineString([(0, 0), (0, H)])],   # Top-left
        6: [LineString([(0, H), (W, H)]), LineString([(W, 0), (W, H)])],   # Top-right
        9: [LineString([(0, 0), (W, 0)]), LineString([(0, 0), (0, H)])],   # Bottom-left
        10: [LineString([(0, 0), (W, 0)]), LineString([(W, 0), (W, H)])],  # Bottom-right
        1: [LineString([(0, 0), (0, H)])],                                 # Left
        2: [LineString([(W, 0), (W, H)])],                                 # Right
        4: [LineString([(0, H), (W, H)])],                                 # Top
        8: [LineString([(0, 0), (W, 0)])],                                 # Bottom
    }

    for index in nz_indices:
        polygon = pred_sol[index]
        edges_to_check = edges.get(nz_values[index], [])

        if not all(polygon.intersects(edge) for edge in edges_to_check):
            viol_count += 1
            print(polygon, nz_values[index])

    return viol_count
