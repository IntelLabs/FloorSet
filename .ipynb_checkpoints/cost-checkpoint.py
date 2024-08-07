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