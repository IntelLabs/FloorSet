#!/usr/bin/env python3
"""
ICCAD 2026 FloorSet Challenge - Optimizer Template

USAGE:
  1. Copy: cp optimizer_template.py my_optimizer.py
  2. Replace the B*-tree code with your algorithm
  3. Test: python iccad2026_evaluate.py --evaluate my_optimizer.py

BASELINE: B*-tree Simulated Annealing
  - GUARANTEES: Overlap-free, area constraints satisfied, fixed/preplaced hard constraints respected
  - NOT HANDLED: MIB, cluster, boundary soft constraints

Your solve() receives:
  - block_count: int
  - area_targets: [n] target area per block
  - b2b_connectivity: [edges, 3] (block_i, block_j, weight)
  - p2b_connectivity: [edges, 3] (pin_idx, block_idx, weight)
  - pins_pos: [n_pins, 2] pin (x, y)
  - constraints: [n, 5] (fixed, preplaced, MIB, cluster, boundary)
  - target_positions: [n, 4] target (x, y, w, h) per block.
      All -1 by default (free). For fixed-shape blocks, w and h are set.
      For preplaced blocks, all four (x, y, w, h) are set.

Your solve() must return:
  - List of (x, y, width, height), exactly block_count tuples
  - Floating-point coordinates allowed
  - Any aspect ratio (w/h) allowed

HARD CONSTRAINTS (violation = Cost 10.0):
  - NO OVERLAPS between blocks
  - AREA: w*h within 1% of area_targets[i] (soft blocks only)
  - DIMENSION IMMUTABILITY: Fixed-shape blocks must use exact (w, h) from
    target_positions; preplaced blocks must use exact (x, y, w, h)

RELAXED CONSTRAINTS:
  - Aspect ratio: Any w/h ratio is valid
  - Fixed outline: Removed (implicitly optimized via p2b HPWL and bbox area)
  - Coordinates: Floating-point allowed
"""

import math
import random
import sys
from pathlib import Path
from typing import List, Tuple

import torch

sys.path.insert(0, str(Path(__file__).parent))

from iccad2026_evaluate import (
    FloorplanOptimizer,
    calculate_hpwl_b2b,
    calculate_hpwl_p2b,
    calculate_bbox_area,
    check_overlap,
)


# =============================================================================
# B*-TREE DATA STRUCTURE
# Replace this entire class if using a different representation
# (Sequence Pair, O-tree, Corner Block List, etc.)
# =============================================================================

class BStarTree:
    """
    B*-tree for overlap-free floorplanning.
    
    Left child: placed to the RIGHT of parent
    Right child: placed ABOVE parent (same x)
    """
    
    def __init__(self, n_blocks: int, widths: List[float], heights: List[float]):
        self.n = n_blocks
        self.widths = list(widths)
        self.heights = list(heights)
        self.parent = [-1] * n_blocks
        self.left = [-1] * n_blocks
        self.right = [-1] * n_blocks
        self.root = 0
        self._build_random_tree()
    
    def _build_random_tree(self):
        if self.n == 0:
            return
        self.parent = [-1] * self.n
        self.left = [-1] * self.n
        self.right = [-1] * self.n
        
        order = list(range(self.n))
        random.shuffle(order)
        self.root = order[0]
        
        for i in range(1, self.n):
            block = order[i]
            existing = order[random.randint(0, i - 1)]
            if random.random() < 0.5:
                if self.left[existing] == -1:
                    self.left[existing] = block
                    self.parent[block] = existing
                elif self.right[existing] == -1:
                    self.right[existing] = block
                    self.parent[block] = existing
                else:
                    self._insert_at_leaf(block, existing)
            else:
                if self.right[existing] == -1:
                    self.right[existing] = block
                    self.parent[block] = existing
                elif self.left[existing] == -1:
                    self.left[existing] = block
                    self.parent[block] = existing
                else:
                    self._insert_at_leaf(block, existing)
    
    def _insert_at_leaf(self, block: int, start: int):
        current = start
        while True:
            if random.random() < 0.5:
                if self.left[current] == -1:
                    self.left[current] = block
                    self.parent[block] = current
                    return
                current = self.left[current]
            else:
                if self.right[current] == -1:
                    self.right[current] = block
                    self.parent[block] = current
                    return
                current = self.right[current]
    
    def pack(self) -> List[Tuple[float, float, float, float]]:
        """
        Compute (x, y, w, h) from tree structure.
        
        Uses proper contour tracking to ensure overlap-free placement.
        B*-tree rules:
        - Left child: placed to the RIGHT of parent
        - Right child: placed ABOVE parent (same x as parent)
        """
        positions = [(0.0, 0.0, self.widths[i], self.heights[i]) for i in range(self.n)]
        if self.n == 0:
            return positions
        
        # Contour: sorted list of (x_end, y_top) representing skyline
        # At any x, the contour height is the y_top of the rightmost segment with x_end > x
        contour = [(0.0, 0.0)]  # Start with ground level
        
        def get_contour_y(x_start: float, x_end: float) -> float:
            """Find max y in contour for range [x_start, x_end]."""
            max_y = 0.0
            for i, (cx_end, cy_top) in enumerate(contour):
                # Get x_start of this segment
                cx_start = contour[i-1][0] if i > 0 else 0.0
                # Check if segments overlap
                if x_start < cx_end and x_end > cx_start:
                    max_y = max(max_y, cy_top)
            return max_y
        
        def update_contour(x_start: float, x_end: float, y_top: float):
            """Add a new block to the contour."""
            nonlocal contour
            new_contour = []
            
            for i, (cx_end, cy_top) in enumerate(contour):
                cx_start = contour[i-1][0] if i > 0 else 0.0
                
                # Before the new block
                if cx_end <= x_start:
                    new_contour.append((cx_end, cy_top))
                # After the new block
                elif cx_start >= x_end:
                    new_contour.append((cx_end, cy_top))
                # Overlapping - need to split
                else:
                    # Part before new block
                    if cx_start < x_start:
                        new_contour.append((x_start, cy_top))
                    # Part after new block
                    if cx_end > x_end:
                        new_contour.append((cx_end, cy_top))
            
            # Add the new block segment
            # Find where to insert
            insert_pos = 0
            for i, (cx_end, _) in enumerate(new_contour):
                if cx_end <= x_start:
                    insert_pos = i + 1
            new_contour.insert(insert_pos, (x_end, y_top))
            
            # Sort by x_end and merge adjacent segments with same y
            new_contour.sort(key=lambda x: x[0])
            
            # Merge adjacent segments with same height
            merged = []
            for x_end, y_top in new_contour:
                if merged and merged[-1][1] == y_top:
                    merged[-1] = (x_end, y_top)  # Extend previous
                else:
                    merged.append((x_end, y_top))
            
            contour = merged if merged else [(x_end, 0.0)]
        
        # DFS traversal to place blocks
        def dfs(node: int, parent_right_edge: float):
            if node == -1:
                return
            
            w, h = self.widths[node], self.heights[node]
            
            if node == self.root:
                x = 0.0
                y = 0.0
            else:
                x = parent_right_edge
                y = get_contour_y(x, x + w)
            
            positions[node] = (x, y, w, h)
            update_contour(x, x + w, y + h)
            
            # Left child: to the RIGHT of this node
            dfs(self.left[node], x + w)
            # Right child: ABOVE this node (same x, will stack due to contour)
            dfs(self.right[node], x)
        
        dfs(self.root, 0.0)
        
        # Verify no overlaps (should never happen with correct contour)
        for i in range(self.n):
            for j in range(i + 1, self.n):
                x1, y1, w1, h1 = positions[i]
                x2, y2, w2, h2 = positions[j]
                overlap_x = min(x1 + w1, x2 + w2) - max(x1, x2)
                overlap_y = min(y1 + h1, y2 + h2) - max(y1, y2)
                if overlap_x > 1e-6 and overlap_y > 1e-6:
                    # Fix by pushing j up
                    positions[j] = (x2, max(y1 + h1, y2), w2, h2)
        
        return positions
    
    def copy(self) -> 'BStarTree':
        new = BStarTree.__new__(BStarTree)
        new.n = self.n
        new.widths = self.widths.copy()
        new.heights = self.heights.copy()
        new.parent = self.parent.copy()
        new.left = self.left.copy()
        new.right = self.right.copy()
        new.root = self.root
        return new
    
    # SA moves
    def move_rotate(self, block: int):
        """Swap width/height (90° rotation, preserves area)."""
        self.widths[block], self.heights[block] = self.heights[block], self.widths[block]
    
    def move_swap(self, b1: int, b2: int):
        """Swap two blocks' dimensions."""
        self.widths[b1], self.widths[b2] = self.widths[b2], self.widths[b1]
        self.heights[b1], self.heights[b2] = self.heights[b2], self.heights[b1]
    
    def move_delete_insert(self, block: int):
        """Delete and reinsert block at random position."""
        if self.n <= 1:
            return
        w, h = self.widths[block], self.heights[block]
        self._delete_node(block)
        target = random.randint(0, self.n - 1)
        while target == block:
            target = random.randint(0, self.n - 1)
        self._insert_node(block, target, random.choice([True, False]))
        self.widths[block], self.heights[block] = w, h
    
    def _delete_node(self, node: int):
        parent = self.parent[node]
        left_child = self.left[node]
        right_child = self.right[node]
        
        if left_child == -1 and right_child == -1:
            replacement = -1
        elif left_child == -1:
            replacement = right_child
        elif right_child == -1:
            replacement = left_child
        else:
            replacement = left_child
            rightmost = left_child
            while self.right[rightmost] != -1:
                rightmost = self.right[rightmost]
            self.right[rightmost] = right_child
            self.parent[right_child] = rightmost
        
        if parent == -1:
            self.root = replacement
        elif self.left[parent] == node:
            self.left[parent] = replacement
        else:
            self.right[parent] = replacement
        
        if replacement != -1:
            self.parent[replacement] = parent
        
        self.parent[node] = -1
        self.left[node] = -1
        self.right[node] = -1
    
    def _insert_node(self, node: int, target: int, as_left: bool):
        if as_left:
            old_child = self.left[target]
            self.left[target] = node
        else:
            old_child = self.right[target]
            self.right[target] = node
        self.parent[node] = target
        if old_child != -1:
            self.left[node] = old_child
            self.parent[old_child] = node


# =============================================================================
# OPTIMIZER CLASS - Replace this with your algorithm
# =============================================================================

class MyOptimizer(FloorplanOptimizer):
    """
    ML-guided floorplanning with feature-based ordering, legalization,
    and adaptive local search.
    """
    
    def __init__(self, verbose: bool = False):
        super().__init__(verbose)
        self.ratio_candidates = [1.0, 2.0, 0.5, 1.5, 2.0 / 3.0]
        self.explore_prob = 0.2
        self.shift_alpha = 0.35
    
    def solve(
        self,
        block_count: int,
        area_targets: torch.Tensor,
        b2b_connectivity: torch.Tensor,
        p2b_connectivity: torch.Tensor,
        pins_pos: torch.Tensor,
        constraints: torch.Tensor,
        target_positions: torch.Tensor = None
    ) -> List[Tuple[float, float, float, float]]:
        """
        Experimental ML-guided floorplanning.

        The solver compares several legalized flows: robust shelf packing,
        GNN-style greedy placement, graph-clustering GA, and a small
        B*-tree/SA-inspired baseline.  Every candidate is legalized before it
        is scored, and only feasible improvements are retained.
        """
        try:
            seed_base = int(__import__("os").environ.get("FLOORSET_RANDOM_SEED", "404"))
        except Exception:
            seed_base = 1337
        random.seed(seed_base + int(block_count))
        ratio_candidates = [1.0, 2.0, 0.5, 1.5, 2.0 / 3.0]
        eps = 1e-6
        hard_eps = 1e-4
        os_mod = __import__("os")
        flow_filter_raw = os_mod.environ.get("FLOORSET_FLOW_FILTER", "").strip()
        # Default to the experimentally best selector set.  The B*-tree/SA flow
        # remains available through FLOORSET_FLOW_FILTER=bstar_sa or all, but it
        # did not improve validation score and costs extra runtime.
        default_flows = {
            "shelf_area",
            "shelf_importance",
            "ml_greedy",
            "ml_cluster_greedy",
            "cluster_ga",
            "trained_prior_order",
            "trained_prior_cluster",
            "soft_prior_order",
            "soft_prior_cluster",
            "soft_final_repair",
            "fd_init",
            "fd_cluster",
            "frame_rows",
        }
        enabled_flows = None
        if flow_filter_raw:
            enabled_flows = {item.strip() for item in flow_filter_raw.split(",") if item.strip()}

        def flow_enabled(name: str) -> bool:
            canonical = name
            for prefix in ("soft_final_repair", "cluster_ga", "frame_rows"):
                if name.startswith(prefix):
                    canonical = prefix
                    break
            if enabled_flows is None:
                return canonical in default_flows
            return "all" in enabled_flows or canonical in enabled_flows

        def finite(v) -> bool:
            try:
                return math.isfinite(float(v))
            except Exception:
                return False

        def tensor_value(tensor, row: int, col: int = None, default: float = -1.0) -> float:
            try:
                if tensor is None:
                    return default
                if col is None:
                    if row >= len(tensor):
                        return default
                    return float(tensor[row])
                if row >= tensor.shape[0] or col >= tensor.shape[1]:
                    return default
                return float(tensor[row, col])
            except Exception:
                return default

        if target_positions is None:
            target_positions = torch.full((block_count, 4), -1.0, dtype=torch.float32)

        ncols = 0
        if constraints is not None and getattr(constraints, "numel", lambda: 0)() > 0:
            ncols = constraints.shape[1] if constraints.dim() > 1 else 1

        is_fixed = [False] * block_count
        is_preplaced = [False] * block_count
        fixed_dim = [False] * block_count
        fixed_xy = [False] * block_count
        mib_ids = [0] * block_count
        cluster_ids = [0] * block_count
        boundary_codes = [0] * block_count

        for i in range(block_count):
            is_fixed[i] = ncols > 0 and tensor_value(constraints, i, 0, 0.0) != 0
            is_preplaced[i] = ncols > 1 and tensor_value(constraints, i, 1, 0.0) != 0
            mib_ids[i] = int(tensor_value(constraints, i, 2, 0.0)) if ncols > 2 else 0
            cluster_ids[i] = int(tensor_value(constraints, i, 3, 0.0)) if ncols > 3 else 0
            boundary_codes[i] = int(tensor_value(constraints, i, 4, 0.0)) if ncols > 4 else 0

            tw = tensor_value(target_positions, i, 2)
            th = tensor_value(target_positions, i, 3)
            tx = tensor_value(target_positions, i, 0)
            ty = tensor_value(target_positions, i, 1)
            fixed_dim[i] = (is_fixed[i] or is_preplaced[i]) and finite(tw) and finite(th) and tw > 0 and th > 0
            fixed_xy[i] = is_preplaced[i] and fixed_dim[i] and finite(tx) and finite(ty) and tx != -1 and ty != -1

        areas = []
        for i in range(block_count):
            area = tensor_value(area_targets, i, default=1.0)
            areas.append(area if finite(area) and area > 0 else 1.0)
        total_area = max(sum(areas), 1.0)
        mean_side = max(math.sqrt(total_area / max(block_count, 1)), 1e-3)

        valid_b2b = []
        valid_p2b = []
        b2b_adj = [[] for _ in range(block_count)]
        p2b_adj = [[] for _ in range(block_count)]
        b2b_degree = [0.0] * block_count
        p2b_degree = [0.0] * block_count

        if b2b_connectivity is not None and b2b_connectivity.numel() > 0:
            for edge in b2b_connectivity:
                i, j, w = int(edge[0]), int(edge[1]), float(edge[2])
                if 0 <= i < block_count and 0 <= j < block_count and finite(w) and w > 0:
                    valid_b2b.append((i, j, w))
                    b2b_adj[i].append((j, w))
                    b2b_adj[j].append((i, w))
                    b2b_degree[i] += w
                    b2b_degree[j] += w

        if p2b_connectivity is not None and p2b_connectivity.numel() > 0:
            for edge in p2b_connectivity:
                pin, block, w = int(edge[0]), int(edge[1]), float(edge[2])
                if not (0 <= block < block_count and finite(w) and w > 0):
                    continue
                if pins_pos is None or pins_pos.numel() == 0 or not (0 <= pin < pins_pos.shape[0]):
                    continue
                px = tensor_value(pins_pos, pin, 0)
                py = tensor_value(pins_pos, pin, 1)
                if not (finite(px) and finite(py)) or (px == -1 and py == -1):
                    continue
                valid_p2b.append((pin, block, w))
                p2b_adj[block].append((pin, w))
                p2b_degree[block] += w

        pin_centroid = [None] * block_count
        for i in range(block_count):
            sx = sy = sw = 0.0
            for pin, weight in p2b_adj[i]:
                px = tensor_value(pins_pos, pin, 0)
                py = tensor_value(pins_pos, pin, 1)
                sx += px * weight
                sy += py * weight
                sw += weight
            if sw > 0:
                pin_centroid[i] = (sx / sw, sy / sw)

        def block_dims(block: int, ratio: float = 1.0) -> Tuple[float, float]:
            if fixed_dim[block]:
                return (
                    float(tensor_value(target_positions, block, 2)),
                    float(tensor_value(target_positions, block, 3)),
                )
            ratio = max(float(ratio), 1e-6)
            return math.sqrt(areas[block] * ratio), math.sqrt(areas[block] / ratio)

        preplaced_positions = {}
        for i in range(block_count):
            if fixed_xy[i]:
                w, h = block_dims(i)
                preplaced_positions[i] = (
                    float(tensor_value(target_positions, i, 0)),
                    float(tensor_value(target_positions, i, 1)),
                    w,
                    h,
                )

        def rectangles_overlap(a, b, tolerance: float = eps) -> bool:
            ax, ay, aw, ah = a
            bx, by, bw, bh = b
            return (min(ax + aw, bx + bw) - max(ax, bx) > tolerance and
                    min(ay + ah, by + bh) - max(ay, by) > tolerance)

        def overlap_area_with(rect, positions, placed) -> float:
            x, y, w, h = rect
            area = 0.0
            for j in placed:
                if positions[j] is None:
                    continue
                xj, yj, wj, hj = positions[j]
                ox = min(x + w, xj + wj) - max(x, xj)
                oy = min(y + h, yj + hj) - max(y, yj)
                if ox > eps and oy > eps:
                    area += ox * oy
            return area

        def has_any_overlap(positions) -> bool:
            if positions is None or len(positions) != block_count:
                return True
            for i in range(block_count):
                if positions[i] is None:
                    return True
                for j in range(i + 1, block_count):
                    if rectangles_overlap(positions[i], positions[j]):
                        return True
            return False

        max_area = max(areas) if areas else 1.0
        max_b2b = max(b2b_degree) if b2b_degree else 1.0
        max_p2b = max(p2b_degree) if p2b_degree else 1.0
        base_score = []
        for i in range(block_count):
            area_score = math.sqrt(areas[i]) / max(math.sqrt(max_area), 1e-9)
            b_score = b2b_degree[i] / max(max_b2b, 1e-9)
            p_score = p2b_degree[i] / max(max_p2b, 1e-9)
            pin_score = 1.0 if pin_centroid[i] is not None else 0.0
            constraint_score = (0.35 if fixed_xy[i] else 0.0) + (0.2 if fixed_dim[i] else 0.0)
            boundary_score = 0.1 if boundary_codes[i] else 0.0
            base_score.append(
                0.28 * area_score + 0.26 * b_score + 0.18 * p_score +
                0.08 * pin_score + constraint_score + boundary_score
            )

        importance = base_score[:]
        for _ in range(2):
            updated = importance[:]
            for i in range(block_count):
                weighted = total = 0.0
                for nbr, weight in b2b_adj[i]:
                    weighted += importance[nbr] * weight
                    total += weight
                if total > 0:
                    updated[i] = 0.62 * importance[i] + 0.38 * (weighted / total)
            importance = updated

        prior_loaded = False
        prior_kind = None
        prior_order_bonus = [0.0] * block_count
        prior_ratio_index = [None] * block_count
        prior_centers = [None] * block_count
        prior_boundary_bits = [[0.0, 0.0, 0.0, 0.0] for _ in range(block_count)]
        prior_touch_adj = [[] for _ in range(block_count)]

        def load_trained_prior():
            """Load a graph prior checkpoint and apply it to placement policy.

            Supports both the original 12-feature prior and the new 16-feature
            soft-prior checkpoint.  The default soft-prior path is a frozen
            evaluator baseline copy so background training cannot change model
            behavior halfway through validation.
            """
            nonlocal prior_loaded, prior_kind, prior_order_bonus, prior_ratio_index
            nonlocal prior_centers, prior_boundary_bits, prior_touch_adj, importance
            if block_count <= 0:
                return

            root = Path(__file__).resolve().parent.parent
            os_mod = __import__("os")
            env_model = os_mod.environ.get("FLOORSET_SOFT_PRIOR_PATH", "").strip()
            model_paths = []
            if env_model:
                model_paths.append(Path(env_model).expanduser())
            model_paths.extend([
                root / "models" / "floorset_soft_prior_eval_base.pt",
                root / "models" / "floorset_soft_prior_h384_mp6.pt",
                root / "models" / "floorset_prior.pt",
                root / "models" / "floorset_prior_seed.pt",
            ])
            model_path = next((p for p in model_paths if p.exists()), None)
            if model_path is None:
                return

            class ClassicPriorNet(torch.nn.Module):
                def __init__(self, feature_dim: int, hidden: int, message_passes: int):
                    super().__init__()
                    self.message_passes = message_passes
                    self.node_in = torch.nn.Sequential(
                        torch.nn.Linear(feature_dim, hidden),
                        torch.nn.LayerNorm(hidden),
                        torch.nn.SiLU(),
                        torch.nn.Linear(hidden, hidden),
                        torch.nn.SiLU(),
                    )
                    self.self_update = torch.nn.Linear(hidden, hidden)
                    self.msg_update = torch.nn.Linear(hidden, hidden)
                    self.norms = torch.nn.ModuleList([torch.nn.LayerNorm(hidden) for _ in range(message_passes)])
                    self.center_head = torch.nn.Sequential(torch.nn.Linear(hidden, hidden), torch.nn.SiLU(), torch.nn.Linear(hidden, 2))
                    self.ratio_head = torch.nn.Sequential(torch.nn.Linear(hidden, hidden), torch.nn.SiLU(), torch.nn.Linear(hidden, len(ratio_candidates)))
                    self.order_head = torch.nn.Sequential(torch.nn.Linear(hidden, hidden // 2), torch.nn.SiLU(), torch.nn.Linear(hidden // 2, 1))

                def forward(self, features, edges, weights):
                    h = self.node_in(features)
                    n = h.shape[0]
                    for layer in range(self.message_passes):
                        agg = torch.zeros_like(h)
                        if edges.numel() > 0:
                            src = edges[:, 0].long()
                            dst = edges[:, 1].long()
                            w = weights.view(-1, 1).to(h.dtype)
                            agg.index_add_(0, src, h[dst] * w)
                            agg.index_add_(0, dst, h[src] * w)
                        h = h + torch.nn.functional.silu(self.self_update(h) + self.msg_update(agg / math.sqrt(max(n, 1))))
                        h = self.norms[layer](h)
                    return {
                        "center": self.center_head(h),
                        "ratio_logits": self.ratio_head(h),
                        "order": self.order_head(h).squeeze(-1),
                    }

            class SoftPriorNet(torch.nn.Module):
                def __init__(self, feature_dim: int, hidden: int, message_passes: int):
                    super().__init__()
                    self.message_passes = message_passes
                    self.node_in = torch.nn.Sequential(
                        torch.nn.Linear(feature_dim, hidden),
                        torch.nn.LayerNorm(hidden),
                        torch.nn.SiLU(),
                        torch.nn.Linear(hidden, hidden),
                        torch.nn.SiLU(),
                    )
                    self.edge_type_embed = torch.nn.Embedding(3, hidden)
                    self.msg_layers = torch.nn.ModuleList([torch.nn.Linear(hidden, hidden) for _ in range(message_passes)])
                    self.self_layers = torch.nn.ModuleList([torch.nn.Linear(hidden, hidden) for _ in range(message_passes)])
                    self.norms = torch.nn.ModuleList([torch.nn.LayerNorm(hidden) for _ in range(message_passes)])
                    self.center_head = torch.nn.Sequential(torch.nn.Linear(hidden, hidden), torch.nn.SiLU(), torch.nn.Linear(hidden, 2))
                    self.ratio_head = torch.nn.Sequential(torch.nn.Linear(hidden, hidden), torch.nn.SiLU(), torch.nn.Linear(hidden, len(ratio_candidates)))
                    self.order_head = torch.nn.Sequential(torch.nn.Linear(hidden, hidden // 2), torch.nn.SiLU(), torch.nn.Linear(hidden // 2, 1))
                    self.boundary_head = torch.nn.Sequential(torch.nn.Linear(hidden, hidden // 2), torch.nn.SiLU(), torch.nn.Linear(hidden // 2, 4))
                    self.mib_ratio_head = torch.nn.Sequential(torch.nn.Linear(hidden, hidden // 2), torch.nn.SiLU(), torch.nn.Linear(hidden // 2, len(ratio_candidates)))
                    self.edge_touch_head = torch.nn.Sequential(
                        torch.nn.Linear(hidden * 3, hidden),
                        torch.nn.SiLU(),
                        torch.nn.Linear(hidden, hidden // 2),
                        torch.nn.SiLU(),
                        torch.nn.Linear(hidden // 2, 1),
                    )

                def forward(self, features, edge_index, edge_weight, edge_type):
                    h = self.node_in(features)
                    n = h.shape[0]
                    for layer in range(self.message_passes):
                        agg = torch.zeros_like(h)
                        if edge_index.numel() > 0:
                            src = edge_index[:, 0].long()
                            dst = edge_index[:, 1].long()
                            weight = edge_weight.to(h.dtype).view(-1, 1)
                            edge_bias = self.edge_type_embed(edge_type.long())
                            src_msg = self.msg_layers[layer](h[src] + edge_bias) * weight
                            dst_msg = self.msg_layers[layer](h[dst] + edge_bias) * weight
                            agg.index_add_(0, dst, src_msg)
                            agg.index_add_(0, src, dst_msg)
                        update = self.self_layers[layer](h) + agg / math.sqrt(max(n, 1))
                        h = self.norms[layer](h + torch.nn.functional.silu(update))
                    return {
                        "embedding": h,
                        "center": self.center_head(h),
                        "ratio_logits": self.ratio_head(h),
                        "order": self.order_head(h).squeeze(-1),
                        "boundary_logits": self.boundary_head(h),
                        "mib_ratio_logits": self.mib_ratio_head(h),
                    }

                def edge_touch_logits(self, embeddings, pair_index):
                    if pair_index.numel() == 0:
                        return embeddings.new_empty(0)
                    a = embeddings[pair_index[:, 0]]
                    b = embeddings[pair_index[:, 1]]
                    pair_features = torch.cat([a + b, (a - b).abs(), a * b], dim=1)
                    return self.edge_touch_head(pair_features).squeeze(-1)

            def group_pairs_from_ids(ids):
                pairs = []
                groups = {}
                for idx, gid in enumerate(ids):
                    if gid > 0:
                        groups.setdefault(gid, []).append(idx)
                for group in groups.values():
                    if len(group) <= 1:
                        continue
                    for a_idx in range(len(group)):
                        for b_idx in range(a_idx + 1, len(group)):
                            pairs.append((group[a_idx], group[b_idx]))
                return pairs

            try:
                stat = model_path.stat()
                cache = getattr(self, "_floorset_prior_cache", None)
                if cache is None or cache.get("path") != str(model_path) or cache.get("mtime") != stat.st_mtime:
                    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
                    state = checkpoint.get("model_state", checkpoint)
                    hidden = int(checkpoint.get("hidden", 128))
                    message_passes = int(checkpoint.get("message_passes", 4))
                    feature_dim = int(checkpoint.get("feature_dim", 12))
                    is_soft = feature_dim >= 16 or "edge_type_embed.weight" in state
                    model = SoftPriorNet(feature_dim, hidden, message_passes) if is_soft else ClassicPriorNet(feature_dim, hidden, message_passes)
                    model.load_state_dict(state, strict=False)
                    model.eval()
                    self._floorset_prior_cache = {
                        "path": str(model_path),
                        "mtime": stat.st_mtime,
                        "model": model,
                        "kind": "soft" if is_soft else "classic",
                        "feature_dim": feature_dim,
                    }
                else:
                    model = cache["model"]
                cache = getattr(self, "_floorset_prior_cache", {})
                prior_kind = cache.get("kind", "classic")
                feature_dim = int(cache.get("feature_dim", 12))

                area_t = torch.tensor(areas, dtype=torch.float32).clamp_min(1e-6)
                scale = torch.sqrt(area_t.sum()).clamp_min(1.0)
                b_degree_t = torch.tensor(b2b_degree, dtype=torch.float32)
                p_degree_t = torch.tensor(p2b_degree, dtype=torch.float32)
                pin_cx = torch.zeros(block_count, dtype=torch.float32)
                pin_cy = torch.zeros(block_count, dtype=torch.float32)
                has_pin = torch.zeros(block_count, dtype=torch.float32)
                for i, centroid in enumerate(pin_centroid):
                    if centroid is not None:
                        has_pin[i] = 1.0
                        pin_cx[i] = float(centroid[0])
                        pin_cy[i] = float(centroid[1])
                fixed_t = torch.tensor([1.0 if fixed_dim[i] else 0.0 for i in range(block_count)], dtype=torch.float32)
                fixed_xy_t = torch.tensor([1.0 if fixed_xy[i] else 0.0 for i in range(block_count)], dtype=torch.float32)
                mib_t = torch.tensor([1.0 if mib_ids[i] > 0 else 0.0 for i in range(block_count)], dtype=torch.float32)
                cluster_t = torch.tensor([1.0 if cluster_ids[i] > 0 else 0.0 for i in range(block_count)], dtype=torch.float32)
                boundary_t = torch.tensor([1.0 if boundary_codes[i] > 0 else 0.0 for i in range(block_count)], dtype=torch.float32)
                boundary_bit_t = torch.tensor(
                    [[1.0 if boundary_codes[i] & bit else 0.0 for bit in (1, 2, 4, 8)] for i in range(block_count)],
                    dtype=torch.float32,
                )

                if prior_kind == "soft" or feature_dim >= 16:
                    features = torch.cat(
                        [
                            (area_t / area_t.sum().clamp_min(1.0)).view(-1, 1),
                            (torch.sqrt(area_t) / scale).view(-1, 1),
                            (b_degree_t / b_degree_t.max().clamp_min(1.0)).view(-1, 1),
                            (p_degree_t / p_degree_t.max().clamp_min(1.0)).view(-1, 1),
                            (pin_cx / scale).view(-1, 1),
                            (pin_cy / scale).view(-1, 1),
                            has_pin.view(-1, 1),
                            fixed_t.view(-1, 1),
                            fixed_xy_t.view(-1, 1),
                            mib_t.view(-1, 1),
                            cluster_t.view(-1, 1),
                            boundary_t.view(-1, 1),
                            boundary_bit_t,
                        ],
                        dim=1,
                    )
                else:
                    max_area_t = area_t.max().clamp_min(1.0)
                    features = torch.stack(
                        [
                            area_t / max_area_t,
                            torch.sqrt(area_t) / torch.sqrt(max_area_t),
                            b_degree_t / b_degree_t.max().clamp_min(1.0),
                            p_degree_t / p_degree_t.max().clamp_min(1.0),
                            has_pin,
                            pin_cx / scale,
                            pin_cy / scale,
                            fixed_t,
                            fixed_xy_t,
                            mib_t,
                            cluster_t,
                            boundary_t,
                        ],
                        dim=1,
                    )
                if features.shape[1] != feature_dim:
                    if features.shape[1] > feature_dim:
                        features = features[:, :feature_dim]
                    else:
                        pad = torch.zeros((block_count, feature_dim - features.shape[1]), dtype=features.dtype)
                        features = torch.cat([features, pad], dim=1)

                if valid_b2b:
                    edges = torch.tensor([[i, j] for i, j, _ in valid_b2b], dtype=torch.long)
                    weights = torch.tensor([w for _, _, w in valid_b2b], dtype=torch.float32)
                    weights = weights / weights.max().clamp_min(1.0)
                else:
                    edges = torch.empty((0, 2), dtype=torch.long)
                    weights = torch.empty(0, dtype=torch.float32)

                if prior_kind == "soft":
                    edge_rows = []
                    edge_weights = []
                    edge_types = []
                    if valid_b2b:
                        for i, j, weight in valid_b2b:
                            edge_rows.append((i, j))
                            edge_weights.append(weight)
                            edge_types.append(0)
                    for ids, edge_type in ((cluster_ids, 1), (mib_ids, 2)):
                        for i, j in group_pairs_from_ids(ids):
                            edge_rows.append((i, j))
                            edge_weights.append(1.0)
                            edge_types.append(edge_type)
                    if edge_rows:
                        edge_index = torch.tensor(edge_rows, dtype=torch.long)
                        edge_weight = torch.tensor(edge_weights, dtype=torch.float32)
                        type0 = torch.tensor([w for t, w in zip(edge_types, edge_weights) if t == 0], dtype=torch.float32)
                        norm = type0.max().clamp_min(1.0) if type0.numel() else torch.tensor(1.0)
                        edge_weight = torch.where(torch.tensor(edge_types, dtype=torch.long) == 0, edge_weight / norm, torch.ones_like(edge_weight))
                        edge_type = torch.tensor(edge_types, dtype=torch.long)
                    else:
                        edge_index = torch.empty((0, 2), dtype=torch.long)
                        edge_weight = torch.empty(0, dtype=torch.float32)
                        edge_type = torch.empty(0, dtype=torch.long)
                    with torch.no_grad():
                        pred = model(features, edge_index, edge_weight, edge_type)
                else:
                    with torch.no_grad():
                        pred = model(features, edges, weights)

                order_values = pred["order"].detach().float().cpu().tolist()
                if order_values:
                    lo = min(order_values)
                    hi = max(order_values)
                    denom = hi - lo
                    prior_order_bonus = [(v - lo) / denom for v in order_values] if denom > 1e-9 else [0.5] * block_count

                ratio_ids = torch.argmax(pred["ratio_logits"], dim=1).detach().cpu().tolist()
                mib_ratio_ids = None
                if "mib_ratio_logits" in pred:
                    mib_ratio_ids = torch.argmax(pred["mib_ratio_logits"], dim=1).detach().cpu().tolist()
                for i, rid in enumerate(ratio_ids[:block_count]):
                    if fixed_dim[i]:
                        continue
                    if mib_ratio_ids is not None and mib_ids[i] > 0:
                        prior_ratio_index[i] = int(mib_ratio_ids[i]) % len(ratio_candidates)
                    else:
                        prior_ratio_index[i] = int(rid) % len(ratio_candidates)

                centers = (pred["center"].detach().float().cpu() * scale).tolist()
                for i, center in enumerate(centers[:block_count]):
                    cx, cy = float(center[0]), float(center[1])
                    if finite(cx) and finite(cy):
                        prior_centers[i] = (cx, cy)

                boundary_conf = [0.0] * block_count
                if "boundary_logits" in pred:
                    bits = torch.sigmoid(pred["boundary_logits"]).detach().float().cpu().tolist()
                    prior_boundary_bits = [list(row) for row in bits[:block_count]]
                    for i in range(block_count):
                        required = [idx for idx, bit in enumerate((1, 2, 4, 8)) if boundary_codes[i] & bit]
                        if required:
                            boundary_conf[i] = sum(prior_boundary_bits[i][idx] for idx in required) / max(len(required), 1)
                        else:
                            boundary_conf[i] = max(prior_boundary_bits[i]) if prior_boundary_bits[i] else 0.0

                if prior_kind == "soft" and "embedding" in pred:
                    pair_set = set()
                    for i, j, _ in valid_b2b:
                        if i != j:
                            pair_set.add((min(i, j), max(i, j)))
                    for i, j in group_pairs_from_ids(cluster_ids):
                        if i != j:
                            pair_set.add((min(i, j), max(i, j)))
                    if pair_set:
                        pair_list = sorted(pair_set)
                        pair_index = torch.tensor(pair_list, dtype=torch.long)
                        with torch.no_grad():
                            touch_probs = torch.sigmoid(model.edge_touch_logits(pred["embedding"], pair_index)).detach().float().cpu().tolist()
                        for (i, j), prob in zip(pair_list, touch_probs):
                            if prob >= 0.50:
                                prior_touch_adj[i].append((j, float(prob)))
                                prior_touch_adj[j].append((i, float(prob)))
                        for i in range(block_count):
                            prior_touch_adj[i].sort(key=lambda item: item[1], reverse=True)
                            prior_touch_adj[i] = prior_touch_adj[i][:12]

                importance = [
                    0.90 * importance[i] + 0.08 * prior_order_bonus[i] + 0.02 * boundary_conf[i]
                    for i in range(block_count)
                ]
                prior_loaded = True
            except Exception:
                prior_loaded = False
                prior_kind = None


        load_trained_prior()

        def bbox_area(positions) -> float:
            xs = [p[0] for p in positions]
            ys = [p[1] for p in positions]
            xe = [p[0] + p[2] for p in positions]
            ye = [p[1] + p[3] for p in positions]
            return max(max(xe) - min(xs), 0.0) * max(max(ye) - min(ys), 0.0)

        def bbox_bounds(positions):
            return (
                min(p[0] for p in positions),
                min(p[1] for p in positions),
                max(p[0] + p[2] for p in positions),
                max(p[1] + p[3] for p in positions),
            )

        def touches_edge(rect, bounds, code: int) -> bool:
            if code == 0:
                return True
            x, y, w, h = rect
            min_x, min_y, max_x, max_y = bounds
            checks = {
                1: abs(x - min_x) <= 1e-5,
                2: abs(x + w - max_x) <= 1e-5,
                4: abs(y + h - max_y) <= 1e-5,
                8: abs(y - min_y) <= 1e-5,
            }
            return all(checks[bit] for bit in (1, 2, 4, 8) if code & bit)

        def edge_connected(a, b) -> bool:
            ax, ay, aw, ah = a
            bx, by, bw, bh = b
            x_overlap = min(ax + aw, bx + bw) - max(ax, bx)
            y_overlap = min(ay + ah, by + bh) - max(ay, by)
            vertical_touch = (abs(ax + aw - bx) <= 1e-5 or abs(bx + bw - ax) <= 1e-5) and y_overlap > 1e-5
            horizontal_touch = (abs(ay + ah - by) <= 1e-5 or abs(by + bh - ay) <= 1e-5) and x_overlap > 1e-5
            return vertical_touch or horizontal_touch

        def edge_gap_distance(a, b) -> float:
            ax, ay, aw, ah = a
            bx, by, bw, bh = b
            x_overlap = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
            y_overlap = max(0.0, min(ay + ah, by + bh) - max(ay, by))
            vertical_gap = min(abs(ax + aw - bx), abs(bx + bw - ax))
            horizontal_gap = min(abs(ay + ah - by), abs(by + bh - ay))
            if y_overlap > 1e-5:
                return vertical_gap
            if x_overlap > 1e-5:
                return horizontal_gap
            center_gap = abs((ax + aw / 2.0) - (bx + bw / 2.0)) + abs((ay + ah / 2.0) - (by + bh / 2.0))
            return min(vertical_gap + horizontal_gap, center_gap)

        def groups_from_ids(ids):
            groups = {}
            for i, gid in enumerate(ids):
                if gid > 0:
                    groups.setdefault(gid, []).append(i)
            return groups

        def edge_components(group, positions):
            if len(group) <= 1:
                return [list(group)] if group else []
            parent = {i: i for i in group}

            def find(a):
                while parent[a] != a:
                    parent[a] = parent[parent[a]]
                    a = parent[a]
                return a

            def union(a, b):
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[rb] = ra

            for idx, a in enumerate(group):
                for b in group[idx + 1:]:
                    if edge_connected(positions[a], positions[b]):
                        union(a, b)
            comps = {}
            for i in group:
                comps.setdefault(find(i), []).append(i)
            return list(comps.values())

        def component_violations(group, positions) -> int:
            return max(0, len(edge_components(group, positions)) - 1)

        def soft_violation_summary(positions):
            if not positions or any(p is None for p in positions):
                return {
                    "boundary": block_count,
                    "grouping": block_count,
                    "mib": block_count,
                    "total": block_count * 3,
                    "n_soft": max(block_count, 1),
                    "relative": 1.0,
                    "boundary_distance": 1e12,
                    "boundary_blocks": list(range(block_count)),
                    "broken_clusters": [],
                    "broken_mibs": [],
                }

            bounds = bbox_bounds(positions)
            boundary_blocks = []
            boundary_distance = 0.0
            for i, code in enumerate(boundary_codes):
                if not code or touches_edge(positions[i], bounds, code):
                    continue
                boundary_blocks.append(i)
                x, y, w, h = positions[i]
                min_x, min_y, max_x, max_y = bounds
                if code & 1:
                    boundary_distance += abs(x - min_x)
                if code & 2:
                    boundary_distance += abs((x + w) - max_x)
                if code & 4:
                    boundary_distance += abs((y + h) - max_y)
                if code & 8:
                    boundary_distance += abs(y - min_y)

            mib_violations = 0
            broken_mibs = []
            mib_groups = groups_from_ids(mib_ids)
            for group in mib_groups.values():
                shapes = set((round(positions[i][2], 4), round(positions[i][3], 4)) for i in group)
                delta = max(0, len(shapes) - 1)
                mib_violations += delta
                if delta:
                    broken_mibs.append(group)

            grouping_violations = 0
            broken_clusters = []
            cluster_groups = groups_from_ids(cluster_ids)
            for group in cluster_groups.values():
                delta = component_violations(group, positions)
                grouping_violations += delta
                if delta:
                    broken_clusters.append(group)

            n_soft = sum(1 for code in boundary_codes if code)
            n_soft += sum(max(0, len(group) - 1) for group in mib_groups.values())
            n_soft += sum(max(0, len(group) - 1) for group in cluster_groups.values())
            total = len(boundary_blocks) + grouping_violations + mib_violations
            return {
                "boundary": len(boundary_blocks),
                "grouping": grouping_violations,
                "mib": mib_violations,
                "total": total,
                "n_soft": n_soft,
                "relative": total / max(n_soft, 1),
                "boundary_distance": boundary_distance,
                "boundary_blocks": boundary_blocks,
                "broken_clusters": broken_clusters,
                "broken_mibs": broken_mibs,
            }

        def soft_penalty(positions) -> float:
            summary = soft_violation_summary(positions)
            return summary["total"] * 220.0 * mean_side + 3.0 * summary["boundary_distance"]

        def hard_feasible(positions) -> bool:
            if positions is None or len(positions) != block_count:
                return False
            for i, pos in enumerate(positions):
                if pos is None or len(pos) != 4:
                    return False
                x, y, w, h = pos
                if not all(finite(v) for v in pos) or w <= 0 or h <= 0:
                    return False
                if fixed_dim[i]:
                    tw = tensor_value(target_positions, i, 2)
                    th = tensor_value(target_positions, i, 3)
                    if abs(w - tw) > hard_eps or abs(h - th) > hard_eps:
                        return False
                if fixed_xy[i]:
                    tx = tensor_value(target_positions, i, 0)
                    ty = tensor_value(target_positions, i, 1)
                    if abs(x - tx) > hard_eps or abs(y - ty) > hard_eps:
                        return False
                if not fixed_dim[i]:
                    if abs(w * h - areas[i]) / max(areas[i], 1e-9) > 0.01:
                        return False
            return not has_any_overlap(positions)

        def full_cost(positions) -> float:
            if positions is None or len(positions) != block_count or any(p is None for p in positions):
                return 1e18
            penalty = 0.0
            if has_any_overlap(positions):
                penalty += 1e15
            for i, (x, y, w, h) in enumerate(positions):
                if not all(finite(v) for v in (x, y, w, h)) or w <= 0 or h <= 0:
                    penalty += 1e15
                    continue
                if fixed_dim[i]:
                    penalty += 1e12 * (
                        abs(w - tensor_value(target_positions, i, 2)) +
                        abs(h - tensor_value(target_positions, i, 3))
                    )
                elif abs(w * h - areas[i]) / max(areas[i], 1e-9) > 0.01:
                    penalty += 1e12
                if fixed_xy[i]:
                    penalty += 1e12 * (
                        abs(x - tensor_value(target_positions, i, 0)) +
                        abs(y - tensor_value(target_positions, i, 1))
                    )

            wire = 0.0
            for i, j, weight in valid_b2b:
                xi, yi, wi, hi = positions[i]
                xj, yj, wj, hj = positions[j]
                wire += weight * (abs((xi + wi / 2.0) - (xj + wj / 2.0)) +
                                  abs((yi + hi / 2.0) - (yj + hj / 2.0)))
            for pin, block, weight in valid_p2b:
                x, y, w, h = positions[block]
                px = tensor_value(pins_pos, pin, 0)
                py = tensor_value(pins_pos, pin, 1)
                wire += weight * (abs((x + w / 2.0) - px) + abs((y + h / 2.0) - py))
            return wire + 0.02 * bbox_area(positions) + soft_penalty(positions) + penalty

        def normalize_if_safe(positions):
            if not positions:
                return positions
            if preplaced_positions:
                return positions
            min_x = min(p[0] for p in positions)
            min_y = min(p[1] for p in positions)
            sx = -min_x if min_x < 0 else 0.0
            sy = -min_y if min_y < 0 else 0.0
            if sx == 0.0 and sy == 0.0:
                return positions
            return [(x + sx, y + sy, w, h) for x, y, w, h in positions]

        def local_candidate_cost(block, rect, positions, placed) -> float:
            x, y, w, h = rect
            overlap = overlap_area_with(rect, positions, placed)
            cx = x + w / 2.0
            cy = y + h / 2.0
            wire = 0.0
            for nbr, weight in b2b_adj[block]:
                if nbr in placed and positions[nbr] is not None:
                    nx, ny, nw, nh = positions[nbr]
                    wire += weight * (abs(cx - (nx + nw / 2.0)) + abs(cy - (ny + nh / 2.0)))
            for pin, weight in p2b_adj[block]:
                px = tensor_value(pins_pos, pin, 0)
                py = tensor_value(pins_pos, pin, 1)
                wire += weight * (abs(cx - px) + abs(cy - py))

            existing = [positions[j] for j in placed if positions[j] is not None]
            if existing:
                min_x = min([p[0] for p in existing] + [x])
                min_y = min([p[1] for p in existing] + [y])
                max_x = max([p[0] + p[2] for p in existing] + [x + w])
                max_y = max([p[1] + p[3] for p in existing] + [y + h])
                area = (max_x - min_x) * (max_y - min_y)
            else:
                min_x = x
                min_y = y
                max_x = x + w
                max_y = y + h
                area = w * h

            boundary = 0.0
            code = boundary_codes[block]
            if code:
                # Scale boundary penalty to dominate the area/HPWL terms.
                # exp(2*V_rel) is a multiplier on the contest score, so a single
                # satisfied bit is worth a meaningful slice of wire/area cost.
                bw = float(__import__("os").environ.get("FLOORSET_BOUNDARY_WEIGHT", "150.0")) * mean_side
                if code & 1 and abs(x - min_x) > 1e-5:
                    boundary += bw
                if code & 2 and abs(x + w - max_x) > 1e-5:
                    boundary += bw
                if code & 4 and abs(y + h - max_y) > 1e-5:
                    boundary += bw
                if code & 8 and abs(y - min_y) > 1e-5:
                    boundary += bw
                # Prefer aspect ratios aligned with the edge: tall+thin for
                # left/right (small w), wide+short for top/bottom (small h).
                # Neutral for corners (both bits set). Magnitude calibrated to
                # the area term so it tips the choice between satisfying ratios
                # without overriding wire/satisfaction signals.
                has_lr = (code & 1) or (code & 2)
                has_tb = (code & 4) or (code & 8)
                if has_lr and not has_tb and w > h:
                    boundary += 0.05 * mean_side * (w - h)
                if has_tb and not has_lr and h > w:
                    boundary += 0.05 * mean_side * (h - w)
            touch = 0.0
            for nbr, affinity in prior_touch_adj[block]:
                if nbr in placed and positions[nbr] is not None:
                    other = positions[nbr]
                    if edge_connected(rect, other):
                        touch -= 0.30 * mean_side * affinity
                    else:
                        touch += 0.08 * affinity * edge_gap_distance(rect, other)

            # Grouping (cluster) constraint: members of the same cluster must
            # be edge-connected. Reward this rect if it abuts a placed sibling;
            # penalize proportional to the smallest gap when no sibling is
            # touched. Weight is below boundary's so quality isn't sacrificed
            # for clusters that have no feasible adjacency.
            cluster_term = 0.0
            cid = cluster_ids[block]
            if cid > 0:
                siblings = [j for j in placed
                            if positions[j] is not None and cluster_ids[j] == cid]
                if siblings:
                    cw = float(__import__("os").environ.get("FLOORSET_CLUSTER_WEIGHT", "30.0"))
                    connected = False
                    min_gap = float("inf")
                    for j in siblings:
                        if edge_connected(rect, positions[j]):
                            connected = True
                            break
                        gap = edge_gap_distance(rect, positions[j])
                        if gap < min_gap:
                            min_gap = gap
                    if connected:
                        cluster_term -= 8.0 * mean_side
                    else:
                        cluster_term += cw * mean_side + 3.0 * min_gap

            return wire + 0.02 * area + 1e9 * overlap + boundary + touch + cluster_term

        def unique_points(points):
            seen = set()
            result = []
            for x, y in points:
                if not finite(x) or not finite(y):
                    continue
                key = (round(float(x), 5), round(float(y), 5))
                if key not in seen:
                    seen.add(key)
                    result.append((float(x), float(y)))
            return result

        def generate_candidates(block, w, h, positions, placed, draft=None, limit=None):
            points = [(0.0, 0.0)]
            if draft is not None and draft[block] is not None:
                points.append((draft[block][0], draft[block][1]))
            if pin_centroid[block] is not None:
                px, py = pin_centroid[block]
                points.append((px - w / 2.0, py - h / 2.0))
            if prior_centers[block] is not None:
                cx, cy = prior_centers[block]
                points.append((cx - w / 2.0, cy - h / 2.0))

            connected = []
            for nbr, weight in b2b_adj[block]:
                if nbr in placed and positions[nbr] is not None:
                    connected.append((weight, nbr))
            for nbr, affinity in prior_touch_adj[block]:
                if nbr in placed and positions[nbr] is not None:
                    connected.append((0.75 * affinity, nbr))
            connected.sort(reverse=True)
            seen_connected = set()
            for _, nbr in connected[:10]:
                if nbr in seen_connected:
                    continue
                seen_connected.add(nbr)
                x, y, nw, nh = positions[nbr]
                points.extend([(x + nw, y), (x, y + nh), (x - w, y), (x, y - h)])

            block_cid = cluster_ids[block]
            cluster_siblings_placed = set()
            if block_cid > 0:
                for j in placed:
                    if positions[j] is not None and cluster_ids[j] == block_cid:
                        cluster_siblings_placed.add(j)
                        jx, jy, jw, jh = positions[j]
                        # All four edge-adjacent positions for sibling abutment
                        points.extend([
                            (jx + jw, jy),
                            (jx - w, jy),
                            (jx, jy + jh),
                            (jx, jy - h),
                        ])
            # Keep b2b/touch as the HPWL prioritization; cluster siblings only
            # contribute via their explicit candidates above + must_keep.
            connected_set = {nbr for nbr, _ in b2b_adj[block]} | {nbr for nbr, _ in prior_touch_adj[block]}
            useful_placed = placed
            if limit is not None and len(useful_placed) > limit:
                useful_placed = sorted(
                    useful_placed,
                    key=lambda j: (j not in connected_set, -importance[j])
                )[:limit]
            for j in useful_placed:
                if positions[j] is None:
                    continue
                x, y, jw, jh = positions[j]
                points.append((x + jw, y))
                points.append((x, y + jh))

            if placed:
                max_x = max(positions[j][0] + positions[j][2] for j in placed if positions[j] is not None)
                max_y = max(positions[j][1] + positions[j][3] for j in placed if positions[j] is not None)
                points.extend([(max_x, 0.0), (0.0, max_y), (max_x, max_y)])
                bit_conf = prior_boundary_bits[block]
                code = boundary_codes[block]
                if code & 1 or bit_conf[0] >= 0.60:
                    points.extend([(0.0, 0.0), (0.0, max_y)])
                if code & 2 or bit_conf[1] >= 0.60:
                    # Extending: block at x=max_x grows bbox by w.
                    points.extend([(max_x, 0.0), (max_x, max_y)])
                    # Aligning: block at x=max_x-w shares the current right wall
                    # (only valid if w fits inside current bbox; clamp handles
                    # negative). Plus aligned stacks above existing right-wall
                    # blocks so we don't have to extend bbox.
                    if w < max_x + 1e-9:
                        points.append((max_x - w, 0.0))
                        for j in placed:
                            if positions[j] is None:
                                continue
                            jx, jy, jw, jh = positions[j]
                            if abs(jx + jw - max_x) < 1e-5:
                                points.append((max_x - w, jy + jh))
                if code & 4 or bit_conf[2] >= 0.60:
                    points.extend([(0.0, max_y), (max_x, max_y)])
                    if h < max_y + 1e-9:
                        points.append((0.0, max_y - h))
                        for j in placed:
                            if positions[j] is None:
                                continue
                            jx, jy, jw, jh = positions[j]
                            if abs(jy + jh - max_y) < 1e-5:
                                points.append((jx + jw, max_y - h))
                if code & 8 or bit_conf[3] >= 0.60:
                    points.extend([(0.0, 0.0), (max_x, 0.0)])
                # Corner candidates that respect the current bbox (no expansion).
                if (code & 2) and (code & 4) and w < max_x + 1e-9 and h < max_y + 1e-9:
                    points.append((max_x - w, max_y - h))
                if (code & 1) and (code & 4) and h < max_y + 1e-9:
                    points.append((0.0, max_y - h))
                if (code & 2) and (code & 8) and w < max_x + 1e-9:
                    points.append((max_x - w, 0.0))

            if boundary_codes[block] & 1:
                points.append((0.0, 0.0))
            if boundary_codes[block] & 8:
                points.append((0.0, 0.0))

            anchors = unique_points(points)
            if limit is not None and len(anchors) > limit:
                desired = anchors[1] if len(anchors) > 1 else anchors[0]
                # Boundary blocks need at least the edge-satisfying anchors to
                # survive the trim. Cluster blocks need at least one anchor
                # edge-adjacent to a placed sibling. Identify both kinds and
                # keep them up front; fill the rest by distance to `desired`.
                code = boundary_codes[block]
                must_keep = []
                if code and placed:
                    max_x = max(positions[j][0] + positions[j][2] for j in placed if positions[j] is not None)
                    max_y = max(positions[j][1] + positions[j][3] for j in placed if positions[j] is not None)
                    for p in anchors:
                        px, py = p
                        ok = True
                        if code & 1 and abs(px - 0.0) > 1e-5:
                            ok = False
                        if code & 2 and abs(px + w - max_x) > 1e-5 and abs(px - max_x) > 1e-5:
                            ok = False
                        if code & 4 and abs(py + h - max_y) > 1e-5 and abs(py - max_y) > 1e-5:
                            ok = False
                        if code & 8 and abs(py - 0.0) > 1e-5:
                            ok = False
                        if ok:
                            must_keep.append(p)
                if cluster_siblings_placed:
                    for p in anchors:
                        if p in must_keep:
                            continue
                        px, py = p
                        rect_p = (px, py, w, h)
                        for j in cluster_siblings_placed:
                            if edge_connected(rect_p, positions[j]):
                                must_keep.append(p)
                                break
                rest = [p for p in anchors if p not in must_keep]
                rest.sort(key=lambda p: ((p[0] - desired[0]) ** 2 + (p[1] - desired[1]) ** 2, p[1], p[0]))
                slots = max(limit - len(must_keep), 0)
                anchors = must_keep + rest[:slots]
                if not anchors:
                    anchors = rest[:limit]
            return anchors

        def exhaustive_fallback_point(block, w, h, positions, placed):
            # Fast deterministic legal fallback.  The earlier MILP-like full
            # x/y grid was too expensive on constrained cases; placing at the
            # current right edge is always non-overlapping with existing blocks.
            if not placed:
                return 0.0, 0.0

            quick = []
            for j in placed:
                if positions[j] is None:
                    continue
                x, y, jw, jh = positions[j]
                quick.append((x + jw, y))
                quick.append((x, y + jh))
            quick.sort(key=lambda p: (p[1], p[0]))
            for x, y in quick[:80]:
                rect = (x, y, w, h)
                if overlap_area_with(rect, positions, placed) <= eps:
                    return x, y

            max_x = max(positions[j][0] + positions[j][2] for j in placed if positions[j] is not None)
            min_y = min(positions[j][1] for j in placed if positions[j] is not None)
            return max_x, min(0.0, min_y)

        def legalize(draft, order, ratio_map=None, try_ratios=False, candidate_limit=(14 if block_count > 80 else 24)):
            result = [None] * block_count
            placed = []
            for i, pos in preplaced_positions.items():
                result[i] = pos
                placed.append(i)

            seen = set(placed)
            complete_order = []
            for i in order:
                if 0 <= i < block_count and i not in seen:
                    complete_order.append(i)
                    seen.add(i)
            for i in sorted(range(block_count), key=lambda b: importance[b], reverse=True):
                if i not in seen:
                    complete_order.append(i)
                    seen.add(i)

            for block in complete_order:
                if fixed_xy[block]:
                    continue

                ratios = [1.0]
                if not fixed_dim[block]:
                    if ratio_map is not None and block in ratio_map:
                        ratios = [ratio_candidates[ratio_map[block] % len(ratio_candidates)]]
                    elif try_ratios:
                        if prior_ratio_index[block] is not None:
                            preferred = prior_ratio_index[block] % len(ratio_candidates)
                            ratios = [ratio_candidates[preferred]] + [r for idx, r in enumerate(ratio_candidates) if idx != preferred]
                        else:
                            ratios = ratio_candidates

                best_rect = None
                best_cost = float("inf")
                for ratio in ratios:
                    if draft is not None and draft[block] is not None:
                        w, h = draft[block][2], draft[block][3]
                        if not fixed_dim[block] and try_ratios:
                            w, h = block_dims(block, ratio)
                    else:
                        w, h = block_dims(block, ratio)
                    if fixed_dim[block]:
                        w, h = block_dims(block)
                    if not (finite(w) and finite(h) and w > 0 and h > 0):
                        w, h = block_dims(block, 1.0)

                    candidates = generate_candidates(block, w, h, result, placed, draft=draft, limit=candidate_limit)
                    for x, y in candidates:
                        if not preplaced_positions:
                            x = max(0.0, x)
                            y = max(0.0, y)
                        rect = (x, y, w, h)
                        if overlap_area_with(rect, result, placed) > eps:
                            continue
                        cost = local_candidate_cost(block, rect, result, placed)
                        if cost < best_cost:
                            best_cost = cost
                            best_rect = rect

                if best_rect is None:
                    w, h = block_dims(block, 1.0)
                    if draft is not None and draft[block] is not None:
                        w, h = draft[block][2], draft[block][3]
                    if fixed_dim[block]:
                        w, h = block_dims(block)
                    x, y = exhaustive_fallback_point(block, w, h, result, placed)
                    best_rect = (x, y, w, h)

                result[block] = best_rect
                placed.append(block)

            return normalize_if_safe(result)

        def make_draft(order, ratio_map=None):
            draft = [None] * block_count
            cursor_x = 0.0
            cursor_y = 0.0
            row_h = 0.0
            row_w = max(math.sqrt(total_area) * 1.2, mean_side)
            for i in range(block_count):
                if fixed_xy[i]:
                    draft[i] = preplaced_positions[i]
            for block in order:
                if fixed_xy[block]:
                    continue
                ratio = 1.0
                if ratio_map is not None and block in ratio_map:
                    ratio = ratio_candidates[ratio_map[block] % len(ratio_candidates)]
                w, h = block_dims(block, ratio)
                if cursor_x > 0 and cursor_x + w > row_w:
                    cursor_x = 0.0
                    cursor_y += row_h
                    row_h = 0.0
                draft[block] = (cursor_x, cursor_y, w, h)
                cursor_x += w
                row_h = max(row_h, h)
            for i in range(block_count):
                if draft[i] is None:
                    w, h = block_dims(i)
                    draft[i] = (0.0, 0.0, w, h)
            return draft

        def force_directed_draft(initial_draft, iters=None, attract_lr=0.45, repel_lr=0.55):
            # Refine draft positions toward HPWL optimum with pairwise
            # repulsion. Attractive force: weighted centroid of b2b + p2b
            # connections. Repulsive force: push overlapping pairs apart along
            # smaller-overlap axis. Output may still overlap — legalize
            # cleans it up, but uses these positions as candidate seeds for
            # individual block placement.
            if iters is None:
                iters = 18 if block_count > 80 else 28
            pos = [list(p) if p is not None else [0.0, 0.0, mean_side, mean_side]
                   for p in initial_draft]
            for _ in range(iters):
                for i in range(block_count):
                    if fixed_xy[i]:
                        continue
                    if boundary_codes[i]:
                        # Don't drag boundary blocks off their edges with
                        # connection forces; the boundary pipeline owns them.
                        continue
                    x, y, w, h = pos[i]
                    cx = x + w / 2.0
                    cy = y + h / 2.0
                    fx = 0.0
                    fy = 0.0
                    wsum = 0.0
                    for nbr, weight in b2b_adj[i]:
                        nx, ny, nw, nh = pos[nbr]
                        fx += weight * (nx + nw / 2.0)
                        fy += weight * (ny + nh / 2.0)
                        wsum += weight
                    for pin, weight in p2b_adj[i]:
                        px = tensor_value(pins_pos, pin, 0)
                        py = tensor_value(pins_pos, pin, 1)
                        fx += weight * px
                        fy += weight * py
                        wsum += weight
                    if wsum > 0:
                        target_cx = fx / wsum
                        target_cy = fy / wsum
                        new_cx = cx + attract_lr * (target_cx - cx)
                        new_cy = cy + attract_lr * (target_cy - cy)
                        pos[i][0] = new_cx - w / 2.0
                        pos[i][1] = new_cy - h / 2.0

                for i in range(block_count):
                    if fixed_xy[i]:
                        continue
                    xi, yi, wi, hi = pos[i]
                    for j in range(i + 1, block_count):
                        if fixed_xy[j]:
                            continue
                        xj, yj, wj, hj = pos[j]
                        ox = min(xi + wi, xj + wj) - max(xi, xj)
                        oy = min(yi + hi, yj + hj) - max(yi, yj)
                        if ox <= 0 or oy <= 0:
                            continue
                        cxi = xi + wi / 2.0
                        cxj = xj + wj / 2.0
                        cyi = yi + hi / 2.0
                        cyj = yj + hj / 2.0
                        i_locked = boundary_codes[i] != 0
                        j_locked = boundary_codes[j] != 0
                        if ox < oy:
                            push = repel_lr * (ox + 0.01)
                            half = push / (1 if (i_locked or j_locked) else 2)
                            if cxi < cxj:
                                if not i_locked:
                                    pos[i][0] -= half
                                if not j_locked:
                                    pos[j][0] += half
                            else:
                                if not i_locked:
                                    pos[i][0] += half
                                if not j_locked:
                                    pos[j][0] -= half
                            xi, yi, wi, hi = pos[i]
                        else:
                            push = repel_lr * (oy + 0.01)
                            half = push / (1 if (i_locked or j_locked) else 2)
                            if cyi < cyj:
                                if not i_locked:
                                    pos[i][1] -= half
                                if not j_locked:
                                    pos[j][1] += half
                            else:
                                if not i_locked:
                                    pos[i][1] += half
                                if not j_locked:
                                    pos[j][1] -= half
                            xi, yi, wi, hi = pos[i]

            return [tuple(p) for p in pos]

        def build_clusters():
            if block_count == 0:
                return []
            if block_count < 48 or not valid_b2b:
                return [[i] for i in sorted(range(block_count), key=lambda b: importance[b], reverse=True)]
            parent = list(range(block_count))
            size = [1] * block_count
            area_sum = areas[:]
            target_size = max(3, int(math.sqrt(block_count)))
            target_cluster_count = max(4, int(math.sqrt(block_count)))
            max_cluster_area = total_area / target_cluster_count * 1.45

            def find(a):
                while parent[a] != a:
                    parent[a] = parent[parent[a]]
                    a = parent[a]
                return a

            def union(a, b):
                ra, rb = find(a), find(b)
                if ra == rb:
                    return
                if size[ra] < size[rb]:
                    ra, rb = rb, ra
                parent[rb] = ra
                size[ra] += size[rb]
                area_sum[ra] += area_sum[rb]

            for i, j, weight in sorted(valid_b2b, key=lambda e: e[2], reverse=True):
                ri, rj = find(i), find(j)
                if ri == rj:
                    continue
                if size[ri] + size[rj] <= target_size and area_sum[ri] + area_sum[rj] <= max_cluster_area:
                    union(ri, rj)

            prior_edges = []
            for i, nbrs in enumerate(prior_touch_adj):
                for j, affinity in nbrs:
                    if i < j and affinity >= 0.58:
                        prior_edges.append((affinity, i, j))
            for affinity, i, j in sorted(prior_edges, reverse=True):
                ri, rj = find(i), find(j)
                if ri == rj:
                    continue
                if size[ri] + size[rj] <= target_size and area_sum[ri] + area_sum[rj] <= max_cluster_area:
                    union(ri, rj)

            groups = {}
            for i in range(block_count):
                groups.setdefault(find(i), []).append(i)
            clusters = list(groups.values())
            clusters.sort(key=lambda c: sum(importance[i] for i in c), reverse=True)
            return [sorted(c, key=lambda b: importance[b], reverse=True) for c in clusters]

        clusters = build_clusters()
        order_importance = sorted(range(block_count), key=lambda b: importance[b], reverse=True)
        order_prior = sorted(range(block_count), key=lambda b: (prior_order_bonus[b], importance[b]), reverse=True) if prior_loaded else order_importance[:]
        order_area = sorted(range(block_count), key=lambda b: areas[b], reverse=True)
        order_cluster = [b for c in clusters for b in c]
        mib_groups_for_ratio = {}
        for i, gid in enumerate(mib_ids):
            if gid > 0 and not fixed_dim[i]:
                mib_groups_for_ratio.setdefault(gid, []).append(i)

        def mib_group_can_share_ratio(group):
            if len(group) <= 1:
                return False
            base = areas[group[0]]
            return all(abs(areas[b] - base) <= max(1e-5, 1e-5 * base) for b in group)

        def enforce_mib_ratio_map(ratio_map):
            result = dict(ratio_map or {})
            for group in mib_groups_for_ratio.values():
                if not mib_group_can_share_ratio(group):
                    continue
                chosen = None
                for b in sorted(group, key=lambda x: importance[x], reverse=True):
                    if b in result:
                        chosen = result[b] % len(ratio_candidates)
                        break
                if chosen is None:
                    chosen = 0
                for b in group:
                    result[b] = chosen
            return result

        prior_ratio_map = enforce_mib_ratio_map({i: idx for i, idx in enumerate(prior_ratio_index) if idx is not None})

        # ------------------------------------------------------------------
        # Constructive frame tiling ("frame_rows"): build a rectangular die
        # from horizontal rows of exact-area cells.  Boundary-coded blocks
        # are pinned to the perimeter (bottom/top rows, row starts/ends),
        # clusters stay contiguous runs, MIB groups share dimensions, and
        # whitespace stays near zero because soft cell widths absorb their
        # row's height exactly.  Preplaced blocks cut rows into intervals.
        # ------------------------------------------------------------------

        def frame_anchor_frame():
            # Die frame: edges anchored by preplaced blocks carrying boundary
            # codes (the bbox edge must pass through their rect edge), area
            # from the content budget, aspect hinted by the pin span.
            left_anchor = right_anchor = bottom_anchor = top_anchor = None
            for i, rect in preplaced_positions.items():
                code = boundary_codes[i]
                x, y, w, h = rect
                if code & 1:
                    left_anchor = x if left_anchor is None else min(left_anchor, x)
                if code & 2:
                    right_anchor = (x + w) if right_anchor is None else max(right_anchor, x + w)
                if code & 8:
                    bottom_anchor = y if bottom_anchor is None else min(bottom_anchor, y)
                if code & 4:
                    top_anchor = (y + h) if top_anchor is None else max(top_anchor, y + h)

            pre_rects = list(preplaced_positions.values())

            x_lo = 0.0 if left_anchor is None else left_anchor
            y_lo = 0.0 if bottom_anchor is None else bottom_anchor
            x_hi_min = None
            y_hi_min = None
            for rx, ry, rw, rh in pre_rects:
                x_lo = min(x_lo, rx)
                y_lo = min(y_lo, ry)
                x_hi_min = rx + rw if x_hi_min is None else max(x_hi_min, rx + rw)
                y_hi_min = ry + rh if y_hi_min is None else max(y_hi_min, ry + rh)

            place_area = sum(areas[i] for i in range(block_count) if not fixed_xy[i])
            pre_area = sum(r[2] * r[3] for r in pre_rects)
            budget = (place_area + pre_area) * 1.015

            aspect = 1.0
            pin_xs = []
            pin_ys = []
            for pin, _block, _w in valid_p2b:
                pin_xs.append(tensor_value(pins_pos, pin, 0))
                pin_ys.append(tensor_value(pins_pos, pin, 1))
            if pin_xs:
                span_x = max(pin_xs) - min(pin_xs)
                span_y = max(pin_ys) - min(pin_ys)
                if span_x > eps and span_y > eps:
                    aspect = max(0.25, min(4.0, span_x / span_y))

            x_hi = x_hi_min
            y_hi = y_hi_min
            if right_anchor is not None:
                x_hi = right_anchor if x_hi is None else max(x_hi, right_anchor)
            if top_anchor is not None:
                y_hi = top_anchor if y_hi is None else max(y_hi, top_anchor)

            if right_anchor is not None and top_anchor is None:
                cand = y_lo + budget / max(x_hi - x_lo, eps)
                y_hi = cand if y_hi is None else max(y_hi, cand)
            elif top_anchor is not None and right_anchor is None:
                cand = x_lo + budget / max(y_hi - y_lo, eps)
                x_hi = cand if x_hi is None else max(x_hi, cand)
            elif right_anchor is None and top_anchor is None:
                cand_x = x_lo + math.sqrt(max(budget, eps) * aspect)
                x_hi = cand_x if x_hi is None else max(x_hi, cand_x)
                cand_y = y_lo + budget / max(x_hi - x_lo, eps)
                y_hi = cand_y if y_hi is None else max(y_hi, cand_y)

            # Content must fit; if anchors leave too little room grow the
            # edge that breaks fewer boundary constraints.
            avail = (x_hi - x_lo) * (y_hi - y_lo) - pre_area
            if avail < place_area * 1.002:
                n_right = sum(1 for i in range(block_count) if boundary_codes[i] & 2 and not fixed_xy[i])
                n_top = sum(1 for i in range(block_count) if boundary_codes[i] & 4 and not fixed_xy[i])
                need = place_area * 1.004 + pre_area
                if n_top <= n_right:
                    y_hi = y_lo + need / max(x_hi - x_lo, eps)
                else:
                    x_hi = x_lo + need / max(y_hi - y_lo, eps)
            pre_top_max = max((r[1] + r[3] for r in pre_rects), default=None)
            return x_lo, y_lo, x_hi, y_hi, pre_top_max

        def frame_target_centers(frame):
            # Connectivity-driven desired centers: damped Jacobi averaging of
            # neighbour centres with pins and preplaced blocks as anchors.
            x_lo, y_lo, x_hi, y_hi = frame[:4]
            cx0 = (x_lo + x_hi) / 2.0
            cy0 = (y_lo + y_hi) / 2.0
            txs = [cx0] * block_count
            tys = [cy0] * block_count
            for i in range(block_count):
                if fixed_xy[i]:
                    px, py, pw, ph = preplaced_positions[i]
                    txs[i] = px + pw / 2.0
                    tys[i] = py + ph / 2.0
                elif pin_centroid[i] is not None:
                    txs[i], tys[i] = pin_centroid[i]
            for _ in range(40):
                new_tx = txs[:]
                new_ty = tys[:]
                for i in range(block_count):
                    if fixed_xy[i]:
                        continue
                    sx = sy = sw = 0.0
                    for nbr, weight in b2b_adj[i]:
                        sx += weight * txs[nbr]
                        sy += weight * tys[nbr]
                        sw += weight
                    if pin_centroid[i] is not None and p2b_degree[i] > 0:
                        sx += p2b_degree[i] * pin_centroid[i][0]
                        sy += p2b_degree[i] * pin_centroid[i][1]
                        sw += p2b_degree[i]
                    if sw > 0:
                        new_tx[i] = 0.7 * (sx / sw) + 0.3 * txs[i]
                        new_ty[i] = 0.7 * (sy / sw) + 0.3 * tys[i]
                txs, tys = new_tx, new_ty
            for i in range(block_count):
                txs[i] = min(max(txs[i], x_lo), x_hi)
                tys[i] = min(max(tys[i], y_lo), y_hi)
            return txs, tys

        def build_frame_rows(h_scale=1.0, target_override=None):
            if block_count <= 0:
                return None
            frame = frame_anchor_frame()
            x_lo, y_lo, x_hi, y_hi, pre_top_max = frame
            width_total = x_hi - x_lo
            height_total = y_hi - y_lo
            if width_total <= eps or height_total <= eps:
                return None
            if target_override is not None:
                txs, tys = target_override
            else:
                txs, tys = frame_target_centers(frame)

            placeable = [i for i in range(block_count) if not fixed_xy[i]]
            pre_rects = list(preplaced_positions.values())
            pre_area = sum(r[2] * r[3] for r in pre_rects)
            place_area = sum(areas[i] for i in placeable)
            if not placeable:
                return [tuple(preplaced_positions[i]) for i in range(block_count)] if len(preplaced_positions) == block_count else None
            avail = width_total * height_total - pre_area
            fill_factor = min(1.0, max(0.85, place_area / max(avail, eps)))

            def free_intervals(y0, y1):
                # Free x-spans of the frame not occluded by preplaced rects
                # intersecting the horizontal band (y0, y1).
                blocked = []
                for rx, ry, rw, rh in pre_rects:
                    if ry < y1 - 1e-9 and ry + rh > y0 + 1e-9:
                        blocked.append((max(rx, x_lo), min(rx + rw, x_hi)))
                blocked.sort()
                intervals = []
                cursor = x_lo
                for bx0, bx1 in blocked:
                    if bx1 <= cursor + 1e-9:
                        continue
                    if bx0 > cursor + 1e-9:
                        intervals.append((cursor, bx0))
                    cursor = max(cursor, bx1)
                if x_hi > cursor + 1e-9:
                    intervals.append((cursor, x_hi))
                return [(a, b) for a, b in intervals if b - a > 1e-7]

            # ---- units ---------------------------------------------------
            # A unit is a horizontal run of cells placed together in a row.
            # Cells: ('soft', block) spans the full row height with width
            # area/h; ('fixed', block, w, h) keeps exact dims bottom-aligned.
            areas_sorted_hint = sorted(areas[b] for b in placeable)
            med_area_hint = areas_sorted_hint[len(areas_sorted_hint) // 2]
            mib_groups = {}
            for i in placeable:
                if mib_ids[i] > 0:
                    mib_groups.setdefault(mib_ids[i], []).append(i)
            cluster_groups = {}
            for i in placeable:
                if cluster_ids[i] > 0:
                    cluster_groups.setdefault(cluster_ids[i], []).append(i)

            # MIB groups become fixed-dim cells when a member has hard dims
            # or members are scattered into clusters; otherwise they form a
            # same-row run of identical full-height cells.
            mib_cell_dims = {}
            mib_as_run = {}
            for gid, members in mib_groups.items():
                fixed_members = [b for b in members if fixed_dim[b]]
                in_cluster = any(cluster_ids[b] > 0 for b in members)
                if fixed_members:
                    fw = float(tensor_value(target_positions, fixed_members[0], 2))
                    fh = float(tensor_value(target_positions, fixed_members[0], 3))
                    if fw > eps and fh > eps:
                        for b in members:
                            if fixed_dim[b] or abs(fw * fh - areas[b]) / max(areas[b], 1e-9) <= 0.009:
                                mib_cell_dims[b] = (fw, fh)
                elif in_cluster:
                    side = math.sqrt(max(min(areas[b] for b in members), eps))
                    for b in members:
                        mib_cell_dims[b] = (side, areas[b] / side)
                else:
                    mib_as_run[gid] = members

            def block_cell(b):
                if b in mib_cell_dims:
                    w, h = mib_cell_dims[b]
                    return ("fixed", b, w, h)
                if fixed_dim[b]:
                    return ("fixed", b,
                            float(tensor_value(target_positions, b, 2)),
                            float(tensor_value(target_positions, b, 3)))
                return ("soft", b)

            units = []
            in_unit = set()

            def add_unit(members, kind):
                members = [b for b in members if b not in in_unit]
                if not members:
                    return
                in_unit.update(members)
                code_l = any(boundary_codes[b] & 1 for b in members)
                code_r = any(boundary_codes[b] & 2 for b in members)
                code_t = any(boundary_codes[b] & 4 for b in members)
                code_b = any(boundary_codes[b] & 8 for b in members)
                units.append({
                    "members": members,
                    "kind": kind,
                    "cells": [block_cell(b) for b in members],
                    "area": sum(areas[b] for b in members),
                    "L": code_l, "R": code_r, "T": code_t, "B": code_b,
                    "tx": sum(txs[b] for b in members) / len(members),
                    "ty": sum(tys[b] for b in members) / len(members),
                })

            for gid, members in sorted(cluster_groups.items()):
                # Bottom/top-coded members split into their own chunk so the
                # forced edge row only carries them; the remainder is given a
                # ty just above/below the edge so the chunks land in adjacent
                # rows and stay edge-connected.
                b_members = [b for b in members if boundary_codes[b] & 8]
                t_members = [b for b in members if boundary_codes[b] & 4 and not (boundary_codes[b] & 8)]
                rest = [b for b in members if b not in b_members and b not in t_members]

                def cluster_ordered(sub):
                    lefts = [b for b in sub if boundary_codes[b] & 1]
                    rights = [b for b in sub if boundary_codes[b] & 2 and not (boundary_codes[b] & 1)]
                    interior = [b for b in sub if b not in lefts and b not in rights]
                    return lefts + sorted(interior, key=lambda b: txs[b]) + rights

                if b_members and rest and not t_members:
                    add_unit(cluster_ordered(b_members), "cluster")
                    add_unit(cluster_ordered(rest), "cluster")
                    units[-1]["ty"] = y_lo + 1.3 * math.sqrt(max(med_area_hint, eps))
                elif t_members and rest and not b_members:
                    add_unit(cluster_ordered(t_members), "cluster")
                    add_unit(cluster_ordered(rest), "cluster")
                    units[-1]["ty"] = y_hi - 1.3 * math.sqrt(max(med_area_hint, eps))
                else:
                    add_unit(cluster_ordered(members), "cluster")
            for gid, members in sorted(mib_as_run.items()):
                lefts = [b for b in members if boundary_codes[b] & 1]
                rights = [b for b in members if boundary_codes[b] & 2 and not (boundary_codes[b] & 1)]
                interior = [b for b in members if b not in lefts and b not in rights]
                add_unit(lefts + sorted(interior, key=lambda b: txs[b]) + rights, "mib")
            for b in placeable:
                if b not in in_unit:
                    add_unit([b], "single")

            # ---- row plan --------------------------------------------------
            sorted_areas = sorted(areas[b] for b in placeable)
            med_area = sorted_areas[len(sorted_areas) // 2]
            h_target = max(math.sqrt(max(med_area, eps) * 1.35) * h_scale, height_total / 60.0)
            est_rows = max(2, int(round(height_total / h_target)))

            def edge_pending(flag):
                return [u for u in units if u[flag] and not (u["B"] or u["T"])]

            if (len(edge_pending("L")) > est_rows - 1
                    or len(edge_pending("R")) > est_rows - 1):
                # Pair small single left/right units into vertical stacks so
                # one row start/end can host two edge blocks.
                def pair_stacks(flag):
                    while True:
                        edge_now = edge_pending(flag)
                        if len(edge_now) <= max(est_rows - 1, 1):
                            break
                        singles = [u for u in edge_now
                                   if u["kind"] in ("single", "stack") and len(u["members"]) < 3]
                        singles.sort(key=lambda u: (len(u["members"]), u["ty"]))
                        if len(singles) < 2:
                            break
                        first, second = singles[0], singles[1]
                        units.remove(first)
                        units.remove(second)
                        merged_members = first["members"] + second["members"]
                        in_unit.difference_update(merged_members)
                        add_unit(merged_members, "stack")
                pair_stacks("L")
                pair_stacks("R")

            bottom_units = [u for u in units if u["B"] and not u["T"]]
            top_units = [u for u in units if u["T"]]
            left_queue = sorted(edge_pending("L"), key=lambda u: u["ty"])
            right_queue = sorted([u for u in edge_pending("R") if not u["L"]], key=lambda u: u["ty"])
            reserved_ids = {id(u) for u in bottom_units + top_units + left_queue + right_queue}
            pool = sorted([u for u in units if id(u) not in reserved_ids], key=lambda u: u["ty"])

            positions = [None] * block_count
            for i, rect in preplaced_positions.items():
                positions[i] = tuple(rect)

            cuts = sorted({float(y_hi)} | {float(r[1]) for r in pre_rects} | {float(r[1] + r[3]) for r in pre_rects})

            def patchable(u):
                return (u["kind"] == "single"
                        and not (u["L"] or u["R"] or u["B"] or u["T"])
                        and len(u["cells"]) == 1 and u["cells"][0][0] == "soft")

            def take_patch(target_area, max_area):
                # Pull the best-fitting small soft single out of the pool to
                # fill a pocket above a fixed-dimension cell.
                best = None
                best_diff = None
                for u in pool:
                    if not patchable(u) or u["area"] > max_area:
                        continue
                    diff = abs(u["area"] - target_area)
                    if best_diff is None or diff < best_diff:
                        best_diff = diff
                        best = u
                if best is not None and best["area"] >= 0.30 * target_area:
                    pool.remove(best)
                    return best
                return None

            def remaining_area():
                return (sum(u["area"] for u in pool)
                        + sum(u["area"] for u in left_queue)
                        + sum(u["area"] for u in right_queue)
                        + sum(u["area"] for u in top_units))

            # ---- build rows bottom-up --------------------------------------
            current_y = y_lo
            row_no = 0
            max_rows = 4 * est_rows + 12
            pending_b = list(bottom_units)
            failed_units = []

            # Split oversized cluster runs into row-sized chunks; chunks keep
            # similar ty so they land in consecutive rows and stay
            # edge-connected through vertical adjacency.  Bottom/top-coded
            # members are grouped into the first chunk so the forced row
            # keeps them.
            chunk_cap = 0.62 * width_total * h_target
            for u in [u_ for u_ in units if u_["kind"] == "cluster" and u_["area"] > 1.20 * chunk_cap]:
                members = sorted(u["members"], key=lambda b: (
                    0 if boundary_codes[b] & 8 else (2 if boundary_codes[b] & 4 else 1), txs[b]))
                chunks = []
                cur = []
                cur_area = 0.0
                for b in members:
                    cur.append(b)
                    cur_area += areas[b]
                    if cur_area >= 0.85 * chunk_cap:
                        chunks.append(cur)
                        cur = []
                        cur_area = 0.0
                if cur:
                    if chunks and cur_area < 0.25 * chunk_cap:
                        chunks[-1].extend(cur)
                    else:
                        chunks.append(cur)
                if len(chunks) <= 1:
                    continue
                units.remove(u)
                in_unit.difference_update(u["members"])
                for chunk in chunks:
                    add_unit(chunk, "cluster")
            bottom_units = [u for u in units if u["B"] and not u["T"]]
            top_units = [u for u in units if u["T"]]
            left_queue = sorted(edge_pending("L"), key=lambda u: u["ty"])
            right_queue = sorted([u for u in edge_pending("R") if not u["L"]], key=lambda u: u["ty"])
            reserved_ids = {id(u) for u in bottom_units + top_units + left_queue + right_queue}
            pool = sorted([u for u in units if id(u) not in reserved_ids], key=lambda u: u["ty"])
            pending_b = list(bottom_units)

            def unit_width(u, h_val):
                width_sum = 0.0
                for cell in u["cells"]:
                    if cell[0] == "soft":
                        width_sum += areas[cell[1]] / h_val
                    else:
                        width_sum += cell[2]
                return width_sum

            def pack_row(ordered, h_val, iv, row_y_val):
                # Dry-run interval packing.  Returns rect assignments, the
                # units that fit, leftover units, and pocket gaps above
                # fixed cells.
                rects = {}
                pockets = []
                fitted = []
                leftover = []
                iv_idx = 0
                cursor = iv[0][0]
                for u in ordered:
                    uw = unit_width(u, h_val)
                    while iv_idx < len(iv) and cursor + uw > iv[iv_idx][1] + 1e-9:
                        iv_idx += 1
                        if iv_idx < len(iv):
                            cursor = iv[iv_idx][0]
                    if iv_idx >= len(iv):
                        leftover.append(u)
                        iv_idx = len(iv) - 1
                        cursor = iv[iv_idx][1]
                        continue
                    if (u["R"] and not u["L"] and u is ordered[-1]
                            and abs(iv[-1][1] - x_hi) < 1e-9 and iv_idx == len(iv) - 1):
                        cursor = max(x_hi - uw, cursor)
                    x_run = cursor
                    for cell in u["cells"]:
                        if cell[0] == "soft":
                            b = cell[1]
                            cw = areas[b] / h_val
                            rects[b] = (x_run, row_y_val, cw, h_val)
                            x_run += cw
                        else:
                            b, fw, fh = cell[1], cell[2], cell[3]
                            rects[b] = (x_run, row_y_val, fw, fh)
                            if h_val - fh > 1e-7:
                                pockets.append((x_run, row_y_val + fh, fw, h_val - fh))
                            x_run += fw
                    cursor = x_run
                    fitted.append(u)
                return rects, fitted, leftover, pockets

            # Pre-pin bottom-coded units as bottom-anchored obstacles with
            # natural heights; subsequent rows flow around them exactly like
            # preplaced blocks.  This avoids forcing one uniform-height
            # bottom row for all of them.
            def overhead_limit(x0, x1):
                lim = max(y_hi - y_lo, h_target)
                for rx, ry, rw, rh in pre_rects:
                    if rx < x1 - 1e-9 and rx + rw > x0 + 1e-9:
                        lim = min(lim, ry - y_lo)
                return lim

            if pending_b:
                corner_l = [u for u in pending_b if u["L"]]
                corner_r = [u for u in pending_b if u["R"] and not u["L"]]
                middle_b = [u for u in pending_b if not u["L"] and not u["R"]]
                ordered_b = (corner_l + sorted(middle_b, key=lambda u: u["tx"]) + corner_r)
                cursor = x_lo
                unpinned = []
                pinned_rects = []
                pinned_blocks = {}
                for u in ordered_b:
                    u_soft = sum(areas[c[1]] for c in u["cells"] if c[0] == "soft")
                    u_fixed_w = sum(c[2] for c in u["cells"] if c[0] == "fixed")
                    u_fixed_h = max((c[3] for c in u["cells"] if c[0] == "fixed"), default=0.0)
                    h_i = max(min(math.sqrt(max(u["area"], eps) * 1.2), 2.4 * h_target),
                              0.8 * h_target, u_fixed_h, eps)
                    placed_pin = False
                    for _ in range(3):
                        w_i = u_soft / max(h_i, eps) + u_fixed_w
                        start = cursor
                        if u["R"] and not u["L"]:
                            start = max(x_hi - w_i, cursor)
                        lim = overhead_limit(start, start + w_i)
                        if max(u_fixed_h, 0.45 * h_target) > lim + 1e-9:
                            # Too little headroom here; leave to the row engine.
                            break
                        if h_i <= lim + 1e-9:
                            if start + w_i <= x_hi + 1e-9:
                                # Commit the pin.
                                x_run = start
                                for cell in u["cells"]:
                                    if cell[0] == "soft":
                                        b = cell[1]
                                        cw = areas[b] / h_i
                                        pinned_blocks[b] = (x_run, y_lo, cw, h_i)
                                        x_run += cw
                                    else:
                                        b, fw, fh = cell[1], cell[2], cell[3]
                                        pinned_blocks[b] = (x_run, y_lo, fw, fh)
                                        x_run += fw
                                if u["kind"] == "stack":
                                    xs = start
                                    stack_w = sum(areas[b] for b in u["members"]) / h_i
                                    yy = y_lo
                                    for b in u["members"]:
                                        bh = areas[b] / stack_w
                                        pinned_blocks[b] = (xs, yy, stack_w, bh)
                                        yy += bh
                                pinned_rects.append((start, y_lo, x_run - start, h_i))
                                cursor = x_run
                                placed_pin = True
                            break
                        h_i = max(lim, u_fixed_h)
                    if not placed_pin:
                        unpinned.append(u)
                if pinned_rects:
                    for b, rect in pinned_blocks.items():
                        positions[b] = rect
                    pre_rects.extend(pinned_rects)
                    pinned_area = sum(r[2] * r[3] for r in pinned_rects)
                    pre_area += pinned_area
                    place_area = max(place_area - sum(
                        u["area"] for u in ordered_b if u not in unpinned), eps)
                    avail = width_total * height_total - pre_area
                    fill_factor = min(1.0, max(0.85, place_area / max(avail, eps)))
                    cuts = sorted(set(cuts) | {float(r[1] + r[3]) for r in pinned_rects})
                pending_b = unpinned

            while (pending_b or pool or left_queue or right_queue or top_units) and row_no < max_rows:
                row_no += 1
                room_left = max(y_hi - current_y, 0.0)
                rem = remaining_area() + sum(u["area"] for u in pending_b)
                if os_mod.environ.get("FLOORSET_FRAME_DEBUG", "") == "2":
                    print(f"      enter row {row_no}: y={current_y:.1f} pend_b={len(pending_b)} pool={len(pool)} "
                          f"lq={len(left_queue)} rq={len(right_queue)} top={len(top_units)} rem={rem:.0f}")

                # Rows live inside bands delimited by preplaced y-edges, so
                # the free intervals are constant for the whole row and the
                # height computation cannot feed back on itself.
                next_cuts = [c for c in cuts if c > current_y + 0.30 * h_target]
                band_top = next_cuts[0] if next_cuts else max(y_hi, current_y + h_target)
                h_cap = band_top - current_y
                iv = free_intervals(current_y, current_y + min(h_cap, h_target) * 0.999)
                if not iv:
                    current_y = band_top
                    continue
                length = sum(b - a for a, b in iv)
                widest = max(b - a for a, b in iv)

                is_bottom = bool(pending_b)
                is_top = (not pending_b and (
                    rem <= 1.9 * h_target * max(length, eps)
                    or not (pool or left_queue or right_queue)))

                content = []
                if is_bottom:
                    content.extend(pending_b)
                    pending_b = []
                if is_top:
                    content.extend(top_units)
                    top_units = []
                    content.extend(left_queue)
                    left_queue = []
                    content.extend(right_queue)
                    right_queue = []
                    content.extend(pool)
                    pool = []
                else:
                    # One left/right edge unit per row when the row actually
                    # touches the corresponding frame edge.
                    rows_left_est = max(room_left / max(h_target, eps), 1.0)
                    if (left_queue and abs(iv[0][0] - x_lo) < 1e-7
                            and not any(u["L"] for u in content)):
                        mid_y = current_y + h_target / 2.0
                        pick = min(left_queue, key=lambda u: abs(u["ty"] - mid_y))
                        if (abs(pick["ty"] - mid_y) < 2.5 * h_target
                                or len(left_queue) >= rows_left_est - 1):
                            left_queue.remove(pick)
                            content.append(pick)
                    if (right_queue and abs(iv[-1][1] - x_hi) < 1e-7
                            and not any(u["R"] for u in content)):
                        mid_y = current_y + h_target / 2.0
                        pick = min(right_queue, key=lambda u: abs(u["ty"] - mid_y))
                        if (abs(pick["ty"] - mid_y) < 2.5 * h_target
                                or len(right_queue) >= rows_left_est - 1):
                            right_queue.remove(pick)
                            content.append(pick)
                    # Fill from the ty-sorted pool with a small lookahead.
                    cap = max(length * h_target * 1.04,
                              sum(u["area"] for u in content))
                    load = sum(u["area"] for u in content)
                    while pool and load < cap:
                        picked = None
                        for u in pool[:6]:
                            if load + u["area"] > cap * 1.30:
                                continue
                            # Skip units that can never fit this band's
                            # widest interval even at the band-cap height.
                            if unit_width(u, max(h_cap, h_target)) > 0.95 * widest:
                                continue
                            picked = u
                            break
                        if picked is None:
                            fitting = [u for u in pool[:6]
                                       if unit_width(u, max(h_cap, h_target)) <= 0.95 * widest]
                            if load < 0.55 * cap and fitting:
                                picked = min(fitting, key=lambda u: u["area"])
                            else:
                                break
                        pool.remove(picked)
                        content.append(picked)
                        load += picked["area"]

                if not content:
                    break

                def content_sums(units_list):
                    s_area = 0.0
                    f_w = 0.0
                    f_h = 0.0
                    for u in units_list:
                        for cell in u["cells"]:
                            if cell[0] == "soft":
                                s_area += areas[cell[1]]
                            else:
                                f_w += cell[2]
                                f_h = max(f_h, cell[3])
                    return s_area, f_w, f_h

                def min_fit_height(units_list, widest_len):
                    h_fit = 0.0
                    for u in units_list:
                        u_soft = sum(areas[c[1]] for c in u["cells"] if c[0] == "soft")
                        u_fixed = sum(c[2] for c in u["cells"] if c[0] == "fixed")
                        cap_w = 0.95 * widest_len - u_fixed
                        if cap_w > eps and u_soft > 0:
                            h_fit = max(h_fit, u_soft / cap_w)
                    return h_fit

                soft_area, fixed_w_sum, max_fixed_h = content_sums(content)
                denom = length * fill_factor - fixed_w_sum
                h_needed = soft_area / denom if denom > eps else h_cap
                h_needed = max(h_needed, min_fit_height(content, widest), max_fixed_h)

                row_y = current_y
                grew_past_cap = False
                if is_top:
                    # Land the top row flush with the highest hard edge so
                    # top-coded blocks touch the bounding box; float the row
                    # upward when content ends below that edge.
                    h_row = h_needed
                    if pre_top_max is not None and any(u["T"] for u in content):
                        row_y = max(current_y, pre_top_max - h_row)
                        h_row = max(h_row, pre_top_max - row_y)
                    if row_y + h_row > band_top + 1e-9:
                        grew_past_cap = True
                elif h_needed > h_cap + eps:
                    if is_bottom:
                        # Bottom rows must keep their content; cross the cut
                        # conservatively.
                        h_row = h_needed
                        grew_past_cap = True
                    else:
                        # Cap at the band edge and shed the overflow: first
                        # units too wide for the band's widest interval, then
                        # by area until the height fits.
                        h_row = h_cap
                        for u in [u_ for u_ in content
                                  if not (u_["L"] or u_["R"] or u_["B"] or u_["T"])]:
                            if unit_width(u, h_cap) > 0.95 * widest:
                                content.remove(u)
                                pool.insert(0, u)
                        droppable = sorted(
                            [u for u in content
                             if not (u["L"] or u["R"] or u["B"] or u["T"])],
                            key=lambda u: u["area"], reverse=True)
                        while droppable:
                            s_a, f_w, f_h = content_sums(content)
                            d = length * fill_factor - f_w
                            need = s_a / d if d > eps else h_cap * 2
                            if max(need, f_h) <= h_cap:
                                break
                            drop = droppable.pop(0)
                            content.remove(drop)
                            pool.insert(0, drop)
                        soft_area, fixed_w_sum, max_fixed_h = content_sums(content)
                        if max_fixed_h > h_cap:
                            h_row = max_fixed_h
                            grew_past_cap = True
                elif h_needed > 0.72 * h_cap:
                    # Close enough to the band edge: stretch and absorb the
                    # slack horizontally so no sliver remains above the row.
                    h_row = h_cap
                else:
                    h_row = h_needed

                if grew_past_cap:
                    # Recompute intervals conservatively over the true span
                    # (single pass, no feedback loop).
                    iv = free_intervals(row_y, row_y + h_row)
                    length = sum(b - a for a, b in iv)
                    if not iv or length <= eps:
                        for u in content:
                            if u["B"]:
                                pending_b.append(u)
                            elif u["T"]:
                                top_units.append(u)
                            else:
                                pool.insert(0, u)
                        current_y = band_top
                        continue
                    denom = length * fill_factor - fixed_w_sum
                    if denom > eps:
                        h_row = max(soft_area / denom,
                                    min_fit_height(content, max(b - a for a, b in iv)),
                                    max_fixed_h, h_row)

                # Top up stretched rows so the taller row stays filled.
                if not is_top:
                    deficit = (length * fill_factor - fixed_w_sum) * h_row - soft_area
                    guard = 0
                    while deficit > 0.6 * med_area and pool and guard < 12:
                        guard += 1
                        picked = None
                        for u in pool[:10]:
                            if u["area"] <= deficit * 1.06 and not (u["L"] or u["R"]):
                                if any(c[0] == "fixed" and c[3] > h_row - 1e-7 for c in u["cells"]):
                                    continue
                                if picked is None or u["area"] > picked["area"]:
                                    picked = u
                        if picked is None:
                            break
                        pool.remove(picked)
                        content.append(picked)
                        deficit -= picked["area"]
                        soft_area += sum(areas[c[1]] for c in picked["cells"] if c[0] == "soft")
                        fixed_w_sum += sum(c[2] for c in picked["cells"] if c[0] == "fixed")

                total_w = sum(unit_width(u, h_row) for u in content)

                # Pop uncoded units (smallest first) back into the pool until
                # the content fits the available length.
                if total_w > length - 1e-9:
                    droppable = sorted(
                        [u for u in content
                         if not (u["L"] or u["R"] or u["B"] or u["T"])],
                        key=lambda u: u["area"])
                    while total_w > length - 1e-9 and droppable and not is_top:
                        drop = droppable.pop(0)
                        content.remove(drop)
                        total_w -= unit_width(drop, h_row)
                        pool.insert(0, drop)
                    if total_w > length - 1e-9:
                        # Grow height instead (last resort / top row).
                        denom = length * 0.999 - fixed_w_sum
                        if denom > eps:
                            soft_now = sum(areas[c[1]] for u in content for c in u["cells"] if c[0] == "soft")
                            new_h = max(h_row, soft_now / denom, max_fixed_h)
                            if new_h > h_row + eps:
                                h_row = new_h
                                iv = free_intervals(row_y, row_y + h_row)
                                length = sum(b - a for a, b in iv)
                                total_w = sum(unit_width(u, h_row) for u in content)

                # Order: left units first, right unit last, middle by tx.
                left_first = [u for u in content if u["L"]]
                right_last = [u for u in content if u["R"] and not u["L"]]
                middle = [u for u in content if not u["L"] and not u["R"]]
                middle.sort(key=lambda u: u["tx"])
                ordered = left_first + middle + right_last

                # Final interval refresh so packing always matches the true
                # row span (h_row may have grown above).
                iv = free_intervals(row_y, row_y + h_row)
                length = sum(b - a for a, b in iv)
                if not iv or length <= eps:
                    for u in content:
                        if u["B"]:
                            pending_b.append(u)
                        elif u["T"]:
                            top_units.append(u)
                        else:
                            pool.insert(0, u)
                    current_y = band_top
                    continue

                rects, fitted, leftover, pockets = pack_row(ordered, h_row, iv, row_y)
                if leftover:
                    # Fragmented row: retry widest-first, keep the better fit.
                    ordered_ffd = (left_first
                                   + sorted(middle, key=lambda u: unit_width(u, h_row), reverse=True)
                                   + right_last)
                    rects2, fitted2, leftover2, pockets2 = pack_row(ordered_ffd, h_row, iv, row_y)
                    if sum(u["area"] for u in leftover2) < sum(u["area"] for u in leftover):
                        rects, fitted, leftover, pockets = rects2, fitted2, leftover2, pockets2
                if leftover and fitted and not is_top:
                    # The row was sized for content that leaked out; shrink it
                    # to the fitted content and re-pack so no dead band stays.
                    s_f, f_w_f, f_h_f = content_sums(fitted)
                    iv_short = free_intervals(row_y, row_y + max(f_h_f, eps, h_target * 0.4))
                    len_short = sum(b - a for a, b in iv_short)
                    d2 = len_short * fill_factor - f_w_f
                    if d2 > eps:
                        h2 = max(s_f / d2, f_h_f, h_target * 0.4)
                        if h2 < h_row - 1e-6:
                            iv2 = free_intervals(row_y, row_y + h2)
                            len2 = sum(b - a for a, b in iv2)
                            if iv2 and len2 > eps:
                                ordered2 = [u for u in ordered if u in fitted]
                                rects3, fitted3, leftover3, pockets3 = pack_row(ordered2, h2, iv2, row_y)
                                if not leftover3:
                                    rects, fitted, pockets = rects3, fitted3, pockets3
                                    h_row = h2
                                    iv = iv2
                                    length = len2
                if os_mod.environ.get("FLOORSET_FRAME_DEBUG", "") == "2" and leftover:
                    print(f"        pack: h={h_row:.2f} iv={[(round(a,1),round(b,1)) for a,b in iv]} "
                          f"fit={len(fitted)} left={[(u['kind'], round(u['area']), round(unit_width(u, h_row),1)) for u in leftover]}")

                if not fitted:
                    # Nothing fits this row; recycle and advance to the next
                    # band edge so the loop keeps making progress.
                    for u in leftover:
                        if u["B"]:
                            pending_b.append(u)
                        elif u["T"]:
                            top_units.append(u)
                        else:
                            pool.insert(0, u)
                    current_y = band_top
                    continue

                # Commit.
                for b, rect in rects.items():
                    positions[b] = rect
                for u in leftover:
                    if is_top:
                        failed_units.append(u)
                    elif u["B"]:
                        pending_b.append(u)
                    elif u["T"]:
                        top_units.append(u)
                    else:
                        pool.insert(0, u)

                # Stack units: re-place members vertically inside the unit's
                # footprint so each touches the row's start edge.
                for u in fitted:
                    if u["kind"] != "stack":
                        continue
                    xs = min(positions[b][0] for b in u["members"])
                    stack_w = sum(areas[b] for b in u["members"]) / h_row
                    yy = row_y
                    for b in u["members"]:
                        bh = areas[b] / stack_w
                        positions[b] = (xs, yy, stack_w, bh)
                        yy += bh

                # Fill pockets above fixed cells with best-fitting soft
                # singles pulled from the pool.
                for px, py, pw_, ph_ in pockets:
                    rem_h = ph_
                    guard = 0
                    while rem_h * pw_ > 0.25 * med_area and guard < 3:
                        guard += 1
                        patch = take_patch(rem_h * pw_, rem_h * pw_ * (1.0 - 1e-9))
                        if patch is None:
                            break
                        b = patch["members"][0]
                        bh = areas[b] / pw_
                        positions[b] = (px, py + (ph_ - rem_h), pw_, bh)
                        rem_h -= bh

                if os_mod.environ.get("FLOORSET_FRAME_DEBUG", "") == "2":
                    ws_row = length * h_row - sum(u["area"] for u in fitted)
                    print(f"      row {row_no}: y={row_y:.1f} h={h_row:.1f} len={length:.1f} "
                          f"units={len(content)} placed={len(fitted)} totw={total_w:.1f} "
                          f"ws={ws_row:.0f} top={is_top} bot={is_bottom} maxfh={max_fixed_h:.1f}")
                current_y = row_y + h_row

            # Anything that never fit: simple shelf above everything.
            leftover_units = []
            seen_left = set()
            for u in failed_units + pending_b + top_units + left_queue + right_queue + pool:
                if id(u) in seen_left:
                    continue
                seen_left.add(id(u))
                if any(positions[b] is None for b in u["members"]):
                    leftover_units.append(u)
            if leftover_units:
                shelf_y = max(current_y, y_hi)
                cursor = x_lo
                shelf_h = max(h_target, max(
                    (cell[3] for u in leftover_units for cell in u["cells"] if cell[0] == "fixed"),
                    default=h_target))
                row_max_h = shelf_h
                for u in leftover_units:
                    # Oversized units get a taller cell so they stay inside
                    # the frame width.
                    h_u = max(shelf_h, u["area"] / max(0.95 * (x_hi - x_lo), eps))
                    uw = unit_width(u, h_u)
                    if cursor > x_lo and cursor + uw > x_hi + 1e-9:
                        # Wrap to a new shelf row instead of growing sideways.
                        cursor = x_lo
                        shelf_y += row_max_h
                        row_max_h = shelf_h
                    row_max_h = max(row_max_h, h_u)
                    for cell in u["cells"]:
                        if cell[0] == "soft":
                            b = cell[1]
                            if positions[b] is None:
                                cw = areas[b] / h_u
                                positions[b] = (cursor, shelf_y, cw, h_u)
                                cursor += cw
                        else:
                            b, fw, fh = cell[1], cell[2], cell[3]
                            if positions[b] is None:
                                positions[b] = (cursor, shelf_y, fw, fh)
                                cursor += fw
            if any(p is None for p in positions):
                return None
            return positions

        best_positions = None
        best_cost = float("inf")
        flow_scores = {}
        candidate_pool = []

        def remember_candidate(name, positions, cost):
            summary = soft_violation_summary(positions)
            candidate_pool.append({
                "name": name,
                "positions": [tuple(p) for p in positions],
                "cost": cost,
                "summary": summary,
            })
            candidate_pool.sort(key=lambda item: (item["cost"], item["summary"]["total"], item["summary"]["boundary_distance"]))
            del candidate_pool[18:]

        def consider(name, positions):
            nonlocal best_positions, best_cost
            if not flow_enabled(name):
                return
            positions = normalize_if_safe(positions)
            cost = full_cost(positions)
            flow_scores[name] = min(flow_scores.get(name, float("inf")), cost)
            if hard_feasible(positions):
                remember_candidate(name, positions, cost)
                if (name.startswith("trained_prior") or name.startswith("soft_prior")) and best_positions is not None and cost > 0.94 * best_cost:
                    return
                if cost < best_cost:
                    best_cost = cost
                    best_positions = positions

        if flow_enabled("shelf_area"):
            consider("shelf_area", legalize(make_draft(order_area), order_area, try_ratios=False, candidate_limit=(16 if block_count > 80 else 28)))
        if flow_enabled("shelf_importance"):
            consider("shelf_importance", legalize(make_draft(order_importance), order_importance, try_ratios=False, candidate_limit=(16 if block_count > 80 else 28)))
        if flow_enabled("ml_greedy"):
            consider("ml_greedy", legalize(make_draft(order_importance), order_importance, try_ratios=True, candidate_limit=(16 if block_count > 80 else 28)))
        if flow_enabled("ml_cluster_greedy"):
            consider("ml_cluster_greedy", legalize(make_draft(order_cluster), order_cluster, try_ratios=True, candidate_limit=(14 if block_count > 80 else 24)))
        if prior_loaded and prior_kind == "soft" and flow_enabled("soft_prior_order"):
            consider("soft_prior_order", legalize(make_draft(order_prior, prior_ratio_map), order_prior, ratio_map=prior_ratio_map, try_ratios=False, candidate_limit=(18 if block_count > 80 else 30)))
        if prior_loaded and prior_kind == "soft" and flow_enabled("soft_prior_cluster"):
            consider("soft_prior_cluster", legalize(make_draft(order_cluster, prior_ratio_map), order_cluster, ratio_map=prior_ratio_map, try_ratios=False, candidate_limit=(16 if block_count > 80 else 28)))
        if prior_loaded and prior_kind != "soft" and flow_enabled("trained_prior_order"):
            consider("trained_prior_order", legalize(make_draft(order_prior, prior_ratio_map), order_prior, ratio_map=prior_ratio_map, try_ratios=False, candidate_limit=(18 if block_count > 80 else 30)))
        if prior_loaded and prior_kind != "soft" and flow_enabled("trained_prior_cluster"):
            consider("trained_prior_cluster", legalize(make_draft(order_cluster, prior_ratio_map), order_cluster, ratio_map=prior_ratio_map, try_ratios=False, candidate_limit=(16 if block_count > 80 else 28)))

        if flow_enabled("fd_init"):
            base_draft = make_draft(order_importance)
            fd_draft = force_directed_draft(base_draft)
            consider("fd_init", legalize(fd_draft, order_importance, try_ratios=True, candidate_limit=(18 if block_count > 80 else 30)))
        if flow_enabled("fd_cluster"):
            base_draft = make_draft(order_cluster, prior_ratio_map if prior_loaded else None)
            fd_draft = force_directed_draft(base_draft)
            consider("fd_cluster", legalize(fd_draft, order_cluster, ratio_map=prior_ratio_map if prior_loaded else None, try_ratios=True, candidate_limit=(16 if block_count > 80 else 28)))

        if flow_enabled("frame_rows"):
            frame_debug = os_mod.environ.get("FLOORSET_FRAME_DEBUG", "") == "1"
            for fr_scale in (1.0, 1.4, 0.75):
                try:
                    fr_positions = build_frame_rows(fr_scale)
                except Exception:
                    if frame_debug:
                        import traceback
                        traceback.print_exc()
                    fr_positions = None
                if fr_positions is not None:
                    if frame_debug:
                        summary = soft_violation_summary(fr_positions)
                        bb = bbox_bounds(fr_positions)
                        print(f"    [frame s={fr_scale}] feas={hard_feasible(fr_positions)} "
                              f"cost={full_cost(fr_positions):.1f} bbox=({bb[0]:.1f},{bb[1]:.1f},{bb[2]:.1f},{bb[3]:.1f}) "
                              f"area={(bb[2]-bb[0])*(bb[3]-bb[1]):.0f}/{total_area:.0f} "
                              f"bnd={summary['boundary']} grp={summary['grouping']} mib={summary['mib']}")
                        for i_ in range(block_count):
                            for j_ in range(i_ + 1, block_count):
                                if rectangles_overlap(fr_positions[i_], fr_positions[j_]):
                                    print(f"      OVERLAP {i_}{'P' if fixed_xy[i_] else ''} {fr_positions[i_]} "
                                          f"vs {j_}{'P' if fixed_xy[j_] else ''} {fr_positions[j_]}")
                        for vb in summary["boundary_blocks"]:
                            x, y, w, h = fr_positions[vb]
                            print(f"      bndV blk={vb} code={boundary_codes[vb]} rect=({x:.1f},{y:.1f},{w:.1f},{h:.1f}) "
                                  f"fixdim={fixed_dim[vb]} prepl={fixed_xy[vb]} clust={cluster_ids[vb]} mib={mib_ids[vb]}")
                    consider(f"frame_rows_s{fr_scale}", fr_positions)

        def flatten_genome(cluster_seq, inner_orders):
            order = []
            for cid in cluster_seq:
                order.extend(inner_orders[cid])
            return order

        def random_ratio_map(prob=0.35):
            result = {}
            grouped = set()
            for group in mib_groups_for_ratio.values():
                if mib_group_can_share_ratio(group) and random.random() < prob:
                    rid = random.randrange(len(ratio_candidates))
                    for b in group:
                        result[b] = rid
                        grouped.add(b)
            for i in range(block_count):
                if i not in grouped and not fixed_dim[i] and random.random() < prob:
                    result[i] = random.randrange(len(ratio_candidates))
            return enforce_mib_ratio_map(result)

        def soft_violation_sets(positions):
            summary = soft_violation_summary(positions)
            repairable_mibs = []
            for group in summary["broken_mibs"]:
                if any(not fixed_dim[b] and not fixed_xy[b] for b in group):
                    repairable_mibs.append(group)
            return list(summary["boundary_blocks"]), list(summary["broken_clusters"]), repairable_mibs

        def soft_repair_order(positions, mode="mixed"):
            boundary_blocks, broken_clusters, broken_mibs = soft_violation_sets(positions)
            chosen = []
            seen_blocks = set()

            def add_block(block):
                if 0 <= block < block_count and block not in seen_blocks:
                    chosen.append(block)
                    seen_blocks.add(block)

            boundary_sorted = sorted(
                boundary_blocks,
                key=lambda b: (max(prior_boundary_bits[b]) if prior_boundary_bits[b] else 0.0, importance[b], areas[b]),
                reverse=True,
            )
            for b in boundary_sorted:
                add_block(b)

            cluster_ranked = sorted(
                broken_clusters,
                key=lambda group: (len(group), sum(importance[b] for b in group)),
                reverse=True,
            )
            for group in cluster_ranked:
                group_order = sorted(
                    group,
                    key=lambda b: (sum(score for _, score in prior_touch_adj[b]), prior_order_bonus[b], importance[b]),
                    reverse=True,
                )
                for b in group_order:
                    add_block(b)
                for b in group_order[:4]:
                    for nbr, _score in prior_touch_adj[b][:4]:
                        add_block(nbr)

            for group in broken_mibs:
                for b in sorted(group, key=lambda x: prior_order_bonus[x], reverse=True):
                    add_block(b)

            if mode == "prior":
                filler = order_prior
            elif mode == "cluster":
                filler = order_cluster or order_prior
            else:
                filler = sorted(range(block_count), key=lambda b: (b not in boundary_blocks, -importance[b], -prior_order_bonus[b]))
            for b in filler:
                add_block(b)
            return chosen

        def make_soft_repair_draft(base_positions, mode="horizontal"):
            draft = [tuple(p) for p in base_positions]
            if not draft or any(p is None for p in draft):
                return draft
            min_x, min_y, max_x, max_y = bbox_bounds(draft)
            boundary_blocks, broken_clusters, _broken_mibs = soft_violation_sets(draft)

            for b in boundary_blocks:
                if fixed_xy[b]:
                    continue
                x, y, w, h = draft[b]
                code = boundary_codes[b]
                if code & 1:
                    x = min_x
                if code & 2:
                    x = max_x - w
                if code & 8:
                    y = min_y
                if code & 4:
                    y = max_y - h
                draft[b] = (x, y, w, h)

            for group in broken_clusters:
                movable_group = [b for b in group if not fixed_xy[b]]
                if len(movable_group) <= 1:
                    continue
                ordered = sorted(
                    movable_group,
                    key=lambda b: (sum(score for _, score in prior_touch_adj[b]), prior_order_bonus[b], importance[b]),
                    reverse=True,
                )
                anchor = ordered[0]
                ax, ay, aw, ah = draft[anchor]
                cursor_x = ax
                cursor_y = ay
                row_h = ah
                for idx, b in enumerate(ordered):
                    x, y, w, h = draft[b]
                    if idx == 0:
                        draft[b] = (ax, ay, w, h)
                        continue
                    if mode == "vertical":
                        cursor_y += draft[ordered[idx - 1]][3]
                        draft[b] = (ax, cursor_y, w, h)
                    elif mode == "snake" and idx % 2:
                        cursor_y += row_h
                        draft[b] = (ax, cursor_y, w, h)
                        row_h = h
                    else:
                        cursor_x += draft[ordered[idx - 1]][2]
                        draft[b] = (cursor_x, ay, w, h)
                        row_h = max(row_h, h)
            return draft

        def active_repair_stages():
            raw = os_mod.environ.get("FLOORSET_SOFT_REPAIR_STAGES", "mib,boundary,cluster,gravity").strip().lower()
            if raw in ("", "none", "off", "0"):
                return []
            allowed = {"mib", "boundary", "cluster", "gravity"}
            stages = [item.strip() for item in raw.split(",") if item.strip()]
            return [stage for stage in stages if stage in allowed]

        def repair_limit_default():
            try:
                return int(os_mod.environ.get("FLOORSET_SOFT_REPAIR_LIMIT", "80" if block_count >= 90 else "34"))
            except Exception:
                return 80 if block_count >= 90 else 34

        def repair_better(base, candidate, allow_soft_margin=True):
            candidate = normalize_if_safe(candidate)
            if not hard_feasible(candidate):
                return False
            base_summary = soft_violation_summary(base)
            cand_summary = soft_violation_summary(candidate)
            base_cost = full_cost(base)
            cand_cost = full_cost(candidate)
            margin = (0.08 if allow_soft_margin else 0.0) * abs(base_cost) + mean_side
            if cand_summary["total"] < base_summary["total"]:
                return cand_cost <= base_cost + margin
            if cand_summary["total"] == base_summary["total"]:
                if cand_summary["boundary_distance"] + 1e-6 < base_summary["boundary_distance"]:
                    return cand_cost <= base_cost + 0.5 * margin
                return cand_cost + 1e-9 < base_cost
            return cand_cost + 1e-9 < base_cost and cand_summary["total"] <= base_summary["total"]

        def choose_repair_candidate(base, candidates):
            best = [tuple(p) for p in base]
            for candidate in candidates:
                if candidate is None:
                    continue
                candidate = normalize_if_safe(candidate)
                if repair_better(best, candidate):
                    best = [tuple(p) for p in candidate]
            return best

        def bounds_for_blocks(positions, blocks):
            rects = [positions[b] for b in blocks if positions[b] is not None]
            if not rects:
                return bbox_bounds(positions)
            return (
                min(p[0] for p in rects),
                min(p[1] for p in rects),
                max(p[0] + p[2] for p in rects),
                max(p[1] + p[3] for p in rects),
            )

        def bounds_excluding(positions, skip):
            blocks = [i for i in range(block_count) if i != skip]
            return bounds_for_blocks(positions, blocks)

        def overlap_except(block, rect, positions):
            return overlap_area_with(rect, positions, [j for j in range(block_count) if j != block])

        def repair_mib_same_shape(base_positions):
            draft = [tuple(p) for p in base_positions]
            changed = False
            for group in groups_from_ids(mib_ids).values():
                if len(group) <= 1:
                    continue
                shape_stats = {}
                fixed_shapes = set()
                for b in group:
                    shape = (round(draft[b][2], 6), round(draft[b][3], 6))
                    count, score = shape_stats.get(shape, (0, 0.0))
                    shape_stats[shape] = (count + 1, score + importance[b])
                    if fixed_dim[b] or fixed_xy[b]:
                        fixed_shapes.add(shape)
                if len(fixed_shapes) > 1:
                    continue
                if fixed_shapes:
                    target_w, target_h = next(iter(fixed_shapes))
                else:
                    target_w, target_h = max(shape_stats.items(), key=lambda item: (item[1][0], item[1][1]))[0]

                feasible = True
                for b in group:
                    if fixed_dim[b]:
                        if abs(draft[b][2] - target_w) > hard_eps or abs(draft[b][3] - target_h) > hard_eps:
                            feasible = False
                            break
                    elif abs(target_w * target_h - areas[b]) / max(areas[b], 1e-9) > 0.01:
                        feasible = False
                        break
                if not feasible:
                    continue

                for b in group:
                    if fixed_xy[b] or fixed_dim[b]:
                        continue
                    x, y, w, h = draft[b]
                    cx = x + w / 2.0
                    cy = y + h / 2.0
                    draft[b] = (cx - target_w / 2.0, cy - target_h / 2.0, target_w, target_h)
                    changed = True

            if not changed:
                return base_positions
            order = soft_repair_order(draft, mode="mixed")
            legalized = legalize(draft, order, ratio_map=None, try_ratios=False, candidate_limit=repair_limit_default())
            return choose_repair_candidate(base_positions, [draft, legalized])

        def repair_boundary_outliers(base_positions):
            best = [tuple(p) for p in base_positions]
            summary = soft_violation_summary(best)
            blocks = sorted(
                summary["boundary_blocks"],
                key=lambda b: (importance[b], areas[b]),
                reverse=True,
            )
            for b in blocks:
                if fixed_xy[b]:
                    continue
                x, y, w, h = best[b]
                code = boundary_codes[b]
                ref_bounds = [bbox_bounds(best), bounds_excluding(best, b)]
                candidates = []
                for min_x, min_y, max_x, max_y in ref_bounds:
                    x_targets = []
                    y_targets = []
                    if code & 1:
                        x_targets.append(min_x)
                    if code & 2:
                        x_targets.append(max_x - w)
                    if not x_targets:
                        x_targets.extend([x, min_x, max_x - w])
                    if code & 8:
                        y_targets.append(min_y)
                    if code & 4:
                        y_targets.append(max_y - h)
                    if not y_targets:
                        y_targets.extend([y, min_y, max_y - h])
                    for nx in x_targets:
                        for ny in y_targets:
                            if finite(nx) and finite(ny):
                                candidates.append((float(nx), float(ny), w, h))

                seen = set()
                for rect in candidates:
                    key = tuple(round(v, 5) for v in rect)
                    if key in seen:
                        continue
                    seen.add(key)
                    if overlap_except(b, rect, best) > eps:
                        continue
                    candidate = best[:]
                    candidate[b] = rect
                    if repair_better(best, candidate):
                        best = [tuple(p) for p in candidate]

            draft = make_soft_repair_draft(best, mode="horizontal")
            order = soft_repair_order(draft, mode="mixed")
            legalized = legalize(draft, order, ratio_map=prior_ratio_map, try_ratios=False, candidate_limit=repair_limit_default())
            return choose_repair_candidate(base_positions, [best, legalized])

        def repair_cluster_edge_abut(base_positions):
            best = [tuple(p) for p in base_positions]

            def translated_component(positions, comp, dx, dy):
                if any(fixed_xy[b] for b in comp):
                    return None
                candidate = [tuple(p) for p in positions]
                for b in comp:
                    x, y, w, h = candidate[b]
                    candidate[b] = (x + dx, y + dy, w, h)
                return candidate

            for group in soft_violation_summary(best)["broken_clusters"]:
                for _pass in range(2):
                    comps = edge_components(group, best)
                    if len(comps) <= 1:
                        break
                    anchor = max(comps, key=lambda comp: (any(fixed_xy[b] for b in comp), sum(importance[b] for b in comp), sum(areas[b] for b in comp)))
                    anchor_bounds = bounds_for_blocks(best, anchor)
                    moved = False
                    for comp in sorted(comps, key=lambda comp: sum(importance[b] for b in comp), reverse=True):
                        if comp == anchor or any(fixed_xy[b] for b in comp):
                            continue
                        cmin_x, cmin_y, cmax_x, cmax_y = bounds_for_blocks(best, comp)
                        amin_x, amin_y, amax_x, amax_y = anchor_bounds
                        shifts = [
                            (amax_x - cmin_x, amin_y - cmin_y),
                            (amin_x - cmax_x, amin_y - cmin_y),
                            (amin_x - cmin_x, amax_y - cmin_y),
                            (amin_x - cmin_x, amin_y - cmax_y),
                        ]
                        anchor_blocks = sorted(anchor, key=lambda b: importance[b], reverse=True)[:4]
                        comp_blocks = sorted(comp, key=lambda b: importance[b], reverse=True)[:4]
                        for a in anchor_blocks:
                            ax, ay, aw, ah = best[a]
                            for b in comp_blocks:
                                bx, by, bw, bh = best[b]
                                shifts.extend([
                                    (ax + aw - bx, ay - by),
                                    (ax - (bx + bw), ay - by),
                                    (ax - bx, ay + ah - by),
                                    (ax - bx, ay - (by + bh)),
                                ])
                        seen = set()
                        for dx, dy in sorted(shifts, key=lambda item: abs(item[0]) + abs(item[1])):
                            key = (round(dx, 5), round(dy, 5))
                            if key in seen:
                                continue
                            seen.add(key)
                            candidate = translated_component(best, comp, dx, dy)
                            if candidate is not None and repair_better(best, candidate):
                                best = [tuple(p) for p in candidate]
                                moved = True
                                break
                        if moved:
                            break
                    if not moved:
                        break

            draft = make_soft_repair_draft(best, mode="snake")
            order = soft_repair_order(draft, mode="cluster")
            legalized = legalize(draft, order, ratio_map=prior_ratio_map, try_ratios=False, candidate_limit=repair_limit_default())
            return choose_repair_candidate(base_positions, [best, legalized])

        def repair_gravity_compact(base_positions):
            current = [tuple(p) for p in base_positions]
            if not current or any(p is None for p in current):
                return base_positions
            floor_x, floor_y, _max_x, _max_y = bbox_bounds(current) if preplaced_positions else (0.0, 0.0, 0.0, 0.0)

            def compact_axis(positions, axis):
                result = [tuple(p) for p in positions]
                order = sorted(
                    [b for b in range(block_count) if not fixed_xy[b]],
                    key=lambda b: (result[b][1], result[b][0]) if axis == "y" else (result[b][0], result[b][1]),
                )
                for b in order:
                    x, y, w, h = result[b]
                    if axis == "y":
                        target = floor_y
                        for j, other in enumerate(result):
                            if j == b:
                                continue
                            ox, oy, ow, oh = other
                            if oy + oh <= y + eps and min(x + w, ox + ow) - max(x, ox) > eps:
                                target = max(target, oy + oh)
                        if target >= y - eps:
                            continue
                        candidate = result[:]
                        candidate[b] = (x, target, w, h)
                    else:
                        target = floor_x
                        for j, other in enumerate(result):
                            if j == b:
                                continue
                            ox, oy, ow, oh = other
                            if ox + ow <= x + eps and min(y + h, oy + oh) - max(y, oy) > eps:
                                target = max(target, ox + ow)
                        if target >= x - eps:
                            continue
                        candidate = result[:]
                        candidate[b] = (target, y, w, h)
                    if hard_feasible(candidate):
                        old_summary = soft_violation_summary(result)
                        new_summary = soft_violation_summary(candidate)
                        if new_summary["total"] <= old_summary["total"]:
                            result = [tuple(p) for p in candidate]
                return result

            for _ in range(3):
                current = compact_axis(current, "y")
                current = compact_axis(current, "x")
            return choose_repair_candidate(base_positions, [current])

        def run_soft_final_repair(seed_positions, tag="soft_final_repair"):
            if not flow_enabled("soft_final_repair") or seed_positions is None:
                return seed_positions
            stages = active_repair_stages()
            if not stages:
                return seed_positions
            current = [tuple(p) for p in seed_positions]
            for stage in stages:
                if stage == "mib":
                    candidate = repair_mib_same_shape(current)
                elif stage == "boundary":
                    candidate = repair_boundary_outliers(current)
                elif stage == "cluster":
                    candidate = repair_cluster_edge_abut(current)
                elif stage == "gravity":
                    candidate = repair_gravity_compact(current)
                else:
                    continue
                if hard_feasible(candidate):
                    consider(f"{tag}_{stage}", candidate)
                    if repair_better(current, candidate):
                        current = [tuple(p) for p in candidate]
            consider(tag, current)
            return current

        def repair_candidate_pool(tag="soft_final_repair_pool"):
            if not flow_enabled("soft_final_repair") or not active_repair_stages():
                return
            snapshot = list(candidate_pool)
            selected = []
            seen = set()
            by_cost = sorted(snapshot, key=lambda item: item["cost"])[:3]
            by_soft = sorted(snapshot, key=lambda item: (item["summary"]["total"], item["summary"]["boundary_distance"]), reverse=True)[:2]
            for item in by_cost + by_soft:
                key = (item["name"], round(item["cost"], 4))
                if key not in seen:
                    selected.append(item)
                    seen.add(key)
            for item in selected:
                run_soft_final_repair(item["positions"], f"{tag}_{item['name']}")

        def ga_rescue_needed(positions):
            if positions is None or os_mod.environ.get("FLOORSET_GA_RESCUE", "0").strip().lower() not in ("1", "true", "yes", "on"):
                return False
            summary = soft_violation_summary(positions)
            if summary["relative"] >= 0.50:
                return True
            if summary["total"] >= 3:
                return True
            if block_count >= 106 and summary["relative"] >= 0.35:
                return True
            if summary["boundary"] >= 2 and summary["boundary_distance"] >= 1.5 * mean_side:
                return True
            return False

        def ga_rescue_severe(positions):
            if positions is None or block_count < 90:
                return False
            summary = soft_violation_summary(positions)
            return summary["relative"] >= 0.60 or summary["total"] >= 5

        cluster_count = len(clusters)

        def run_cluster_ga(tag="cluster_ga", forced_pop=None, forced_gens=None, forced_elite=None, seed_offset=0):
            if cluster_count <= 0 or not flow_enabled("cluster_ga"):
                return
            saved_state = None
            if seed_offset:
                saved_state = random.getstate()
                random.seed(seed_base + int(block_count) * 131 + seed_offset)
            try:
                base_seq = list(range(cluster_count))
                base_inner = {cid: clusters[cid][:] for cid in range(cluster_count)}
                population = []
                population.append((base_seq[:], {cid: base_inner[cid][:] for cid in base_inner}, {}))
                population.append((base_seq[:], {cid: sorted(base_inner[cid], key=lambda b: areas[b], reverse=True) for cid in base_inner}, random_ratio_map(0.2)))
                if prior_loaded:
                    prior_seq = sorted(base_seq, key=lambda cid: sum(prior_order_bonus[b] for b in base_inner[cid]) / max(len(base_inner[cid]), 1), reverse=True)
                    prior_inner = {cid: sorted(base_inner[cid], key=lambda b: prior_order_bonus[b], reverse=True) for cid in base_inner}
                    population.append((prior_seq, prior_inner, dict(prior_ratio_map)))
                if cluster_count > 1:
                    rev = base_seq[:]
                    rev.reverse()
                    population.append((rev, {cid: base_inner[cid][:] for cid in base_inner}, random_ratio_map(0.25)))

                if forced_pop is None or forced_gens is None:
                    try:
                        quality_min_blocks = int(os_mod.environ.get("FLOORSET_QUALITY_GA_MIN_BLOCKS", "1000000"))
                    except Exception:
                        quality_min_blocks = 1000000
                    if block_count >= quality_min_blocks:
                        default_pop, default_gens = 30, 30
                    elif block_count > 80:
                        default_pop, default_gens = 5, 2
                    elif block_count > 55:
                        default_pop, default_gens = 5, 2
                    else:
                        default_pop, default_gens = 8, 3
                else:
                    default_pop, default_gens = forced_pop, forced_gens

                if forced_pop is None:
                    try:
                        pop_size = max(1, int(os_mod.environ.get("FLOORSET_GA_POP", str(default_pop))))
                    except Exception:
                        pop_size = default_pop
                else:
                    pop_size = max(1, int(forced_pop))
                if forced_gens is None:
                    try:
                        generations = max(1, int(os_mod.environ.get("FLOORSET_GA_GENS", str(default_gens))))
                    except Exception:
                        generations = default_gens
                else:
                    generations = max(1, int(forced_gens))
                if forced_elite is None:
                    try:
                        elite_ratio = float(os_mod.environ.get("FLOORSET_GA_ELITE_RATIO", "0.05"))
                    except Exception:
                        elite_ratio = 0.05
                else:
                    elite_ratio = float(forced_elite)
                elite_ratio = min(0.80, max(0.05, elite_ratio))

                while len(population) < pop_size:
                    seq = base_seq[:]
                    random.shuffle(seq)
                    inner = {}
                    for cid in base_inner:
                        members = base_inner[cid][:]
                        if random.random() < 0.45:
                            random.shuffle(members)
                        inner[cid] = members
                    population.append((seq, inner, random_ratio_map(0.45)))

                elite_count = max(1, int(round(pop_size * elite_ratio)))
                for _gen in range(generations):
                    evaluated = []
                    for seq, inner, ratios in population:
                        order = flatten_genome(seq, inner)
                        draft = make_draft(order, ratios)
                        candidate = legalize(draft, order, ratio_map=ratios, try_ratios=False, candidate_limit=(12 if block_count > 80 else 20))
                        cost = full_cost(candidate)
                        evaluated.append((cost, seq, inner, ratios, candidate))
                        consider(tag, candidate)

                    evaluated.sort(key=lambda item: item[0])
                    next_population = [
                        (seq[:], {cid: inner[cid][:] for cid in inner}, dict(ratios))
                        for _, seq, inner, ratios, _ in evaluated[:elite_count]
                    ]
                    parents = evaluated[:max(1, min(len(evaluated), max(3, pop_size // 2)))]

                    while len(next_population) < pop_size:
                        _, seq_a, inner_a, ratios_a, _ = random.choice(parents)
                        _, seq_b, inner_b, ratios_b, _ = random.choice(parents)
                        if cluster_count <= 1:
                            child_seq = seq_a[:]
                        else:
                            cut1 = random.randrange(cluster_count)
                            cut2 = random.randrange(cut1, cluster_count)
                            middle = seq_a[cut1:cut2]
                            child_seq = [cid for cid in seq_b if cid not in middle]
                            child_seq[cut1:cut1] = middle
                        if cluster_count > 1 and random.random() < 0.45:
                            a, b = random.sample(range(cluster_count), 2)
                            child_seq[a], child_seq[b] = child_seq[b], child_seq[a]

                        child_inner = {}
                        for cid in range(cluster_count):
                            source = inner_a if random.random() < 0.5 else inner_b
                            members = source[cid][:]
                            if len(members) > 1 and random.random() < 0.35:
                                a, b = random.sample(range(len(members)), 2)
                                members[a], members[b] = members[b], members[a]
                            child_inner[cid] = members

                        child_ratios = dict(ratios_a if random.random() < 0.5 else ratios_b)
                        for _ in range(max(1, block_count // 20)):
                            if mib_groups_for_ratio and random.random() < 0.35:
                                group = random.choice(list(mib_groups_for_ratio.values()))
                                if mib_group_can_share_ratio(group):
                                    rid = random.randrange(len(ratio_candidates))
                                    for b in group:
                                        child_ratios[b] = rid
                                continue
                            b = random.randrange(block_count)
                            if not fixed_dim[b]:
                                child_ratios[b] = random.randrange(len(ratio_candidates))
                        child_ratios = enforce_mib_ratio_map(child_ratios)
                        next_population.append((child_seq, child_inner, child_ratios))
                    population = next_population
            finally:
                if saved_state is not None:
                    random.setstate(saved_state)

        def run_selective_ga_rescue():
            if best_positions is None or not ga_rescue_needed(best_positions):
                return
            run_cluster_ga("cluster_ga_rescue_30x30", forced_pop=30, forced_gens=30, forced_elite=0.05, seed_offset=10007)
            if best_positions is not None:
                run_soft_final_repair(best_positions, "soft_final_repair_ga30")
            if best_positions is not None and ga_rescue_severe(best_positions):
                run_cluster_ga("cluster_ga_rescue_50x100", forced_pop=50, forced_gens=100, forced_elite=0.05, seed_offset=20011)
                if best_positions is not None:
                    run_soft_final_repair(best_positions, "soft_final_repair_ga50")

        run_cluster_ga("cluster_ga")
        repair_candidate_pool("soft_final_repair_pre")
        if best_positions is not None:
            run_soft_final_repair(best_positions, "soft_final_repair")

        try:
            restarts = 0 if block_count > 80 else 1
            moves = 0 if block_count > 80 else 5
            for _ in range(restarts):
                widths = []
                heights = []
                ratio_map = random_ratio_map(0.5)
                for i in range(block_count):
                    ratio = ratio_candidates[ratio_map.get(i, 0) % len(ratio_candidates)]
                    w, h = block_dims(i, ratio)
                    widths.append(w)
                    heights.append(h)
                tree = BStarTree(block_count, widths, heights)
                best_tree_draft = tree.pack()
                best_tree_cost = full_cost(legalize(best_tree_draft, order_importance, candidate_limit=(12 if block_count > 80 else 20)))
                for _move in range(moves):
                    old = tree.copy()
                    movable_for_tree = [i for i in range(block_count) if not fixed_dim[i]]
                    relocatable_for_tree = [i for i in range(block_count) if not fixed_xy[i]]
                    if random.random() < 0.35 and movable_for_tree:
                        tree.move_rotate(random.choice(movable_for_tree))
                    elif relocatable_for_tree:
                        tree.move_delete_insert(random.choice(relocatable_for_tree))
                    else:
                        continue
                    draft = tree.pack()
                    cand = legalize(draft, order_importance, candidate_limit=(12 if block_count > 80 else 20))
                    cost = full_cost(cand)
                    if cost < best_tree_cost or random.random() < 0.05:
                        best_tree_cost = cost
                        best_tree_draft = draft
                    else:
                        tree = old
                consider("bstar_sa", legalize(best_tree_draft, order_importance, candidate_limit=(12 if block_count > 80 else 24)))
        except Exception:
            pass

        fallback = legalize(make_draft(order_area), order_area, try_ratios=False, candidate_limit=(20 if block_count > 80 else 40))
        if best_positions is None or not hard_feasible(best_positions):
            best_positions = fallback
            best_cost = full_cost(best_positions)

        move_stats = {
            "shift": [0, 0, 0.0],
            "swap": [0, 0, 0.0],
            "aspect": [0, 0, 0.0],
            "compact": [0, 0, 0.0],
            "cluster_shift": [0, 0, 0.0],
        }
        movable = [i for i in range(block_count) if not fixed_xy[i]]
        soft_movable = [i for i in movable if not fixed_dim[i]]
        cluster_members = [c for c in clusters if any(b in movable for b in c)]

        def choose_move():
            if random.random() < 0.22:
                return random.choice(list(move_stats.keys()))
            scored = []
            for name, (attempts, successes, avg_gain) in move_stats.items():
                rate = (successes + 1.0) / (attempts + 2.0)
                scored.append((rate * (1.0 + min(avg_gain, 1e6) / (abs(best_cost) + 1.0)), name))
            total = sum(max(score, 1e-9) for score, _ in scored)
            pick = random.random() * total
            acc = 0.0
            for score, name in scored:
                acc += max(score, 1e-9)
                if acc >= pick:
                    return name
            return scored[-1][1]

        def connectivity_centroid(block, positions):
            sx = sy = sw = 0.0
            for nbr, weight in b2b_adj[block]:
                x, y, w, h = positions[nbr]
                sx += (x + w / 2.0) * weight
                sy += (y + h / 2.0) * weight
                sw += weight
            for pin, weight in p2b_adj[block]:
                sx += tensor_value(pins_pos, pin, 0) * weight
                sy += tensor_value(pins_pos, pin, 1) * weight
                sw += weight
            if sw <= 0:
                return None
            return sx / sw, sy / sw

        def mutate_positions(positions, move):
            draft = list(positions)
            if not movable:
                return draft
            if move == "swap" and len(movable) >= 2:
                a, b = random.sample(movable, 2)
                ax, ay, aw, ah = draft[a]
                bx, by, bw, bh = draft[b]
                draft[a] = (bx, by, aw, ah)
                draft[b] = (ax, ay, bw, bh)
                return draft

            if move == "aspect" and soft_movable:
                b = random.choice(soft_movable)
                x, y, w, h = draft[b]
                cx = x + w / 2.0
                cy = y + h / 2.0
                nw, nh = block_dims(b, random.choice(ratio_candidates))
                draft[b] = (cx - nw / 2.0, cy - nh / 2.0, nw, nh)
                return draft

            if move == "cluster_shift" and cluster_members:
                group = random.choice(cluster_members)
                targets = [connectivity_centroid(b, draft) for b in group if b in movable]
                targets = [t for t in targets if t is not None]
                if not targets:
                    return draft
                tx = sum(t[0] for t in targets) / len(targets)
                ty = sum(t[1] for t in targets) / len(targets)
                centers = []
                for b in group:
                    if b in movable:
                        x, y, w, h = draft[b]
                        centers.append((x + w / 2.0, y + h / 2.0))
                if not centers:
                    return draft
                cx = sum(c[0] for c in centers) / len(centers)
                cy = sum(c[1] for c in centers) / len(centers)
                dx = 0.3 * (tx - cx)
                dy = 0.3 * (ty - cy)
                for b in group:
                    if b in movable:
                        x, y, w, h = draft[b]
                        draft[b] = (x + dx, y + dy, w, h)
                return draft

            b = random.choice(movable)
            x, y, w, h = draft[b]
            if move == "shift":
                target = connectivity_centroid(b, draft)
                if target is not None:
                    tx, ty = target
                    cx = x + w / 2.0
                    cy = y + h / 2.0
                    nx = cx + 0.35 * (tx - cx) - w / 2.0
                    ny = cy + 0.35 * (ty - cy) - h / 2.0
                    draft[b] = (nx, ny, w, h)
                return draft

            if move == "compact":
                draft[b] = (x - 0.25 * w, y - 0.25 * h, w, h)
            return draft

        local_iters = min(500, max(40, 4 * block_count))
        if block_count > 95:
            local_iters = min(local_iters, 25)
        elif block_count > 70:
            local_iters = min(local_iters, 60)

        current = best_positions
        current_cost = best_cost
        for _ in range(local_iters):
            move = choose_move()
            move_stats[move][0] += 1
            draft = mutate_positions(current, move)
            candidate = legalize(draft, order_cluster or order_importance, try_ratios=False, candidate_limit=(12 if block_count > 80 else 20))
            if not hard_feasible(candidate):
                continue
            cost = full_cost(candidate)
            if cost + 1e-9 < current_cost:
                gain = current_cost - cost
                current = candidate
                current_cost = cost
                move_stats[move][1] += 1
                old_avg = move_stats[move][2]
                successes = move_stats[move][1]
                move_stats[move][2] = old_avg + (gain - old_avg) / max(successes, 1)
                if cost < best_cost:
                    best_positions = candidate
                    best_cost = cost

        if best_positions is not None:
            run_soft_final_repair(best_positions, "soft_final_repair_post")
            run_selective_ga_rescue()

        # ------------------------------------------------------------------
        # Contest-cost re-ranking.  full_cost weighs wirelength, bbox area
        # and violations additively, which can prefer a low-wire sprawling
        # layout over a compact one even though the real score multiplies
        # relative gaps by exp(2·V_rel).  Re-rank the remembered candidates
        # with a proxy of the actual formula and continue with that winner;
        # the final refinement chain below is gated on the same proxy.
        # ------------------------------------------------------------------
        def candidate_wire(positions):
            wire = 0.0
            for i, j, weight in valid_b2b:
                xi, yi, wi, hi = positions[i]
                xj, yj, wj, hj = positions[j]
                wire += weight * (abs((xi + wi / 2.0) - (xj + wj / 2.0)) +
                                  abs((yi + hi / 2.0) - (yj + hj / 2.0)))
            for pin, block, weight in valid_p2b:
                x, y, w, h = positions[block]
                px = tensor_value(pins_pos, pin, 0)
                py = tensor_value(pins_pos, pin, 1)
                wire += weight * (abs((x + w / 2.0) - px) + abs((y + h / 2.0) - py))
            return wire

        proxy_pool = [item["positions"] for item in candidate_pool]
        if best_positions is not None:
            proxy_pool.append([tuple(p) for p in best_positions])

        feasible_pool = [p for p in proxy_pool if hard_feasible(p)]
        wire_base = None
        area_base = total_area * 1.035
        for p in feasible_pool:
            w = candidate_wire(p)
            if wire_base is None or w < wire_base:
                wire_base = w
        if wire_base is None or wire_base <= eps:
            wire_base = 1.0

        def contest_proxy(positions):
            if positions is None or len(positions) != block_count:
                return float("inf")
            wire = candidate_wire(positions)
            area = bbox_area(positions)
            summary = soft_violation_summary(positions)
            quality = 1.0 + 0.5 * (max(0.0, wire / wire_base - 1.0)
                                   + max(0.0, area / area_base - 1.0))
            return quality * math.exp(2.0 * summary["relative"])

        if feasible_pool:
            ranked = sorted(feasible_pool, key=contest_proxy)
            best_positions = [tuple(p) for p in ranked[0]]
            best_cost = contest_proxy(best_positions)
        elif best_positions is not None:
            best_cost = contest_proxy(best_positions)

        full_cost = contest_proxy

        # Rebuild the frame tiling seeded with the proxy-best layout's block
        # centres: projects the winning geometry onto the exact-area row
        # structure (zero-whitespace form of the same arrangement).
        if best_positions is not None and flow_enabled("frame_rows"):
            seed_tx = [best_positions[i][0] + best_positions[i][2] / 2.0 for i in range(block_count)]
            seed_ty = [best_positions[i][1] + best_positions[i][3] / 2.0 for i in range(block_count)]
            for fr_scale in (1.0, 1.3):
                try:
                    fr_seeded = build_frame_rows(fr_scale, target_override=(seed_tx, seed_ty))
                except Exception:
                    fr_seeded = None
                if fr_seeded is not None and hard_feasible(fr_seeded):
                    fr_seeded = normalize_if_safe(fr_seeded)
                    seeded_cost = contest_proxy(fr_seeded)
                    if seeded_cost + 1e-9 < best_cost:
                        best_positions = fr_seeded
                        best_cost = seeded_cost
                        flow_scores[f"frame_seeded_s{fr_scale}"] = seeded_cost

        def boundary_snap_pass(positions, max_iters=4):
            # Translate unsatisfied boundary blocks onto their required bbox
            # edge, keeping width/height unchanged. Accept only translations
            # that produce no overlap. Iterate because the bbox shifts.
            if positions is None or len(positions) != block_count:
                return positions
            if not any(boundary_codes):
                return positions
            current = [tuple(p) for p in positions]
            for _ in range(max_iters):
                min_x, min_y, max_x, max_y = bbox_bounds(current)
                bounds = (min_x, min_y, max_x, max_y)
                moves = []
                for i, code in enumerate(boundary_codes):
                    if not code or fixed_xy[i]:
                        continue
                    if touches_edge(current[i], bounds, code):
                        continue
                    x, y, w, h = current[i]
                    dx = dy = 0.0
                    if code & 1:
                        dx = min_x - x
                    elif code & 2:
                        dx = (max_x - w) - x
                    if code & 8:
                        dy = min_y - y
                    elif code & 4:
                        dy = (max_y - h) - y
                    moves.append((abs(dx) + abs(dy), i, dx, dy))
                if not moves:
                    break
                moves.sort()
                changed = False
                for _dist, i, dx, dy in moves:
                    if dx == 0.0 and dy == 0.0:
                        continue
                    x, y, w, h = current[i]
                    new_rect = (x + dx, y + dy, w, h)
                    blocked = False
                    for j in range(block_count):
                        if j == i:
                            continue
                        if rectangles_overlap(new_rect, current[j]):
                            blocked = True
                            break
                    if blocked:
                        continue
                    current[i] = new_rect
                    changed = True
                if not changed:
                    break
            return current

        def compact_pass(positions):
            # Slide interior (non-boundary, non-preplaced) blocks left and down
            # to remove dead space. Boundary-tagged blocks are anchors and never
            # move here; boundary_snap_pass handles them. Preserves dimensions,
            # so area / fixed_dim / MIB constraints are unchanged. Acceptance
            # filtered by full_cost downstream.
            if positions is None or len(positions) != block_count:
                return positions
            result = [tuple(p) for p in positions]
            free_blocks = [
                i for i in range(block_count)
                if not fixed_xy[i] and boundary_codes[i] == 0
            ]
            if not free_blocks:
                return result

            # Floors so we don't drag bbox edges past blocks that anchor them.
            # If no anchor on a side, use 0 (don't let coords go negative for
            # free blocks; normalize_if_safe handles small shifts otherwise).
            anchored_left = [result[i][0] for i in range(block_count) if boundary_codes[i] & 1]
            anchored_bottom = [result[i][1] for i in range(block_count) if boundary_codes[i] & 8]
            x_floor = min(anchored_left) if anchored_left else 0.0
            y_floor = min(anchored_bottom) if anchored_bottom else 0.0

            for _ in range(2):
                changed = False

                order = sorted(free_blocks, key=lambda i: result[i][0])
                for i in order:
                    x, y, w, h = result[i]
                    new_x = x_floor
                    for j in range(block_count):
                        if j == i:
                            continue
                        xj, yj, wj, hj = result[j]
                        y_overlap = min(y + h, yj + hj) - max(y, yj)
                        if y_overlap > eps and xj + wj <= x + 1e-9:
                            if xj + wj > new_x:
                                new_x = xj + wj
                    if new_x < x - 1e-9:
                        result[i] = (new_x, y, w, h)
                        changed = True

                order = sorted(free_blocks, key=lambda i: result[i][1])
                for i in order:
                    x, y, w, h = result[i]
                    new_y = y_floor
                    for j in range(block_count):
                        if j == i:
                            continue
                        xj, yj, wj, hj = result[j]
                        x_overlap = min(x + w, xj + wj) - max(x, xj)
                        if x_overlap > eps and yj + hj <= y + 1e-9:
                            if yj + hj > new_y:
                                new_y = yj + hj
                    if new_y < y - 1e-9:
                        result[i] = (x, new_y, w, h)
                        changed = True

                if not changed:
                    break
            return result

        def force_directed_refine(positions, iters=12, lr=0.4):
            # Cheap HPWL-only refinement: slide each free block toward its
            # connection-weighted centroid; reject moves that overlap. Boundary
            # blocks are anchored (any horizontal/vertical move could break
            # their edge constraint). Stops early if a sweep produces no moves.
            if positions is None or len(positions) != block_count:
                return positions
            current = [tuple(p) for p in positions]
            free_idx = [
                i for i in range(block_count)
                if not fixed_xy[i] and boundary_codes[i] == 0
            ]
            if not free_idx:
                return current
            for _ in range(iters):
                moved = False
                for i in free_idx:
                    x, y, w, h = current[i]
                    cx = x + w / 2.0
                    cy = y + h / 2.0
                    fx = fy = wsum = 0.0
                    for nbr, weight in b2b_adj[i]:
                        nx, ny, nw, nh = current[nbr]
                        fx += weight * (nx + nw / 2.0 - cx)
                        fy += weight * (ny + nh / 2.0 - cy)
                        wsum += weight
                    for pin, weight in p2b_adj[i]:
                        px = tensor_value(pins_pos, pin, 0)
                        py = tensor_value(pins_pos, pin, 1)
                        fx += weight * (px - cx)
                        fy += weight * (py - cy)
                        wsum += weight
                    if wsum <= 0:
                        continue
                    dx = lr * fx / wsum
                    dy = lr * fy / wsum
                    new_rect = (x + dx, y + dy, w, h)
                    blocked = False
                    for j in range(block_count):
                        if j == i:
                            continue
                        if rectangles_overlap(new_rect, current[j]):
                            blocked = True
                            break
                    if not blocked:
                        current[i] = new_rect
                        moved = True
                if not moved:
                    break
            return current

        def force_directed_horizontal_refine(positions, iters=18, lr=0.75):
            # HPWL cleanup for row-like layouts: slide only unconstrained free
            # blocks in x.  Boundary/cluster/MIB/fixed blocks stay anchored so
            # the soft-constraint surface is much narrower than the generic
            # force-directed pass.
            if positions is None or len(positions) != block_count:
                return positions
            current = [tuple(p) for p in positions]
            free_idx = [
                i for i in range(block_count)
                if (not fixed_xy[i] and not fixed_dim[i]
                    and boundary_codes[i] == 0 and cluster_ids[i] == 0 and mib_ids[i] == 0)
            ]
            if not free_idx:
                return current

            def local_x_cost(block, rect):
                x, _y, w, _h = rect
                cx = x + w / 2.0
                total = 0.0
                for nbr, weight in b2b_adj[block]:
                    nx, _ny, nw, _nh = current[nbr]
                    total += weight * abs(cx - (nx + nw / 2.0))
                for pin, weight in p2b_adj[block]:
                    px = tensor_value(pins_pos, pin, 0)
                    total += weight * abs(cx - px)
                return total

            for _ in range(iters):
                moved = False
                for i in free_idx:
                    x, y, w, h = current[i]
                    cx = x + w / 2.0
                    fx = wsum = 0.0
                    for nbr, weight in b2b_adj[i]:
                        nx, _ny, nw, _nh = current[nbr]
                        fx += weight * (nx + nw / 2.0 - cx)
                        wsum += weight
                    for pin, weight in p2b_adj[i]:
                        px = tensor_value(pins_pos, pin, 0)
                        fx += weight * (px - cx)
                        wsum += weight
                    if wsum <= 0:
                        continue
                    dx = max(-0.9 * w, min(0.9 * w, lr * fx / wsum))
                    if abs(dx) <= 1e-7:
                        continue
                    before = local_x_cost(i, current[i])
                    for scale in (1.0, 0.5, 0.25, 0.125):
                        new_rect = (x + dx * scale, y, w, h)
                        if local_x_cost(i, new_rect) >= before - 1e-9:
                            continue
                        blocked = False
                        for j in range(block_count):
                            if j == i:
                                continue
                            if rectangles_overlap(new_rect, current[j]):
                                blocked = True
                                break
                        if not blocked:
                            current[i] = new_rect
                            moved = True
                            break
                if not moved:
                    break
            return current

        def boundary_swap_repair(positions, max_iters=3):
            # For each unsatisfied right/top edge boundary block, try to:
            #   (1) translate to the matching edge if the slot is free
            #   (2) swap with a non-boundary block that's currently at the
            #       required edge but doesn't need to be there
            # Accept only feasible repairs; gated by full_cost downstream.
            if positions is None or len(positions) != block_count:
                return positions
            current = [tuple(p) for p in positions]

            def overlaps_any(rect, exclude):
                for j in range(block_count):
                    if j in exclude:
                        continue
                    if rectangles_overlap(rect, current[j]):
                        return True
                return False

            for _ in range(max_iters):
                min_x, min_y, max_x, max_y = bbox_bounds(current)
                bounds = (min_x, min_y, max_x, max_y)
                changed = False
                # Try right-edge and top-edge first (they're the ones blocked
                # by interior blocks; left/bottom satisfied by translation).
                for i, code in enumerate(boundary_codes):
                    if not code or fixed_xy[i]:
                        continue
                    if touches_edge(current[i], bounds, code):
                        continue
                    x, y, w, h = current[i]
                    tx = x
                    ty = y
                    if code & 1:
                        tx = min_x
                    elif code & 2:
                        tx = max_x - w
                    if code & 8:
                        ty = min_y
                    elif code & 4:
                        ty = max_y - h
                    target = (tx, ty, w, h)

                    if not overlaps_any(target, {i}):
                        current[i] = target
                        changed = True
                        continue

                    # Swap attempt: find a non-boundary block at the required
                    # edge but with a y/x slot we can use.
                    for j in range(block_count):
                        if j == i or boundary_codes[j] or fixed_xy[j]:
                            continue
                        jx, jy, jw, jh = current[j]
                        at_required_edge = False
                        if (code & 2) and abs(jx + jw - max_x) < 1e-4:
                            at_required_edge = True
                        if (code & 4) and abs(jy + jh - max_y) < 1e-4:
                            at_required_edge = True
                        if (code & 1) and abs(jx - min_x) < 1e-4:
                            at_required_edge = True
                        if (code & 8) and abs(jy - min_y) < 1e-4:
                            at_required_edge = True
                        if not at_required_edge:
                            continue

                        # Place i at j's edge slot, j at i's old slot.
                        if code & 2:
                            new_i = (max_x - w, jy, w, h)
                        elif code & 1:
                            new_i = (min_x, jy, w, h)
                        elif code & 4:
                            new_i = (jx, max_y - h, w, h)
                        else:
                            new_i = (jx, min_y, w, h)
                        new_j = (x, y, jw, jh)
                        # Both new rects must be overlap-free w.r.t. everyone
                        # else AND w.r.t. each other.
                        if rectangles_overlap(new_i, new_j):
                            continue
                        ok = True
                        for k in range(block_count):
                            if k == i or k == j:
                                continue
                            if rectangles_overlap(new_i, current[k]) or rectangles_overlap(new_j, current[k]):
                                ok = False
                                break
                        if not ok:
                            continue
                        # Confirm j still satisfies all its hard constraints
                        # (dim immutability preserved; area preserved since
                        # only position changes; preplaced excluded above).
                        current[i] = new_i
                        current[j] = new_j
                        changed = True
                        break

                if not changed:
                    break
            return current

        def top_backfill_pass(positions):
            # Re-home free blocks that sit above the highest hard top edge
            # (preplaced anchor) into row-end gaps below it.  Shrinks the
            # bbox toward the anchored top so top-coded preplaced blocks can
            # touch the bounding box again.
            if positions is None or len(positions) != block_count:
                return positions
            tops = [r[1] + r[3] for r in preplaced_positions.values()]
            if not tops:
                return positions
            y_anchor = max(tops)
            current = [tuple(p) for p in positions]
            bb = bbox_bounds(current)
            if bb[3] <= y_anchor + 1e-7:
                return positions

            rows = {}
            for i in range(block_count):
                if fixed_xy[i]:
                    continue
                x, y, w, h = current[i]
                if y + h <= y_anchor + 1e-7:
                    rows.setdefault((round(y, 5), round(h, 5)), []).append(i)

            def row_gaps(key, members):
                y, h = key
                occ = sorted((current[i][0], current[i][0] + current[i][2]) for i in members)
                for rx, ry, rw, rh in preplaced_positions.values():
                    if ry < y + h - 1e-9 and ry + rh > y + 1e-9:
                        occ.append((rx, rx + rw))
                occ.sort()
                gaps = []
                cursor = bb[0]
                for a, b in occ:
                    if a > cursor + 1e-7:
                        gaps.append([cursor, a])
                    cursor = max(cursor, b)
                if bb[2] > cursor + 1e-7:
                    gaps.append([cursor, bb[2]])
                return gaps

            gap_map = {key: row_gaps(key, members) for key, members in rows.items()}
            movers = [i for i in range(block_count)
                      if not fixed_xy[i] and not fixed_dim[i]
                      and boundary_codes[i] == 0 and cluster_ids[i] == 0
                      and mib_ids[i] == 0
                      and current[i][1] >= y_anchor - 1e-7]
            movers.sort(key=lambda i: areas[i])
            for b in movers:
                placed_flag = False
                for key in sorted(gap_map, key=lambda k: -k[1]):
                    y, h = key
                    if h <= eps:
                        continue
                    w_need = areas[b] / h
                    for gap in gap_map[key]:
                        if gap[1] - gap[0] >= w_need + 1e-9:
                            current[b] = (gap[0], y, w_need, h)
                            gap[0] += w_need
                            placed_flag = True
                            break
                    if placed_flag:
                        break
            return current

        def row_swap_refine(positions, passes=3):
            # Adjacent-pair swaps inside detected rows: blocks sharing the
            # same (y, h) and abutting in x can swap order without touching
            # feasibility (areas, dims, overlaps all preserved).  Greedy
            # accept when the HPWL x-term improves.  Boundary-coded blocks
            # stay put; cluster members only swap within their own cluster.
            if positions is None or len(positions) != block_count:
                return positions
            current = [tuple(p) for p in positions]

            def hpwl_x_for(block, cx_map):
                total = 0.0
                for nbr, weight in b2b_adj[block]:
                    total += weight * abs(cx_map[block] - cx_map[nbr])
                for pin, weight in p2b_adj[block]:
                    px = tensor_value(pins_pos, pin, 0)
                    total += weight * abs(cx_map[block] - px)
                return total

            for _ in range(passes):
                rows = {}
                for i in range(block_count):
                    if fixed_xy[i]:
                        continue
                    x, y, w, h = current[i]
                    rows.setdefault((round(y, 5), round(h, 5)), []).append(i)
                improved = False
                cx = {i: current[i][0] + current[i][2] / 2.0 for i in range(block_count)}
                for key, members in rows.items():
                    if len(members) < 2:
                        continue
                    members.sort(key=lambda i: current[i][0])
                    for a_idx in range(len(members) - 1):
                        i = members[a_idx]
                        j = members[a_idx + 1]
                        xi, yi, wi, hi = current[i]
                        xj, yj, wj, hj = current[j]
                        if abs(xi + wi - xj) > 1e-6:
                            continue
                        if boundary_codes[i] or boundary_codes[j]:
                            continue
                        if cluster_ids[i] != 0 or cluster_ids[j] != 0:
                            # Same-cluster swaps shift member x-intervals and
                            # can break marginal cross-row adjacency that the
                            # evaluator's polygon union does not tolerate.
                            continue
                        if fixed_dim[i] or fixed_dim[j]:
                            continue
                        before = hpwl_x_for(i, cx) + hpwl_x_for(j, cx)
                        # Edge (i, j) counted twice above; consistent after.
                        old_ci, old_cj = cx[i], cx[j]
                        cx[i] = xi + wj + wi / 2.0
                        cx[j] = xi + wj / 2.0
                        after = hpwl_x_for(i, cx) + hpwl_x_for(j, cx)
                        if after + 1e-9 < before:
                            current[i] = (xi + wj, yi, wi, hi)
                            current[j] = (xi, yj, wj, hj)
                            members[a_idx] = j
                            members[a_idx + 1] = i
                            improved = True
                        else:
                            cx[i], cx[j] = old_ci, old_cj
                if not improved:
                    break
            return current

        def row_boundary_reorder(positions):
            # Move boundary-violating left/right-coded blocks to the start or
            # end of their own row by shifting the blocks between — widths
            # and heights unchanged, so feasibility is preserved exactly.
            if positions is None or len(positions) != block_count:
                return positions
            current = [tuple(p) for p in positions]
            bounds = bbox_bounds(current)
            min_x, _, max_x, _ = bounds

            rows = {}
            for i in range(block_count):
                if fixed_xy[i]:
                    continue
                x, y, w, h = current[i]
                rows.setdefault((round(y, 5), round(h, 5)), []).append(i)

            for key, members in rows.items():
                members.sort(key=lambda i: current[i][0])
                if any(cluster_ids[j] for j in members):
                    # Shifting cluster members sideways risks breaking their
                    # cross-row adjacency; leave such rows alone.
                    continue
                # Row must be contiguous from its first block for the shift
                # arithmetic to stay exact.
                for i in members:
                    code = boundary_codes[i]
                    if not code or touches_edge(current[i], bounds, code):
                        continue
                    idx = members.index(i)
                    if (code & 1) and abs(current[members[0]][0] - min_x) < 1e-7:
                        if any(boundary_codes[j] & 1 for j in members[:idx]):
                            continue
                        # Verify blocks first..idx-1 are contiguous with i.
                        run_ok = all(
                            abs(current[members[k]][0] + current[members[k]][2]
                                - current[members[k + 1]][0]) < 1e-6
                            for k in range(idx))
                        if not run_ok:
                            continue
                        xi, yi, wi, hi = current[i]
                        shift = wi
                        new_x = min_x
                        for j in members[:idx]:
                            xj, yj, wj, hj = current[j]
                            current[j] = (xj + shift, yj, wj, hj)
                        current[i] = (new_x, yi, wi, hi)
                        members[:idx + 1] = [i] + members[:idx]
                    elif (code & 2):
                        tail = members[idx + 1:]
                        run_ok = all(
                            abs(current[members[k]][0] + current[members[k]][2]
                                - current[members[k + 1]][0]) < 1e-6
                            for k in range(idx, len(members) - 1))
                        if any(boundary_codes[j] & 2 for j in tail):
                            continue
                        if not run_ok:
                            continue
                        xi, yi, wi, hi = current[i]
                        if max_x - wi < xi - 1e-9:
                            continue
                        for j in tail:
                            xj, yj, wj, hj = current[j]
                            current[j] = (xj - wi, yj, wj, hj)
                        current[i] = (max_x - wi, yi, wi, hi)
                        members[idx:] = tail + [i]
            return current

        def column_boundary_reorder(positions):
            # Vertical companion to row_boundary_reorder: move violating
            # bottom/top-coded blocks to the edge slot of their detected
            # column.  Dimensions and x positions are preserved; hard_feasible
            # and the contest proxy gate decide whether the move survives.
            if positions is None or len(positions) != block_count:
                return positions
            current = [tuple(p) for p in positions]
            bounds = bbox_bounds(current)
            _, min_y, _, max_y = bounds

            columns = {}
            for i in range(block_count):
                if fixed_xy[i]:
                    continue
                x, y, w, h = current[i]
                columns.setdefault((round(x, 5), round(w, 5)), []).append(i)

            for _key, members in columns.items():
                members.sort(key=lambda i: current[i][1])
                if any(cluster_ids[j] for j in members):
                    # Moving cluster members vertically can break edge
                    # connectivity across rows; cluster-specific repair owns
                    # those cases.
                    continue
                for i in members:
                    code = boundary_codes[i]
                    if not code or touches_edge(current[i], bounds, code):
                        continue
                    idx = members.index(i)
                    if (code & 8) and abs(current[members[0]][1] - min_y) < 1e-7:
                        if any(boundary_codes[j] & 8 for j in members[:idx]):
                            continue
                        run_ok = all(
                            abs(current[members[k]][1] + current[members[k]][3]
                                - current[members[k + 1]][1]) < 1e-6
                            for k in range(idx))
                        if not run_ok:
                            continue
                        xi, yi, wi, hi = current[i]
                        for j in members[:idx]:
                            xj, yj, wj, hj = current[j]
                            current[j] = (xj, yj + hi, wj, hj)
                        current[i] = (xi, min_y, wi, hi)
                        members[:idx + 1] = [i] + members[:idx]
                    elif code & 4:
                        tail = members[idx + 1:]
                        run_ok = all(
                            abs(current[members[k]][1] + current[members[k]][3]
                                - current[members[k + 1]][1]) < 1e-6
                            for k in range(idx, len(members) - 1))
                        if any(boundary_codes[j] & 4 for j in tail):
                            continue
                        if not run_ok:
                            continue
                        xi, yi, wi, hi = current[i]
                        if max_y - hi < yi - 1e-9:
                            continue
                        for j in tail:
                            xj, yj, wj, hj = current[j]
                            current[j] = (xj, yj - hi, wj, hj)
                        current[i] = (xi, max_y - hi, wi, hi)
                        members[idx:] = tail + [i]
            return current

        if best_positions is not None:
            compacted = compact_pass(best_positions)
            if hard_feasible(compacted):
                compacted_cost = full_cost(compacted)
                if compacted_cost + 1e-9 < best_cost:
                    best_positions = compacted
                    best_cost = compacted_cost
                    flow_scores["compact"] = compacted_cost

            snapped = boundary_snap_pass(best_positions)
            if hard_feasible(snapped):
                snapped_cost = full_cost(snapped)
                if snapped_cost + 1e-9 < best_cost:
                    best_positions = snapped
                    best_cost = snapped_cost
                    flow_scores["boundary_snap"] = snapped_cost

            swapped = boundary_swap_repair(best_positions)
            if hard_feasible(swapped):
                swapped_cost = full_cost(swapped)
                if swapped_cost + 1e-9 < best_cost:
                    best_positions = swapped
                    best_cost = swapped_cost
                    flow_scores["boundary_swap"] = swapped_cost

            # Re-compact after snap/swap — boundary blocks may have created
            # new dead space inside the tighter bbox.
            recompacted = compact_pass(best_positions)
            if hard_feasible(recompacted):
                rc_cost = full_cost(recompacted)
                if rc_cost + 1e-9 < best_cost:
                    best_positions = recompacted
                    best_cost = rc_cost
                    flow_scores["compact_after_snap"] = rc_cost

            # HPWL refinement: pull free blocks toward their connection
            # centroids without changing the bbox set by boundary anchors.
            forced = force_directed_refine(best_positions)
            if hard_feasible(forced):
                forced_cost = full_cost(forced)
                if forced_cost + 1e-9 < best_cost:
                    best_positions = forced
                    best_cost = forced_cost
                    flow_scores["force_directed"] = forced_cost

            # Final compact in case force-directed left small gaps.
            final_compact = compact_pass(best_positions)
            if hard_feasible(final_compact):
                fc_cost = full_cost(final_compact)
                if fc_cost + 1e-9 < best_cost:
                    best_positions = final_compact
                    best_cost = fc_cost
                    flow_scores["compact_final"] = fc_cost

            # Pull stragglers from above the anchored top edge into row-end
            # gaps below; the smaller bbox restores top anchors and area.
            backfilled = top_backfill_pass(best_positions)
            if hard_feasible(backfilled):
                bf_cost = full_cost(backfilled)
                sv_new = soft_violation_summary(backfilled)
                sv_old = soft_violation_summary(best_positions)
                if (bf_cost + 1e-9 < best_cost
                        and sv_new["grouping"] <= sv_old["grouping"]
                        and sv_new["mib"] <= sv_old["mib"]):
                    best_positions = backfilled
                    best_cost = bf_cost
                    flow_scores["top_backfill"] = bf_cost

            # Row-aware boundary repair: slide violating left/right-coded
            # blocks to their row's edge slot (exact, width-preserving).
            # Horizontal shifts can break vertical cluster adjacency, so
            # require the total soft-violation count not to increase.
            reordered = row_boundary_reorder(best_positions)
            if hard_feasible(reordered):
                ro_cost = full_cost(reordered)
                sv_new = soft_violation_summary(reordered)
                sv_old = soft_violation_summary(best_positions)
                if (ro_cost + 1e-9 < best_cost
                        and sv_new["grouping"] <= sv_old["grouping"]
                        and sv_new["boundary"] <= sv_old["boundary"]
                        and sv_new["mib"] <= sv_old["mib"]):
                    best_positions = reordered
                    best_cost = ro_cost
                    flow_scores["row_reorder"] = ro_cost

            # Column-aware boundary repair: vertical equivalent of row
            # reorder for bottom/top-coded blocks.  It is useful for corner
            # constraints after row_reorder has already fixed left/right.
            col_reordered = column_boundary_reorder(best_positions)
            if hard_feasible(col_reordered):
                cr_cost = full_cost(col_reordered)
                sv_new = soft_violation_summary(col_reordered)
                sv_old = soft_violation_summary(best_positions)
                if (cr_cost + 1e-9 < best_cost
                        and sv_new["grouping"] <= sv_old["grouping"]
                        and sv_new["boundary"] <= sv_old["boundary"]
                        and sv_new["mib"] <= sv_old["mib"]):
                    best_positions = col_reordered
                    best_cost = cr_cost
                    flow_scores["column_reorder"] = cr_cost

            # Within-row adjacent swaps: exact HPWL refinement for row-
            # structured layouts (no-op when rows don't abut).
            if os_mod.environ.get("FLOORSET_NO_ROWSWAP", "") != "1":
                row_swapped = row_swap_refine(best_positions)
                if hard_feasible(row_swapped):
                    rs_cost = full_cost(row_swapped)
                    sv_new = soft_violation_summary(row_swapped)
                    sv_old = soft_violation_summary(best_positions)
                    if (rs_cost + 1e-9 < best_cost
                            and sv_new["grouping"] <= sv_old["grouping"]
                            and sv_new["boundary"] <= sv_old["boundary"]
                            and sv_new["mib"] <= sv_old["mib"]):
                        best_positions = row_swapped
                        best_cost = rs_cost
                        flow_scores["row_swap"] = rs_cost

            # Conservative x-only HPWL cleanup for row-like layouts.
            h_forced = force_directed_horizontal_refine(best_positions)
            if hard_feasible(h_forced):
                hf_cost = full_cost(h_forced)
                sv_new = soft_violation_summary(h_forced)
                sv_old = soft_violation_summary(best_positions)
                if (hf_cost + 1e-9 < best_cost
                        and candidate_wire(h_forced) <= candidate_wire(best_positions) + 1e-9
                        and bbox_area(h_forced) <= bbox_area(best_positions) + 1e-9
                        and sv_new["grouping"] <= sv_old["grouping"]
                        and sv_new["boundary"] <= sv_old["boundary"]
                        and sv_new["mib"] <= sv_old["mib"]):
                    best_positions = h_forced
                    best_cost = hf_cost
                    flow_scores["force_directed_x"] = hf_cost

        result = normalize_if_safe(best_positions)
        if not hard_feasible(result):
            result = normalize_if_safe(fallback)
        if not hard_feasible(result):
            result = legalize(make_draft(order_area), order_area, try_ratios=False, candidate_limit=(24 if block_count > 80 else 50))
        if self.verbose and flow_scores:
            winner = min(flow_scores, key=flow_scores.get)
            print(f"  Experimental flow winner: {winner} ({flow_scores[winner]:.3f})")
        return result

    def _cost(self, positions, b2b_conn, p2b_conn, pins_pos) -> float:
        """Evaluate solution quality (lower is better)."""
        hpwl_b2b = calculate_hpwl_b2b(positions, b2b_conn)
        hpwl_p2b = calculate_hpwl_p2b(positions, p2b_conn, pins_pos)
        area = calculate_bbox_area(positions)
        return hpwl_b2b + hpwl_p2b + area * 0.01

    def _parse_constraints(self, block_count: int, constraints: torch.Tensor):
        ncols = constraints.shape[1] if constraints is not None and constraints.numel() > 0 else 0
        is_fixed = [False] * block_count
        is_preplaced = [False] * block_count
        boundary_flags = [False] * block_count
        for i in range(block_count):
            if ncols > 0 and constraints[i, 0] != 0:
                is_fixed[i] = True
            if ncols > 1 and constraints[i, 1] != 0:
                is_preplaced[i] = True
            if ncols > 4 and constraints[i, 4] != 0:
                boundary_flags[i] = True
        return is_fixed, is_preplaced, boundary_flags

    def _build_connectivity(self, block_count: int, b2b_connectivity: torch.Tensor,
                            p2b_connectivity: torch.Tensor, pins_pos: torch.Tensor):
        b2b_adj = [[] for _ in range(block_count)]
        p2b_adj = [[] for _ in range(block_count)]
        b2b_wsum = [0.0] * block_count
        p2b_wsum = [0.0] * block_count

        for row in b2b_connectivity:
            i, j, w = int(row[0]), int(row[1]), float(row[2])
            if 0 <= i < block_count and 0 <= j < block_count:
                b2b_adj[i].append((j, w))
                b2b_adj[j].append((i, w))
                b2b_wsum[i] += w
                b2b_wsum[j] += w

        for row in p2b_connectivity:
            pin_idx, block_idx, w = int(row[0]), int(row[1]), float(row[2])
            if 0 <= block_idx < block_count:
                p2b_adj[block_idx].append((pin_idx, w))
                p2b_wsum[block_idx] += w

        pin_centroid = [None] * block_count
        for i in range(block_count):
            sum_x = 0.0
            sum_y = 0.0
            sum_w = 0.0
            for pin_idx, w in p2b_adj[i]:
                if 0 <= pin_idx < pins_pos.shape[0]:
                    px, py = float(pins_pos[pin_idx, 0]), float(pins_pos[pin_idx, 1])
                    sum_x += px * w
                    sum_y += py * w
                    sum_w += w
            if sum_w > 0:
                pin_centroid[i] = (sum_x / sum_w, sum_y / sum_w)

        return b2b_adj, p2b_adj, b2b_wsum, p2b_wsum, pin_centroid

    def _compute_importance(self, areas, b2b_wsum, p2b_wsum, is_fixed, is_preplaced):
        total_area = sum(areas) + 1e-9
        total_b2b = sum(b2b_wsum) + 1e-9
        total_p2b = sum(p2b_wsum) + 1e-9
        importance = []
        for i in range(len(areas)):
            area_score = areas[i] / total_area
            b2b_score = b2b_wsum[i] / total_b2b
            p2b_score = p2b_wsum[i] / total_p2b
            constraint_score = (0.6 if is_preplaced[i] else 0.0) + (0.2 if is_fixed[i] else 0.0)
            importance.append(0.35 * area_score + 0.35 * b2b_score + 0.2 * p2b_score + constraint_score)
        return importance

    def _build_order(self, block_count: int, b2b_connectivity: torch.Tensor, importance):
        if block_count < 60 or b2b_connectivity.numel() == 0:
            return sorted(range(block_count), key=lambda i: importance[i], reverse=True)

        parent = list(range(block_count))
        size = [1] * block_count

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra == rb:
                return
            if size[ra] < size[rb]:
                ra, rb = rb, ra
            parent[rb] = ra
            size[ra] += size[rb]

        edges = [(float(row[2]), int(row[0]), int(row[1])) for row in b2b_connectivity]
        edges.sort(key=lambda e: e[0], reverse=True)
        target_size = max(4, int(math.sqrt(block_count)))

        for w, i, j in edges:
            ri, rj = find(i), find(j)
            if ri != rj and size[ri] < target_size and size[rj] < target_size:
                union(ri, rj)

        clusters = {}
        for i in range(block_count):
            root = find(i)
            clusters.setdefault(root, []).append(i)

        cluster_list = list(clusters.values())
        cluster_list.sort(key=lambda c: sum(importance[i] for i in c), reverse=True)

        order = []
        for cluster in cluster_list:
            cluster_sorted = sorted(cluster, key=lambda i: importance[i], reverse=True)
            order.extend(cluster_sorted)
        return order

    def _initialize_positions(self, block_count, areas, target_positions, is_preplaced):
        positions = []
        fixed_positions = {}
        for i in range(block_count):
            if target_positions[i, 2] != -1 and target_positions[i, 3] != -1:
                w = float(target_positions[i, 2])
                h = float(target_positions[i, 3])
            else:
                w = h = math.sqrt(areas[i])
            if is_preplaced[i] and target_positions[i, 0] != -1 and target_positions[i, 1] != -1:
                x = float(target_positions[i, 0])
                y = float(target_positions[i, 1])
                fixed_positions[i] = (x, y, w, h)
                positions.append((x, y, w, h))
            else:
                positions.append((0.0, 0.0, w, h))
        return positions, fixed_positions

    def _greedy_place(self, order, positions, areas, is_fixed, is_preplaced, boundary_flags,
                      fixed_positions, b2b_adj, p2b_adj, pins_pos, pin_centroid):
        placed = set(fixed_positions.keys())
        placed_list = list(placed)

        for block in order:
            if is_preplaced[block]:
                continue

            best = None
            best_cost = float('inf')
            best_overlap = float('inf')

            ratios = [1.0] if is_fixed[block] else self.ratio_candidates
            for ratio in ratios:
                if is_fixed[block]:
                    w, h = positions[block][2], positions[block][3]
                else:
                    w = math.sqrt(areas[block] * ratio)
                    h = math.sqrt(areas[block] / ratio)

                candidates = self._generate_candidates(
                    block, w, h, positions, placed_list, placed, b2b_adj,
                    pin_centroid, boundary_flags[block]
                )

                for cx, cy in candidates:
                    cx = max(0.0, cx)
                    cy = max(0.0, cy)
                    cost, overlap = self._candidate_cost(
                        block, cx, cy, w, h, positions, placed_list, placed,
                        b2b_adj, p2b_adj, pins_pos
                    )
                    if overlap < 1e-6 and cost < best_cost:
                        best = (cx, cy, w, h)
                        best_cost = cost
                        best_overlap = overlap
                    elif overlap < best_overlap:
                        best = (cx, cy, w, h)
                        best_cost = cost
                        best_overlap = overlap

            if best is None:
                w, h = positions[block][2], positions[block][3]
                best = (0.0, 0.0, w, h)

            positions[block] = best
            placed.add(block)
            placed_list.append(block)

        return positions

    def _generate_candidates(self, block, w, h, positions, placed_list, placed_set,
                             b2b_adj, pin_centroid, boundary_flag):
        candidates = []
        candidates.append((0.0, 0.0))

        if pin_centroid[block] is not None:
            cx, cy = pin_centroid[block]
            candidates.append((cx - w / 2.0, cy - h / 2.0))

        neighbor_candidates = []
        for nbr, weight in sorted(b2b_adj[block], key=lambda x: x[1], reverse=True)[:6]:
            if nbr in placed_set:
                xj, yj, wj, hj = positions[nbr]
                neighbor_candidates.append((xj + wj, yj))
                neighbor_candidates.append((xj, yj + hj))
        candidates.extend(neighbor_candidates)

        if placed_list:
            max_x = max(positions[i][0] + positions[i][2] for i in placed_list)
            max_y = max(positions[i][1] + positions[i][3] for i in placed_list)
            candidates.append((max_x, 0.0))
            candidates.append((0.0, max_y))

        if boundary_flag:
            candidates.append((0.0, h * 0.5))
            candidates.append((w * 0.5, 0.0))

        seen = set()
        unique = []
        for x, y in candidates:
            key = (round(x, 4), round(y, 4))
            if key not in seen:
                seen.add(key)
                unique.append((x, y))
        return unique

    def _candidate_cost(self, block, x, y, w, h, positions, placed_list, placed_set,
                        b2b_adj, p2b_adj, pins_pos):
        overlap_area = 0.0
        for j in placed_list:
            xj, yj, wj, hj = positions[j]
            overlap_x = min(x + w, xj + wj) - max(x, xj)
            overlap_y = min(y + h, yj + hj) - max(y, yj)
            if overlap_x > 1e-6 and overlap_y > 1e-6:
                overlap_area += overlap_x * overlap_y

        center_x = x + w / 2.0
        center_y = y + h / 2.0
        wire_cost = 0.0
        for nbr, weight in b2b_adj[block]:
            if nbr in placed_set:
                xn, yn, wn, hn = positions[nbr]
                cnx = xn + wn / 2.0
                cny = yn + hn / 2.0
                wire_cost += weight * (abs(center_x - cnx) + abs(center_y - cny))

        pin_cost = 0.0
        for pin_idx, weight in p2b_adj[block]:
            if 0 <= pin_idx < pins_pos.shape[0]:
                px, py = float(pins_pos[pin_idx, 0]), float(pins_pos[pin_idx, 1])
                pin_cost += weight * (abs(center_x - px) + abs(center_y - py))

        if placed_list:
            min_x = min([positions[i][0] for i in placed_list] + [x])
            min_y = min([positions[i][1] for i in placed_list] + [y])
            max_x = max([positions[i][0] + positions[i][2] for i in placed_list] + [x + w])
            max_y = max([positions[i][1] + positions[i][3] for i in placed_list] + [y + h])
        else:
            min_x, min_y, max_x, max_y = x, y, x + w, y + h
        bbox_area = (max_x - min_x) * (max_y - min_y)

        cost = wire_cost + pin_cost + 0.01 * bbox_area + overlap_area * 1e6
        return cost, overlap_area

    def _legalize_positions(self, order, positions, fixed_positions):
        result = [None] * len(positions)
        placed = []

        for idx, pos in fixed_positions.items():
            result[idx] = pos
            placed.append(idx)

        for block in order:
            if block in fixed_positions:
                continue
            x0, y0, w, h = positions[block]
            candidates = [(max(0.0, x0), max(0.0, y0)), (0.0, 0.0)]

            for j in placed:
                xj, yj, wj, hj = result[j]
                candidates.append((xj + wj, yj))
                candidates.append((xj, yj + hj))

            def cand_key(c):
                cx, cy = c
                dist = (cx - x0) ** 2 + (cy - y0) ** 2
                return (dist, cy, cx)

            candidates.sort(key=cand_key)
            chosen = None
            for cx, cy in candidates:
                if not self._overlaps_any(cx, cy, w, h, result, placed):
                    chosen = (cx, cy)
                    break

            if chosen is None:
                max_x = max([result[i][0] + result[i][2] for i in placed], default=0.0)
                cx, cy = max_x, 0.0
                while self._overlaps_any(cx, cy, w, h, result, placed):
                    max_y = 0.0
                    for j in placed:
                        xj, yj, wj, hj = result[j]
                        if min(cx + w, xj + wj) - max(cx, xj) > 1e-6:
                            max_y = max(max_y, yj + hj)
                    if max_y <= cy:
                        cy += h
                    else:
                        cy = max_y
                chosen = (cx, cy)

            result[block] = (chosen[0], chosen[1], w, h)
            placed.append(block)

        return result

    def _local_search(self, order, positions, areas, is_fixed, is_preplaced, fixed_positions,
                      b2b_adj, p2b_adj, pins_pos, pin_centroid, b2b_conn, p2b_conn,
                      target_positions):
        best_positions = list(positions)
        best_cost = self._cost(best_positions, b2b_conn, p2b_conn, pins_pos)

        max_iter = min(20000, max(1000, 200 * len(positions)))
        if len(positions) > 80:
            max_iter = min(max_iter, 4000)

        move_stats = {name: [0, 0] for name in ['shift', 'swap', 'aspect', 'compact']}
        movable = [i for i in range(len(positions)) if not is_preplaced[i]]

        for _ in range(max_iter):
            move = self._select_move(move_stats)
            candidate = self._apply_move(
                best_positions, areas, is_fixed, is_preplaced, movable,
                b2b_adj, p2b_adj, pins_pos, pin_centroid, move
            )
            candidate = self._legalize_positions(order, candidate, fixed_positions)
            if not self._is_feasible(candidate, areas, target_positions, None, is_fixed, is_preplaced):
                move_stats[move][1] += 1
                continue
            cost = self._cost(candidate, b2b_conn, p2b_conn, pins_pos)
            move_stats[move][1] += 1
            if cost < best_cost:
                best_cost = cost
                best_positions = candidate
                move_stats[move][0] += 1

        return best_positions

    def _select_move(self, move_stats):
        if random.random() < self.explore_prob:
            return random.choice(list(move_stats.keys()))
        scores = {}
        for name, (succ, att) in move_stats.items():
            scores[name] = (succ + 1.0) / (att + 2.0)
        return max(scores, key=scores.get)

    def _apply_move(self, positions, areas, is_fixed, is_preplaced, movable,
                    b2b_adj, p2b_adj, pins_pos, pin_centroid, move_type):
        candidate = list(positions)
        if not movable:
            return candidate

        if move_type == 'swap' and len(movable) >= 2:
            i, j = random.sample(movable, 2)
            xi, yi, wi, hi = candidate[i]
            xj, yj, wj, hj = candidate[j]
            candidate[i] = (xj, yj, wi, hi)
            candidate[j] = (xi, yi, wj, hj)
            return candidate

        block = random.choice(movable)
        x, y, w, h = candidate[block]

        if move_type == 'shift':
            cx = cy = total = 0.0
            for nbr, weight in b2b_adj[block]:
                xn, yn, wn, hn = candidate[nbr]
                cnx = xn + wn / 2.0
                cny = yn + hn / 2.0
                cx += cnx * weight
                cy += cny * weight
                total += weight
            for pin_idx, weight in p2b_adj[block]:
                if 0 <= pin_idx < pins_pos.shape[0]:
                    px, py = float(pins_pos[pin_idx, 0]), float(pins_pos[pin_idx, 1])
                    cx += px * weight
                    cy += py * weight
                    total += weight
            if total > 0:
                cx /= total
                cy /= total
                center_x = x + w / 2.0
                center_y = y + h / 2.0
                new_cx = center_x + self.shift_alpha * (cx - center_x)
                new_cy = center_y + self.shift_alpha * (cy - center_y)
                candidate[block] = (max(0.0, new_cx - w / 2.0), max(0.0, new_cy - h / 2.0), w, h)
            return candidate

        if move_type == 'aspect' and not is_fixed[block]:
            ratio = random.choice(self.ratio_candidates)
            new_w = math.sqrt(areas[block] * ratio)
            new_h = math.sqrt(areas[block] / ratio)
            center_x = x + w / 2.0
            center_y = y + h / 2.0
            candidate[block] = (max(0.0, center_x - new_w / 2.0), max(0.0, center_y - new_h / 2.0), new_w, new_h)
            return candidate

        if move_type == 'compact':
            candidate[block] = (max(0.0, x - 0.2 * w), max(0.0, y - 0.2 * h), w, h)
            return candidate

        return candidate

    def _normalize_positions(self, positions):
        min_x = min(p[0] for p in positions)
        min_y = min(p[1] for p in positions)
        shift_x = -min_x if min_x < 0 else 0.0
        shift_y = -min_y if min_y < 0 else 0.0
        return [(x + shift_x, y + shift_y, w, h) for x, y, w, h in positions]

    def _is_feasible(self, positions, areas, target_positions, constraints, is_fixed, is_preplaced):
        if check_overlap(positions) > 0:
            return False

        if target_positions is None:
            target_positions = torch.full((len(positions), 4), -1.0, dtype=torch.float32)

        for i, (x, y, w, h) in enumerate(positions):
            if is_preplaced[i]:
                tx, ty, tw, th = target_positions[i]
                if abs(x - float(tx)) > 1e-4 or abs(y - float(ty)) > 1e-4:
                    return False
                if abs(w - float(tw)) > 1e-4 or abs(h - float(th)) > 1e-4:
                    return False
                continue
            if is_fixed[i]:
                _, _, tw, th = target_positions[i]
                if abs(w - float(tw)) > 1e-4 or abs(h - float(th)) > 1e-4:
                    return False
                continue
            target_area = areas[i]
            if target_area > 0:
                diff = abs(w * h - target_area) / target_area
                if diff > 0.01:
                    return False
        return True

    def _shelf_fallback(self, positions, areas, is_preplaced, fixed_positions):
        order = sorted(range(len(positions)), key=lambda i: areas[i], reverse=True)
        fallback = list(positions)
        placed = []
        for idx, pos in fixed_positions.items():
            fallback[idx] = pos
            placed.append(idx)

        for i in order:
            if i in fixed_positions:
                continue
            x0, y0, w, h = fallback[i]
            candidates = [(0.0, 0.0)]
            for j in placed:
                xj, yj, wj, hj = fallback[j]
                candidates.append((xj + wj, yj))
                candidates.append((xj, yj + hj))
            candidates.sort(key=lambda c: (c[1], c[0]))
            chosen = None
            for cx, cy in candidates:
                if not self._overlaps_any(cx, cy, w, h, fallback, placed):
                    chosen = (cx, cy)
                    break
            if chosen is None:
                chosen = (0.0, 0.0)
            fallback[i] = (chosen[0], chosen[1], w, h)
            placed.append(i)
        return fallback

    def _overlaps_any(self, x, y, w, h, positions, placed):
        for j in placed:
            xj, yj, wj, hj = positions[j]
            overlap_x = min(x + w, xj + wj) - max(x, xj)
            overlap_y = min(y + h, yj + hj) - max(y, yj)
            if overlap_x > 1e-6 and overlap_y > 1e-6:
                return True
        return False
