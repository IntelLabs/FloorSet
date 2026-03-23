#!/usr/bin/env python3
"""
ICCAD 2026 FloorSet Challenge - Contest Framework

Unified contest framework with all functionality accessible via switches.

Usage:
    python iccad2026_evaluate.py --evaluate my_optimizer.py         # Evaluate optimizer
    python iccad2026_evaluate.py --validate my_optimizer.py         # Validate submission
    python iccad2026_evaluate.py --baseline                         # Generate baseline metrics
    python iccad2026_evaluate.py --score solution.json              # Score a solution file
    python iccad2026_evaluate.py --visualize --test-id 0            # Visualize test case
    python iccad2026_evaluate.py --info                             # Show contest info

Examples:
    python iccad2026_evaluate.py --evaluate my_optimizer.py --test-id 0 --verbose
    python iccad2026_evaluate.py --baseline --output baselines.json
    python iccad2026_evaluate.py --validate my_optimizer.py --quick
"""

import argparse
import importlib.util
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Callable

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from litetestLoader import FloorplanDatasetLiteTest, floorplan_collate
from liteLoader import FloorplanDatasetLite  # Training data (1M samples)
from cost import calculate_weighted_b2b_wirelength, calculate_weighted_p2b_wirelength
from utils import (
    unpad_tensor,
    check_fixed_const, check_preplaced_const, check_mib_const,
    check_clust_const, check_boundary_const
)

try:
    from shapely.geometry import Polygon, box
    from shapely.ops import unary_union
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False

# =============================================================================
# CONTEST PARAMETERS (from problem statement)
# =============================================================================
ALPHA = 0.5       # Quality metrics weight
BETA = 2.0        # Violation penalty exponent  
GAMMA = 0.3       # Runtime factor damping
M_PENALTY = 10.0  # Infeasibility penalty
AREA_TOLERANCE = 0.01  # 1% area tolerance


# =============================================================================
# DATA CLASSES
# =============================================================================
@dataclass
class SolutionMetrics:
    """Container for all evaluation metrics."""
    is_feasible: bool
    overlap_violations: int
    area_violations: int
    hpwl_b2b: float
    hpwl_p2b: float
    hpwl_total: float
    hpwl_baseline: float
    hpwl_gap: float
    bbox_area: float
    bbox_area_baseline: float
    area_gap: float
    fixed_violations: int
    preplaced_violations: int
    boundary_violations: int
    grouping_violations: int
    mib_violations: int
    total_soft_violations: int
    max_possible_violations: int
    violations_relative: float
    runtime_seconds: float
    cost: float


@dataclass
class TestResult:
    """Result for a single test case."""
    test_id: int
    block_count: int
    is_feasible: bool
    hpwl_gap: float
    area_gap: float
    violations_relative: float
    runtime_seconds: float
    cost: float
    positions: Optional[List[Tuple[float, float, float, float]]] = None
    error: Optional[str] = None


@dataclass 
class EvaluationResult:
    """Complete evaluation result."""
    submission_name: str
    timestamp: str
    total_score: float
    test_results: List[TestResult]
    summary: Dict[str, Any]


# =============================================================================
# SCORING FUNCTIONS
# =============================================================================
def calculate_hpwl_b2b(
    positions: List[Tuple[float, float, float, float]],
    b2b_connectivity: torch.Tensor
) -> float:
    """Calculate block-to-block HPWL."""
    if b2b_connectivity is None or len(b2b_connectivity) == 0:
        return 0.0
    
    total_wl = 0.0
    for edge in b2b_connectivity:
        if edge[0] == -1:
            continue
        i, j, weight = int(edge[0]), int(edge[1]), float(edge[2])
        if i < len(positions) and j < len(positions):
            x1 = positions[i][0] + positions[i][2] / 2
            y1 = positions[i][1] + positions[i][3] / 2
            x2 = positions[j][0] + positions[j][2] / 2
            y2 = positions[j][1] + positions[j][3] / 2
            total_wl += weight * (abs(x2 - x1) + abs(y2 - y1))
    return total_wl


def calculate_hpwl_p2b(
    positions: List[Tuple[float, float, float, float]],
    p2b_connectivity: torch.Tensor,
    pins_pos: torch.Tensor
) -> float:
    """Calculate pin-to-block HPWL."""
    if p2b_connectivity is None or len(p2b_connectivity) == 0:
        return 0.0
    
    total_wl = 0.0
    for edge in p2b_connectivity:
        if edge[0] == -1:
            continue
        pin_idx, block_idx, weight = int(edge[0]), int(edge[1]), float(edge[2])
        if block_idx < len(positions) and pin_idx < len(pins_pos):
            px, py = float(pins_pos[pin_idx][0]), float(pins_pos[pin_idx][1])
            bx = positions[block_idx][0] + positions[block_idx][2] / 2
            by = positions[block_idx][1] + positions[block_idx][3] / 2
            total_wl += weight * (abs(px - bx) + abs(py - by))
    return total_wl


def calculate_bbox_area(positions: List[Tuple[float, float, float, float]]) -> float:
    """Calculate bounding box area of all blocks."""
    if not positions:
        return 0.0
    
    x_min = min(p[0] for p in positions)
    y_min = min(p[1] for p in positions)
    x_max = max(p[0] + p[2] for p in positions)
    y_max = max(p[1] + p[3] for p in positions)
    
    return (x_max - x_min) * (y_max - y_min)


def check_overlap(positions: List[Tuple[float, float, float, float]]) -> int:
    """Check for overlapping blocks (touching edges OK)."""
    violations = 0
    n = len(positions)
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1, w1, h1 = positions[i]
            x2, y2, w2, h2 = positions[j]
            
            # Check for actual overlap (not just touching)
            overlap_x = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
            overlap_y = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
            
            if overlap_x > 1e-6 and overlap_y > 1e-6:
                violations += 1
    return violations


def check_area_tolerance(
    positions: List[Tuple[float, float, float, float]],
    target_areas: torch.Tensor,
    tolerance: float = AREA_TOLERANCE
) -> int:
    """Check if block areas are within tolerance of targets."""
    violations = 0
    for i, (x, y, w, h) in enumerate(positions):
        if i >= len(target_areas) or target_areas[i] == -1:
            continue
        actual_area = w * h
        target_area = float(target_areas[i])
        if target_area > 0:
            diff = abs(actual_area - target_area) / target_area
            if diff > tolerance:
                violations += 1
    return violations


def compute_cost(
    hpwl_gap: float,
    area_gap: float,
    violations_relative: float,
    runtime_factor: float,
    is_feasible: bool
) -> float:
    """
    Compute the official contest cost.
    
    Cost = (1 + α·(HPWL_gap + Area_gap)) × exp(β·V_rel) × max(0.7, R^γ)
         = M (10.0) if infeasible
    """
    if not is_feasible:
        return M_PENALTY
    
    quality_factor = 1 + ALPHA * (max(0, hpwl_gap) + max(0, area_gap))
    violation_factor = math.exp(BETA * violations_relative)
    runtime_adjustment = max(0.7, math.pow(max(0.01, runtime_factor), GAMMA))
    
    return quality_factor * violation_factor * runtime_adjustment


def evaluate_solution(
    solution: Dict,
    baseline_metrics: Dict,
    target_constraints: torch.Tensor,
    b2b_connectivity: torch.Tensor,
    p2b_connectivity: torch.Tensor,
    pins_pos: torch.Tensor,
    target_areas: torch.Tensor,
    target_positions: Optional[List] = None,
    median_runtime: float = 1.0
) -> SolutionMetrics:
    """Evaluate a solution and compute all metrics."""
    positions = solution['positions']
    runtime = solution.get('runtime', 1.0)
    block_count = len(positions)
    
    # Calculate HPWL
    hpwl_b2b = calculate_hpwl_b2b(positions, b2b_connectivity)
    hpwl_p2b = calculate_hpwl_p2b(positions, p2b_connectivity, pins_pos)
    hpwl_total = hpwl_b2b + hpwl_p2b
    
    hpwl_baseline = baseline_metrics.get('hpwl_baseline', hpwl_total)
    hpwl_gap = (hpwl_total - hpwl_baseline) / max(hpwl_baseline, 1e-6)
    
    # Calculate area
    bbox_area = calculate_bbox_area(positions)
    area_baseline = baseline_metrics.get('area_baseline', bbox_area)
    area_gap = (bbox_area - area_baseline) / max(area_baseline, 1e-6)
    
    # Check hard constraints (feasibility)
    overlap_violations = check_overlap(positions)
    area_violations = check_area_tolerance(positions, target_areas)
    is_feasible = (overlap_violations == 0) and (area_violations == 0)
    
    # Check soft constraints
    fixed_violations = 0
    preplaced_violations = 0
    boundary_violations = 0
    grouping_violations = 0
    mib_violations = 0
    
    # Count constraint instances
    max_violations = 0
    if target_constraints is not None and len(target_constraints) >= block_count:
        for i in range(block_count):
            if target_constraints.shape[1] > 0 and target_constraints[i, 0] != 0:
                max_violations += 1  # Fixed
            if target_constraints.shape[1] > 1 and target_constraints[i, 1] != 0:
                max_violations += 1  # Preplaced
            if target_constraints.shape[1] > 4 and target_constraints[i, 4] != 0:
                max_violations += 1  # Boundary
        
        # MIB groups
        if target_constraints.shape[1] > 2:
            mib_groups = set(int(target_constraints[i, 2]) for i in range(block_count) 
                           if target_constraints[i, 2] > 0)
            max_violations += len(mib_groups)
        
        # Cluster groups  
        if target_constraints.shape[1] > 3:
            cluster_groups = set(int(target_constraints[i, 3]) for i in range(block_count)
                               if target_constraints[i, 3] > 0)
            max_violations += len(cluster_groups)
    
    total_soft_violations = (fixed_violations + preplaced_violations + 
                            boundary_violations + grouping_violations + mib_violations)
    violations_relative = total_soft_violations / max(max_violations, 1)
    
    # Compute cost
    runtime_factor = runtime / max(median_runtime, 0.01)
    cost = compute_cost(hpwl_gap, area_gap, violations_relative, runtime_factor, is_feasible)
    
    return SolutionMetrics(
        is_feasible=is_feasible,
        overlap_violations=overlap_violations,
        area_violations=area_violations,
        hpwl_b2b=hpwl_b2b,
        hpwl_p2b=hpwl_p2b,
        hpwl_total=hpwl_total,
        hpwl_baseline=hpwl_baseline,
        hpwl_gap=hpwl_gap,
        bbox_area=bbox_area,
        bbox_area_baseline=area_baseline,
        area_gap=area_gap,
        fixed_violations=fixed_violations,
        preplaced_violations=preplaced_violations,
        boundary_violations=boundary_violations,
        grouping_violations=grouping_violations,
        mib_violations=mib_violations,
        total_soft_violations=total_soft_violations,
        max_possible_violations=max_violations,
        violations_relative=violations_relative,
        runtime_seconds=runtime,
        cost=cost
    )


def compute_total_score(costs: List[float], block_counts: List[int]) -> float:
    """Compute weighted average score."""
    if not costs:
        return 0.0
    total_weight = sum(block_counts)
    if total_weight == 0:
        return sum(costs) / len(costs)
    return sum(c * w for c, w in zip(costs, block_counts)) / total_weight


# =============================================================================
# OPTIMIZER BASE CLASS & BASELINES
# =============================================================================
class FloorplanOptimizer:
    """Base class for floorplanning optimizers."""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
    
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
        Solve the floorplanning problem.
        
        Returns: List of (x, y, width, height) for each block
        """
        raise NotImplementedError("Subclasses must implement solve()")
    
    def query_cost(
        self,
        positions: List[Tuple[float, float, float, float]],
        b2b_connectivity: torch.Tensor,
        p2b_connectivity: torch.Tensor,
        pins_pos: torch.Tensor
    ) -> Dict[str, float]:
        """Query cost function during optimization."""
        hpwl_b2b = calculate_hpwl_b2b(positions, b2b_connectivity)
        hpwl_p2b = calculate_hpwl_p2b(positions, p2b_connectivity, pins_pos)
        bbox_area = calculate_bbox_area(positions)
        overlaps = check_overlap(positions)
        
        return {
            'hpwl_b2b': hpwl_b2b,
            'hpwl_p2b': hpwl_p2b,
            'hpwl_total': hpwl_b2b + hpwl_p2b,
            'bbox_area': bbox_area,
            'overlaps': overlaps,
            'total': hpwl_b2b + hpwl_p2b + bbox_area + overlaps * 1000
        }


class RandomOptimizer(FloorplanOptimizer):
    """Simple random placement baseline."""
    
    def solve(self, block_count, area_targets, b2b_connectivity, 
              p2b_connectivity, pins_pos, constraints):
        positions = []
        canvas_size = math.sqrt(sum(float(a) for a in area_targets[:block_count] if a > 0)) * 2
        
        for i in range(block_count):
            area = float(area_targets[i]) if area_targets[i] > 0 else 1.0
            w = h = math.sqrt(area)
            x = random.uniform(0, canvas_size - w)
            y = random.uniform(0, canvas_size - h)
            positions.append((x, y, w, h))
        
        return positions


class SimulatedAnnealingOptimizer(FloorplanOptimizer):
    """Simulated Annealing baseline optimizer."""
    
    def __init__(self, max_iterations: int = 5000, initial_temp: float = 1000.0,
                 cooling_rate: float = 0.995, verbose: bool = False):
        super().__init__(verbose)
        self.max_iterations = max_iterations
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
    
    def solve(self, block_count, area_targets, b2b_connectivity,
              p2b_connectivity, pins_pos, constraints):
        # Initialize with random placement
        positions = []
        total_area = sum(float(a) for a in area_targets[:block_count] if a > 0)
        canvas_size = math.sqrt(total_area) * 1.5
        
        for i in range(block_count):
            area = float(area_targets[i]) if area_targets[i] > 0 else 1.0
            w = h = math.sqrt(area)
            x = random.uniform(0, max(0, canvas_size - w))
            y = random.uniform(0, max(0, canvas_size - h))
            positions.append([x, y, w, h])
        
        # SA optimization
        current_cost = self._evaluate(positions, b2b_connectivity, p2b_connectivity, pins_pos)
        best_positions = [tuple(p) for p in positions]
        best_cost = current_cost
        temp = self.initial_temp
        
        for iteration in range(self.max_iterations):
            # Random move
            idx = random.randint(0, block_count - 1)
            old_pos = positions[idx].copy()
            
            move_type = random.choice(['translate', 'swap', 'resize'])
            
            if move_type == 'translate':
                dx = random.gauss(0, canvas_size * 0.1)
                dy = random.gauss(0, canvas_size * 0.1)
                positions[idx][0] = max(0, positions[idx][0] + dx)
                positions[idx][1] = max(0, positions[idx][1] + dy)
            elif move_type == 'swap' and block_count > 1:
                idx2 = random.randint(0, block_count - 1)
                if idx2 != idx:
                    positions[idx][0], positions[idx2][0] = positions[idx2][0], positions[idx][0]
                    positions[idx][1], positions[idx2][1] = positions[idx2][1], positions[idx][1]
            else:  # resize
                target_area = float(area_targets[idx]) if area_targets[idx] > 0 else positions[idx][2] * positions[idx][3]
                aspect = random.uniform(0.5, 2.0)
                positions[idx][2] = math.sqrt(target_area * aspect)
                positions[idx][3] = math.sqrt(target_area / aspect)
            
            new_cost = self._evaluate(positions, b2b_connectivity, p2b_connectivity, pins_pos)
            delta = new_cost - current_cost
            
            if delta < 0 or random.random() < math.exp(-delta / max(temp, 1e-10)):
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_positions = [tuple(p) for p in positions]
            else:
                positions[idx] = old_pos
            
            temp *= self.cooling_rate
        
        return best_positions
    
    def _evaluate(self, positions, b2b_conn, p2b_conn, pins_pos):
        pos_tuples = [tuple(p) for p in positions]
        hpwl = calculate_hpwl_b2b(pos_tuples, b2b_conn) + calculate_hpwl_p2b(pos_tuples, p2b_conn, pins_pos)
        area = calculate_bbox_area(pos_tuples)
        overlaps = check_overlap(pos_tuples)
        return hpwl + area * 0.01 + overlaps * 10000


# =============================================================================
# EVALUATION ENGINE
# =============================================================================
class ContestEvaluator:
    """Main evaluation engine."""
    
    def __init__(self, data_path: str = "../", verbose: bool = True):
        self.data_path = Path(data_path)
        self.verbose = verbose
        self.dataset = None
    
    def _load_dataset(self):
        if self.dataset is None:
            if self.verbose:
                print("Loading test dataset...")
            self.dataset = FloorplanDatasetLiteTest(str(self.data_path))
            if self.verbose:
                print(f"Loaded {len(self.dataset)} test cases")
    
    def _load_optimizer(self, optimizer_path: str) -> FloorplanOptimizer:
        """Load optimizer from file."""
        path = Path(optimizer_path)
        if not path.exists():
            raise FileNotFoundError(f"Optimizer file not found: {optimizer_path}")
        
        spec = importlib.util.spec_from_file_location("optimizer_module", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Find optimizer class
        for name in dir(module):
            obj = getattr(module, name)
            if (isinstance(obj, type) and 
                issubclass(obj, FloorplanOptimizer) and 
                obj is not FloorplanOptimizer):
                return obj(verbose=self.verbose)
        
        # Try common names
        for name in ['MyOptimizer', 'Optimizer', 'ContestOptimizer']:
            if hasattr(module, name):
                return getattr(module, name)(verbose=self.verbose)
        
        raise ValueError(f"No optimizer class found in {optimizer_path}")
    
    def _extract_baseline(self, idx, labels, b2b_conn, p2b_conn, pins_pos, block_count):
        """Extract baseline metrics from ground truth."""
        polygons, metrics = labels
        
        positions = []
        for i in range(block_count):
            block = polygons[i]
            valid = block[block[:, 0] != -1]
            if len(valid) > 0:
                x_min, y_min = valid.min(dim=0).values
                x_max, y_max = valid.max(dim=0).values
                positions.append((float(x_min), float(y_min), 
                                float(x_max - x_min), float(y_max - y_min)))
            else:
                positions.append((0, 0, 1, 1))
        
        hpwl = calculate_hpwl_b2b(positions, b2b_conn) + calculate_hpwl_p2b(positions, p2b_conn, pins_pos)
        area = calculate_bbox_area(positions)
        
        # Use stored metrics if available
        if metrics is not None and len(metrics) >= 8:
            if metrics[0] > 0:
                area = float(metrics[0])
            if metrics[-2] > 0 and metrics[-1] >= 0:
                hpwl = float(metrics[-2]) + float(metrics[-1])
        
        return {'hpwl_baseline': hpwl, 'area_baseline': area}, positions
    
    def evaluate(
        self,
        optimizer_path: str,
        test_ids: Optional[List[int]] = None,
        timeout: float = 60.0
    ) -> EvaluationResult:
        """Run full evaluation."""
        self._load_dataset()
        optimizer = self._load_optimizer(optimizer_path)
        
        if test_ids is None:
            test_ids = list(range(len(self.dataset)))
        
        results = []
        runtimes = []
        
        iterator = tqdm(test_ids, desc="Evaluating") if self.verbose else test_ids
        
        for idx in iterator:
            try:
                sample = self.dataset[idx]
                inputs, labels = sample['input'], sample['label']
                area_target, b2b_conn, p2b_conn, pins_pos, constraints = inputs
                block_count = int((area_target != -1).sum().item())
                
                baseline, target_pos = self._extract_baseline(
                    idx, labels, b2b_conn, p2b_conn, pins_pos, block_count
                )
                
                # Run optimizer
                start = time.time()
                positions = optimizer.solve(
                    block_count, area_target, b2b_conn, p2b_conn, pins_pos, constraints
                )
                runtime = time.time() - start
                runtimes.append(runtime)
                
                # Evaluate
                metrics = evaluate_solution(
                    {'positions': positions, 'runtime': runtime},
                    baseline,
                    constraints,
                    b2b_conn,
                    p2b_conn,
                    pins_pos,
                    area_target,
                    target_pos,
                    median_runtime=1.0
                )
                
                results.append(TestResult(
                    test_id=idx,
                    block_count=block_count,
                    is_feasible=metrics.is_feasible,
                    hpwl_gap=metrics.hpwl_gap,
                    area_gap=metrics.area_gap,
                    violations_relative=metrics.violations_relative,
                    runtime_seconds=runtime,
                    cost=metrics.cost,
                    positions=positions
                ))
                
            except Exception as e:
                results.append(TestResult(
                    test_id=idx, block_count=0, is_feasible=False,
                    hpwl_gap=0, area_gap=0, violations_relative=1.0,
                    runtime_seconds=0, cost=M_PENALTY, error=str(e)
                ))
        
        # Recompute with median runtime
        if runtimes:
            median_rt = sorted(runtimes)[len(runtimes)//2]
            for r in results:
                if r.error is None:
                    rt_factor = r.runtime_seconds / max(median_rt, 0.01)
                    r.cost = compute_cost(r.hpwl_gap, r.area_gap, r.violations_relative,
                                         rt_factor, r.is_feasible)
        
        costs = [r.cost for r in results]
        blocks = [r.block_count for r in results]
        total_score = compute_total_score(costs, blocks)
        
        return EvaluationResult(
            submission_name=Path(optimizer_path).stem,
            timestamp=datetime.now().isoformat(),
            total_score=total_score,
            test_results=results,
            summary={
                'num_tests': len(results),
                'num_feasible': sum(1 for r in results if r.is_feasible),
                'avg_cost': sum(costs) / len(costs) if costs else 0,
                'avg_runtime': sum(runtimes) / len(runtimes) if runtimes else 0,
            }
        )


# =============================================================================
# SUBMISSION VALIDATOR
# =============================================================================
def validate_submission(optimizer_path: str, quick: bool = False, verbose: bool = True) -> bool:
    """Validate a submission file."""
    checks = []
    
    def log(msg, ok=True):
        status = "✓" if ok else "✗"
        if verbose:
            print(f"  {status} {msg}")
        checks.append((msg, ok))
    
    if verbose:
        print(f"\nValidating: {optimizer_path}")
        print("-" * 50)
    
    # Check file exists
    path = Path(optimizer_path)
    if not path.exists():
        log(f"File exists", False)
        return False
    log("File exists")
    
    # Check Python syntax
    try:
        with open(path) as f:
            compile(f.read(), path, 'exec')
        log("Valid Python syntax")
    except SyntaxError as e:
        log(f"Valid Python syntax: {e}", False)
        return False
    
    # Try to load module
    try:
        spec = importlib.util.spec_from_file_location("test_module", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        log("Module loads successfully")
    except Exception as e:
        log(f"Module loads: {e}", False)
        return False
    
    # Find optimizer class
    optimizer_class = None
    for name in dir(module):
        obj = getattr(module, name)
        if isinstance(obj, type) and hasattr(obj, 'solve'):
            optimizer_class = obj
            break
    
    if optimizer_class is None:
        log("Contains optimizer class with solve() method", False)
        return False
    log(f"Contains optimizer class: {optimizer_class.__name__}")
    
    # Test on dummy data (unless quick mode)
    if not quick:
        try:
            optimizer = optimizer_class()
            dummy_areas = torch.tensor([100.0] * 5)
            dummy_b2b = torch.tensor([[0, 1, 1.0], [1, 2, 1.0]])
            dummy_p2b = torch.tensor([[0, 0, 1.0]])
            dummy_pins = torch.tensor([[0.0, 0.0]])
            dummy_constraints = torch.zeros(5, 5)
            
            start = time.time()
            result = optimizer.solve(5, dummy_areas, dummy_b2b, dummy_p2b, 
                                    dummy_pins, dummy_constraints)
            runtime = time.time() - start
            
            if not isinstance(result, list) or len(result) != 5:
                log(f"Returns correct format (expected list of 5 tuples)", False)
            else:
                log(f"Returns correct format")
                log(f"Sample runtime: {runtime:.3f}s")
        except Exception as e:
            log(f"Runs on sample data: {e}", False)
            return False
    
    passed = all(ok for _, ok in checks)
    if verbose:
        print("-" * 50)
        print(f"Result: {'PASSED' if passed else 'FAILED'}")
    
    return passed


# =============================================================================
# BASELINE GENERATOR
# =============================================================================
def generate_baselines(data_path: str = "../", output_path: str = None, 
                       verbose: bool = True) -> Dict:
    """Generate baseline metrics for all test cases."""
    if verbose:
        print("Generating baseline metrics...")
    
    dataset = FloorplanDatasetLiteTest(data_path)
    baselines = []
    
    iterator = tqdm(range(len(dataset)), desc="Processing") if verbose else range(len(dataset))
    
    for idx in iterator:
        sample = dataset[idx]
        inputs, labels = sample['input'], sample['label']
        area_target, b2b_conn, p2b_conn, pins_pos, constraints = inputs
        polygons, metrics = labels
        
        block_count = int((area_target != -1).sum().item())
        
        # Extract positions from ground truth
        positions = []
        for i in range(block_count):
            block = polygons[i]
            valid = block[block[:, 0] != -1]
            if len(valid) > 0:
                x_min, y_min = valid.min(dim=0).values
                x_max, y_max = valid.max(dim=0).values
                positions.append((float(x_min), float(y_min),
                                float(x_max - x_min), float(y_max - y_min)))
            else:
                positions.append((0, 0, 1, 1))
        
        hpwl_b2b = calculate_hpwl_b2b(positions, b2b_conn)
        hpwl_p2b = calculate_hpwl_p2b(positions, p2b_conn, pins_pos)
        area = calculate_bbox_area(positions)
        
        # Use stored metrics if available
        if metrics is not None and len(metrics) >= 8:
            if metrics[0] > 0:
                area = float(metrics[0])
            if metrics[-2] > 0:
                hpwl_b2b = float(metrics[-2])
            if metrics[-1] >= 0:
                hpwl_p2b = float(metrics[-1])
        
        baselines.append({
            'test_id': idx,
            'block_count': block_count,
            'hpwl_b2b': hpwl_b2b,
            'hpwl_p2b': hpwl_p2b,
            'hpwl_total': hpwl_b2b + hpwl_p2b,
            'area': area
        })
    
    result = {'baselines': baselines, 'generated': datetime.now().isoformat()}
    
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
        if verbose:
            print(f"Saved to {output_path}")
    
    return result


# =============================================================================
# TRAINING DATA UTILITIES
# =============================================================================
def compute_training_loss(
    positions: List[Tuple[float, float, float, float]],
    b2b_connectivity: torch.Tensor,
    p2b_connectivity: torch.Tensor,
    pins_pos: torch.Tensor,
    area_targets: torch.Tensor,
    constraints: Optional[torch.Tensor] = None,
    return_components: bool = False
) -> Dict[str, float]:
    """
    Compute loss/cost for a training sample.
    
    USE THIS IN YOUR TRAINING LOOP to get the same cost function
    used for final evaluation.
    
    Args:
        positions: Predicted [(x, y, w, h), ...] for each block
        b2b_connectivity: Block-to-block edges from training sample
        p2b_connectivity: Pin-to-block edges from training sample
        pins_pos: Pin positions from training sample
        area_targets: Target areas from training sample
        constraints: Constraint flags (optional)
        return_components: If True, return all components separately
    
    Returns:
        Dict with 'total' loss and optionally all components
    
    Example:
        for inputs, labels in dataloader:
            area_target, b2b_conn, p2b_conn, pins_pos, constraints = inputs
            
            # Your model predicts positions
            predicted_positions = my_model(inputs)
            
            # Compute loss using contest cost function
            loss_dict = compute_training_loss(
                predicted_positions,
                b2b_conn[i], p2b_conn[i], pins_pos[i], area_target[i]
            )
            
            loss = loss_dict['total']  # Use this for backprop
    """
    # Compute wirelength
    hpwl_b2b = calculate_hpwl_b2b(positions, b2b_connectivity)
    hpwl_p2b = calculate_hpwl_p2b(positions, p2b_connectivity, pins_pos)
    hpwl_total = hpwl_b2b + hpwl_p2b
    
    # Compute area
    bbox_area = calculate_bbox_area(positions)
    
    # Compute violations (for penalty terms)
    overlap_count = check_overlap(positions)
    area_violations = check_area_tolerance(positions, area_targets)
    
    # Combined loss (tune weights as needed)
    # These weights approximate the contest scoring impact
    loss = (
        hpwl_total +                    # Wirelength (main objective)
        0.01 * bbox_area +              # Area (secondary objective)
        10000 * overlap_count +         # Hard constraint: no overlaps
        5000 * area_violations          # Hard constraint: area tolerance
    )
    
    result = {'total': loss}
    
    if return_components:
        result.update({
            'hpwl_b2b': hpwl_b2b,
            'hpwl_p2b': hpwl_p2b,
            'hpwl_total': hpwl_total,
            'bbox_area': bbox_area,
            'overlap_count': overlap_count,
            'area_violations': area_violations
        })
    
    return result


def compute_training_loss_batch(
    positions_batch: List[List[Tuple[float, float, float, float]]],
    inputs_batch: Tuple
) -> List[Dict[str, float]]:
    """
    Compute loss for a batch of training samples.
    
    Args:
        positions_batch: List of position lists, one per sample in batch
        inputs_batch: Tuple of (area_target, b2b_conn, p2b_conn, pins_pos, constraints)
    
    Returns:
        List of loss dicts, one per sample
    
    Example:
        for inputs, labels in dataloader:
            area_target, b2b_conn, p2b_conn, pins_pos, constraints = inputs
            batch_size = area_target.shape[0]
            
            # Your model predicts positions for whole batch
            predicted_batch = [my_model(inputs, i) for i in range(batch_size)]
            
            # Compute losses
            losses = compute_training_loss_batch(predicted_batch, inputs)
            total_loss = sum(l['total'] for l in losses) / len(losses)
    """
    area_target, b2b_conn, p2b_conn, pins_pos, constraints = inputs_batch
    batch_size = len(positions_batch)
    
    losses = []
    for i in range(batch_size):
        loss = compute_training_loss(
            positions_batch[i],
            b2b_conn[i] if b2b_conn.dim() > 2 else b2b_conn,
            p2b_conn[i] if p2b_conn.dim() > 2 else p2b_conn,
            pins_pos[i] if pins_pos.dim() > 2 else pins_pos,
            area_target[i] if area_target.dim() > 1 else area_target,
            constraints[i] if constraints is not None and constraints.dim() > 2 else constraints
        )
        losses.append(loss)
    
    return losses


def get_training_dataloader(
    data_path: str = "../",
    batch_size: int = 32,
    num_samples: Optional[int] = None,
    shuffle: bool = False
) -> DataLoader:
    """
    Get a DataLoader for the FloorSet-Lite training data (1M samples).
    
    Args:
        data_path: Path to FloorSet data directory
        batch_size: Batch size for training
        num_samples: Limit number of samples (None = all 1M)
        shuffle: Whether to shuffle (False recommended for speed)
    
    Returns:
        DataLoader for training
        
    Example:
        dataloader = get_training_dataloader(batch_size=64)
        for batch in dataloader:
            inputs, labels = batch
            area_target, b2b_conn, p2b_conn, pins_pos, constraints = inputs
            polygons, metrics = labels
            # Train your model here
    """
    dataset = FloorplanDatasetLite(data_path)
    
    if num_samples is not None:
        # Use subset
        indices = list(range(min(num_samples, len(dataset))))
        dataset = torch.utils.data.Subset(dataset, indices)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=floorplan_collate
    )


def explore_training_data(
    data_path: str = "../",
    num_samples: int = 5,
    verbose: bool = True
) -> Dict:
    """
    Explore training data statistics.
    
    Args:
        data_path: Path to FloorSet data
        num_samples: Number of samples to examine
        verbose: Print details
    
    Returns:
        Statistics dict
    """
    print("Loading FloorSet-Lite training data...")
    dataset = FloorplanDatasetLite(data_path)
    print(f"Total training samples: {len(dataset):,}")
    
    stats = {
        'total_samples': len(dataset),
        'sample_stats': []
    }
    
    print(f"\nExamining {num_samples} random samples:")
    print("-" * 60)
    
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    
    for idx in indices:
        sample = dataset[idx]
        inputs, labels = sample['input'], sample['label']
        area_target, b2b_conn, p2b_conn, pins_pos, constraints = inputs
        polygons, metrics = labels
        
        block_count = int((area_target != -1).sum().item())
        num_b2b = int((b2b_conn[:, 0] != -1).sum().item()) if b2b_conn.numel() > 0 else 0
        num_p2b = int((p2b_conn[:, 0] != -1).sum().item()) if p2b_conn.numel() > 0 else 0
        num_pins = int((pins_pos[:, 0] != -1).sum().item()) if pins_pos.numel() > 0 else 0
        
        sample_stat = {
            'index': idx,
            'block_count': block_count,
            'b2b_edges': num_b2b,
            'p2b_edges': num_p2b,
            'pins': num_pins
        }
        stats['sample_stats'].append(sample_stat)
        
        if verbose:
            print(f"Sample {idx}:")
            print(f"  Blocks: {block_count}")
            print(f"  B2B edges: {num_b2b}")
            print(f"  P2B edges: {num_p2b}")
            print(f"  Pins: {num_pins}")
            
            # Count constraints
            num_fixed = int((constraints[:block_count, 0] != 0).sum().item()) if constraints.shape[1] > 0 else 0
            num_preplaced = int((constraints[:block_count, 1] != 0).sum().item()) if constraints.shape[1] > 1 else 0
            num_boundary = int((constraints[:block_count, 4] != 0).sum().item()) if constraints.shape[1] > 4 else 0
            print(f"  Constraints: {num_fixed} fixed, {num_preplaced} preplaced, {num_boundary} boundary")
            print()
    
    return stats


# =============================================================================
# VISUALIZATION
# =============================================================================
def visualize_test_case(test_id: int, data_path: str = "../", 
                       solution_path: str = None):
    """Visualize a test case (and optionally a solution)."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("matplotlib required for visualization")
        return
    
    dataset = FloorplanDatasetLiteTest(data_path)
    sample = dataset[test_id]
    inputs, labels = sample['input'], sample['label']
    area_target, b2b_conn, p2b_conn, pins_pos, constraints = inputs
    polygons, metrics = labels
    
    block_count = int((area_target != -1).sum().item())
    
    # Extract ground truth positions
    gt_positions = []
    for i in range(block_count):
        block = polygons[i]
        valid = block[block[:, 0] != -1]
        if len(valid) > 0:
            x_min, y_min = valid.min(dim=0).values
            x_max, y_max = valid.max(dim=0).values
            gt_positions.append((float(x_min), float(y_min),
                               float(x_max - x_min), float(y_max - y_min)))
    
    fig, axes = plt.subplots(1, 2 if solution_path else 1, figsize=(14, 7))
    if not solution_path:
        axes = [axes]
    
    # Plot ground truth
    ax = axes[0]
    ax.set_title(f"Test Case {test_id} - Ground Truth ({block_count} blocks)")
    
    colors = plt.cm.tab20(np.linspace(0, 1, block_count))
    for i, (x, y, w, h) in enumerate(gt_positions):
        rect = mpatches.Rectangle((x, y), w, h, fill=True, 
                                   facecolor=colors[i], edgecolor='black', alpha=0.7)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, str(i), ha='center', va='center', fontsize=8)
    
    ax.autoscale()
    ax.set_aspect('equal')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    
    plt.tight_layout()
    plt.savefig(f'test_case_{test_id}.png', dpi=150)
    print(f"Saved visualization to test_case_{test_id}.png")
    plt.show()


# =============================================================================
# SCORE SAVED SOLUTIONS
# =============================================================================
def score_saved_solutions(
    solutions_path: str,
    data_path: str = "../",
    output_path: Optional[str] = None
) -> Dict:
    """
    Re-score saved solutions without re-running the optimizer.
    
    Args:
        solutions_path: Path to solutions JSON (from --save-solutions)
        data_path: Path to FloorSet data
        output_path: Output file path (optional)
    
    Returns:
        Dict with scores
    """
    print(f"\nScoring saved solutions: {solutions_path}")
    print("=" * 60)
    
    # Load solutions
    with open(solutions_path) as f:
        data = json.load(f)
    
    solutions = data.get('solutions', [])
    print(f"Loaded {len(solutions)} solutions")
    
    # Load test dataset
    dataset = FloorplanDatasetLiteTest(data_path)
    
    results = []
    
    for sol in tqdm(solutions, desc="Scoring"):
        test_id = sol['test_id']
        positions = [tuple(p) for p in sol['positions']]
        block_count = sol['block_count']
        
        # Load test case data
        sample = dataset[test_id]
        inputs, labels = sample['input'], sample['label']
        area_target, b2b_conn, p2b_conn, pins_pos, constraints = inputs
        polygons, metrics = labels
        
        # Extract baseline from ground truth
        gt_positions = []
        for i in range(block_count):
            block = polygons[i]
            valid = block[block[:, 0] != -1]
            if len(valid) > 0:
                x_min, y_min = valid.min(dim=0).values
                x_max, y_max = valid.max(dim=0).values
                gt_positions.append((float(x_min), float(y_min),
                                   float(x_max - x_min), float(y_max - y_min)))
            else:
                gt_positions.append((0, 0, 1, 1))
        
        # Calculate baselines
        hpwl_baseline = calculate_hpwl_b2b(gt_positions, b2b_conn) + \
                       calculate_hpwl_p2b(gt_positions, p2b_conn, pins_pos)
        area_baseline = calculate_bbox_area(gt_positions)
        
        # Use stored metrics if available
        if metrics is not None and len(metrics) >= 8:
            if metrics[0] > 0:
                area_baseline = float(metrics[0])
            if metrics[-2] > 0 and metrics[-1] >= 0:
                hpwl_baseline = float(metrics[-2]) + float(metrics[-1])
        
        # Evaluate the saved solution
        solution_metrics = evaluate_solution(
            {'positions': positions, 'runtime': 1.0},
            {'hpwl_baseline': hpwl_baseline, 'area_baseline': area_baseline},
            constraints,
            b2b_conn,
            p2b_conn,
            pins_pos,
            area_target,
            gt_positions,
            median_runtime=1.0
        )
        
        results.append({
            'test_id': test_id,
            'block_count': block_count,
            'is_feasible': solution_metrics.is_feasible,
            'hpwl_gap': solution_metrics.hpwl_gap,
            'area_gap': solution_metrics.area_gap,
            'cost': solution_metrics.cost,
            'hpwl_total': solution_metrics.hpwl_total,
            'bbox_area': solution_metrics.bbox_area,
            'overlaps': solution_metrics.overlap_violations,
            'area_violations': solution_metrics.area_violations
        })
    
    # Compute total score
    costs = [r['cost'] for r in results]
    blocks = [r['block_count'] for r in results]
    total_score = compute_total_score(costs, blocks)
    
    # Print summary
    print("\n" + "=" * 60)
    print("SCORING RESULTS")
    print("=" * 60)
    print(f"\nTotal Score: {total_score:.4f}")
    print(f"Tests: {len(results)}")
    print(f"Feasible: {sum(1 for r in results if r['is_feasible'])}")
    print(f"Avg Cost: {sum(costs)/len(costs):.4f}")
    
    output = {
        'source': solutions_path,
        'timestamp': datetime.now().isoformat(),
        'total_score': total_score,
        'results': results
    }
    
    # Save if output path provided
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {output_path}")
    
    return output


# =============================================================================
# CLI INTERFACE
# =============================================================================
def print_contest_info():
    """Print contest information."""
    print("""
╔══════════════════════════════════════════════════════════════════╗
║          ICCAD 2026 FloorSet Challenge - Contest Framework       ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  SCORING FORMULA:                                                ║
║  Cost = (1 + α(HPWL_gap + Area_gap)) × exp(β·V_rel) × R_factor   ║
║                                                                  ║
║  Parameters: α=0.5, β=2.0, γ=0.3, M=10.0 (infeasible)           ║
║                                                                  ║
║  COMMANDS:                                                       ║
║    --evaluate optimizer.py   Run evaluation on test set          ║
║    --validate optimizer.py   Validate submission format          ║
║    --baseline                Generate baseline metrics            ║
║    --training                Explore training data (1M samples)  ║
║    --visualize --test-id N   Visualize test case N               ║
║    --info                    Show this information               ║
║                                                                  ║
║  EXAMPLES:                                                       ║
║    python iccad2026_evaluate.py --evaluate my_opt.py --verbose              ║
║    python iccad2026_evaluate.py --validate my_opt.py --quick                ║
║    python iccad2026_evaluate.py --evaluate my_opt.py --test-id 0            ║
║    python iccad2026_evaluate.py --training                                  ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
""")


def main():
    parser = argparse.ArgumentParser(
        description="ICCAD 2026 FloorSet Challenge - Contest Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Mode switches
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--evaluate', '-e', metavar='OPTIMIZER',
                     help='Evaluate an optimizer file')
    mode.add_argument('--score', '-S', metavar='SOLUTIONS_JSON',
                     help='Re-score saved solutions (from --save-solutions)')
    mode.add_argument('--validate', '-v', metavar='OPTIMIZER',
                     help='Validate a submission file')
    mode.add_argument('--baseline', '-b', action='store_true',
                     help='Generate baseline metrics')
    mode.add_argument('--visualize', '-V', action='store_true',
                     help='Visualize a test case')
    mode.add_argument('--training', '-T', action='store_true',
                     help='Explore training data (1M samples)')
    mode.add_argument('--info', '-i', action='store_true',
                     help='Show contest information')
    
    # Common options
    parser.add_argument('--data-path', '-d', default='../',
                       help='Path to FloorSet data (default: ../)')
    parser.add_argument('--output', '-o', default=None,
                       help='Output file path')
    parser.add_argument('--test-id', '-t', type=int, default=None,
                       help='Specific test case ID (0-99)')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')
    parser.add_argument('--quick', '-q', action='store_true',
                       help='Quick mode (skip some checks)')
    parser.add_argument('--save-solutions', '-s', action='store_true',
                       help='Save solutions (positions) to separate file')
    
    args = parser.parse_args()
    
    if args.info:
        print_contest_info()
        return
    
    if args.evaluate:
        evaluator = ContestEvaluator(args.data_path, verbose=True)
        test_ids = [args.test_id] if args.test_id is not None else None
        
        result = evaluator.evaluate(args.evaluate, test_ids)
        
        # Print summary
        print("\n" + "=" * 60)
        print(f"EVALUATION RESULTS: {result.submission_name}")
        print("=" * 60)
        print(f"\nTotal Score: {result.total_score:.4f}")
        print(f"Tests: {result.summary['num_tests']}")
        print(f"Feasible: {result.summary['num_feasible']}")
        print(f"Avg Cost: {result.summary['avg_cost']:.4f}")
        print(f"Avg Runtime: {result.summary['avg_runtime']:.2f}s")
        
        # Save results
        output = args.output or f"{result.submission_name}_results.json"
        with open(output, 'w') as f:
            json.dump(asdict(result), f, indent=2, default=str)
        print(f"\nResults saved to {output}")
        
        # Save solutions separately if requested
        if args.save_solutions:
            solutions_file = f"{result.submission_name}_solutions.json"
            solutions = {
                'submission': result.submission_name,
                'timestamp': result.timestamp,
                'solutions': [
                    {
                        'test_id': r.test_id,
                        'block_count': r.block_count,
                        'positions': r.positions
                    }
                    for r in result.test_results if r.positions is not None
                ]
            }
            with open(solutions_file, 'w') as f:
                json.dump(solutions, f, indent=2)
            print(f"Solutions saved to {solutions_file}")
    
    elif args.validate:
        success = validate_submission(args.validate, args.quick, verbose=True)
        sys.exit(0 if success else 1)
    
    elif args.baseline:
        output = args.output or 'baseline_metrics.json'
        generate_baselines(args.data_path, output, verbose=True)
    
    elif args.visualize:
        if args.test_id is None:
            print("Error: --test-id required for visualization")
            sys.exit(1)
        visualize_test_case(args.test_id, args.data_path)
    
    elif args.training:
        num_samples = 10 if args.test_id is None else args.test_id
        explore_training_data(args.data_path, num_samples=num_samples, verbose=True)
    
    elif args.score:
        score_saved_solutions(args.score, args.data_path, args.output)


if __name__ == '__main__':
    main()
