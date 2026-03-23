# ICCAD 2026 FloorSet Challenge - Contest Framework

## Quick Start

```bash
# Evaluate your optimizer
python contest.py --evaluate my_optimizer.py

# Validate submission format
python contest.py --validate my_optimizer.py

# Generate baseline metrics
python contest.py --baseline

# Explore training data (1M samples)
python contest.py --training

# Visualize a test case
python contest.py --visualize --test-id 0

# Show contest info
python contest.py --info
```

## Files

| File | Purpose |
|------|---------|
| `contest.py` | Main framework (scoring, evaluation, validation, training data) |
| `optimizer_template.py` | Template for contestants to implement |

## Contestant Workflow

```bash
# 1. Copy the template
cp optimizer_template.py my_optimizer.py

# 2. Explore training data
python contest.py --training

# 3. Implement your algorithm in my_optimizer.py

# 4. Test on single case
python contest.py --evaluate my_optimizer.py --test-id 0 --verbose

# 5. Run full evaluation
python contest.py --evaluate my_optimizer.py

# 6. Validate before submission
python contest.py --validate my_optimizer.py
```

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          CONTESTANT DATA FLOW                               │
└─────────────────────────────────────────────────────────────────────────────┘

PHASE 1: TRAINING (using 1M samples)
═══════════════════════════════════════════════════════════════════════════════

  FloorSet-Lite Training Data (1M samples from HuggingFace)
                              │
                              ▼
              ┌───────────────────────────────┐
              │  get_training_dataloader()    │
              │  (contest.py)                 │
              └───────────────────────────────┘
                              │
                              ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │  for inputs, labels in dataloader:                                  │
    │      area_target, b2b_conn, p2b_conn, pins_pos, constraints = inputs│
    │      polygons, metrics = labels  # Ground truth for supervised      │
    │                                                                     │
    │      predicted_positions = YOUR_MODEL(inputs)                       │
    │                                                                     │
    │      ┌─────────────────────────────────────────────────────────┐   │
    │      │  loss = compute_training_loss(                          │   │
    │      │      predicted_positions,                               │   │
    │      │      b2b_conn, p2b_conn, pins_pos, area_target          │   │
    │      │  )['total']                                             │   │
    │      │                                                         │   │
    │      │  # SAME COST as final evaluation!                       │   │
    │      │  # loss = hpwl + 0.01*area + 10000*overlaps + 5000*area_viol│
    │      └─────────────────────────────────────────────────────────┘   │
    │                                                                     │
    │      loss.backward()  # For neural networks                        │
    │      optimizer.step()                                              │
    └─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │  Trained Model  │
                    │  (saved weights)│
                    └─────────────────┘


PHASE 2: INFERENCE & EVALUATION (on 100 test cases)
═══════════════════════════════════════════════════════════════════════════════

  Test Data (100 cases in LiteTensorDataTest/)
                              │
                              ▼
              ┌───────────────────────────────┐
              │  contest.py --evaluate        │
              │  my_optimizer.py              │
              └───────────────────────────────┘
                              │
          ┌───────────────────┴───────────────────┐
          ▼                                       ▼
    ┌─────────────┐                      ┌─────────────────────┐
    │ Load inputs │                      │ YOUR_MODEL.solve()  │
    │ for test i  │ ──────────────────▶  │ returns positions   │
    └─────────────┘                      │ [(x,y,w,h), ...]    │
                                         └─────────────────────┘
                                                  │
                                                  ▼
                              ┌───────────────────────────────────────┐
                              │  evaluate_solution()                  │
                              │  (SAME cost function as training!)    │
                              │                                       │
                              │  • HPWL_gap = (hpwl - baseline) / base│
                              │  • Area_gap = (area - baseline) / base│
                              │  • Violations (overlaps, area errors) │
                              │  • Runtime factor                     │
                              │                                       │
                              │  Cost = (1 + α(gaps)) × exp(β·V) × R  │
                              └───────────────────────────────────────┘
                                                  │
                                                  ▼
                              ┌───────────────────────────────────────┐
                              │  Final Score = Σ(Cost_i × Blocks_i)   │
                              │                ─────────────────────  │
                              │                    Σ(Blocks_i)        │
                              └───────────────────────────────────────┘
                                                  │
                    ┌─────────────────────────────┼─────────────────────────────┐
                    ▼                             ▼                             ▼
          my_optimizer_results.json      Console Output           my_optimizer_solutions.json
          (scores + positions)           (summary)                (positions only)
```

---

## Training with Cost Function (1M samples)

**IMPORTANT**: The `compute_training_loss()` function uses the **EXACT SAME** cost
metrics as the final evaluation. This ensures your model optimizes the right objective.

```python
from contest import get_training_dataloader, compute_training_loss

# Get dataloader
dataloader = get_training_dataloader(
    batch_size=64,
    num_samples=10000,  # or None for all 1M
    shuffle=False       # False recommended for speed
)

# Training loop with cost function
for inputs, labels in dataloader:
    area_target, b2b_conn, p2b_conn, pins_pos, constraints = inputs
    polygons, metrics = labels  # Ground truth solutions (for supervised learning)
    
    # Your model predicts positions
    predicted_positions = my_model(inputs)
    
    # ═══════════════════════════════════════════════════════════════════════
    # COMPUTE LOSS - Same cost function as final evaluation!
    # ═══════════════════════════════════════════════════════════════════════
    loss_dict = compute_training_loss(
        predicted_positions,
        b2b_conn, p2b_conn, pins_pos, area_target,
        return_components=True  # Get detailed breakdown
    )
    
    loss = loss_dict['total']  # Use this for backprop
    
    # Also available:
    # loss_dict['hpwl_b2b']       - Block-to-block wirelength
    # loss_dict['hpwl_p2b']       - Pin-to-block wirelength
    # loss_dict['hpwl_total']     - Total wirelength
    # loss_dict['bbox_area']      - Bounding box area
    # loss_dict['overlap_count']  - Number of overlapping pairs (should be 0)
    # loss_dict['area_violations']- Blocks outside area tolerance (should be 0)
```

### Loss Formula (Training & Evaluation)

```
Training Loss = hpwl_total + 0.01 × bbox_area + 10000 × overlaps + 5000 × area_violations

Evaluation Cost = (1 + 0.5 × (HPWL_gap + Area_gap)) × exp(2 × V_rel) × R_factor
                = 10.0 if infeasible (overlaps > 0 or area_violations > 0)
```

Both penalize the same things: **wirelength, area, overlaps, and constraint violations**.

---

## Scoring Formula

```
Cost = (1 + α(HPWL_gap + Area_gap)) × exp(β·V_rel) × max(0.7, R^γ)
     = M (10.0) if infeasible (overlaps or area violations)
```

Parameters: α=0.5, β=2.0, γ=0.3

---

## Options

| Flag | Description |
|------|-------------|
| `--evaluate FILE` | Evaluate optimizer on test set |
| `--validate FILE` | Validate submission format |
| `--baseline` | Generate baseline metrics |
| `--training` | Explore training data (1M samples) |
| `--visualize` | Visualize test case (requires --test-id) |
| `--info` | Show contest information |
| `--test-id N` | Run on specific test case (0-99) |
| `--output FILE` | Output file path |
| `--save-solutions` | Save positions to separate JSON file |
| `--verbose` | Verbose output |
| `--quick` | Quick validation (skip runtime test) |
