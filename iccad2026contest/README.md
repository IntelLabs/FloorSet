# ICCAD 2026 FloorSet Challenge

## Getting Started

```bash
# 0. Install dependencies
pip install -r requirements.txt

# 1. Copy the template
cp optimizer_template.py my_optimizer.py

# 2. Implement your algorithm in my_optimizer.py (edit the solve() method)

# 3. Evaluate on all 100 test cases
python iccad2026_evaluate.py --evaluate my_optimizer.py

# 4. Validate before submission
python iccad2026_evaluate.py --validate my_optimizer.py
```

---

## Your Task

Implement `solve()` in your optimizer file:

```python
def solve(self, block_count, area_targets, b2b_connectivity, 
          p2b_connectivity, pins_pos, constraints):
    """
    Place blocks to minimize wirelength and area.
    
    Returns: List of (x, y, width, height) tuples, one per block
    """
    positions = []
    for i in range(block_count):
        x, y = 0.0, 0.0       # Your placement algorithm
        w = h = math.sqrt(area_targets[i])
        positions.append((x, y, w, h))
    return positions
```

**Hard Constraints** (violation = score 10.0):
- No overlapping blocks
- Block area must be within 1% of target

---

## Using Training Data (1M samples)

```bash
# See full example
python training_example.py
```

```python
from iccad2026_evaluate import get_training_dataloader, compute_training_loss

dataloader = get_training_dataloader(batch_size=64, num_samples=10000)

for inputs, labels in dataloader:
    area_target, b2b_conn, p2b_conn, pins_pos, constraints = inputs
    
    # Your model predicts positions
    predicted = my_model(inputs)
    
    # Compute loss (same formula used for final scoring)
    loss = compute_training_loss(predicted, b2b_conn, p2b_conn, 
                                  pins_pos, area_target)['total']
```

---

## Saving and Re-scoring Solutions

```bash
# Run optimizer and save solutions to JSON
python iccad2026_evaluate.py --evaluate my_optimizer.py --save-solutions
# Output: my_optimizer_solutions.json

# Re-score saved solutions (without re-running optimizer)
python iccad2026_evaluate.py --score my_optimizer_solutions.json
```

This is useful for:
- Comparing scores after bug fixes
- Analyzing results without re-running slow optimizers

---

## Scoring

```
Cost = (1 + 0.5×(HPWL_gap + Area_gap)) × exp(2×Violations) × RuntimeFactor
     = 10.0 if infeasible
```

**Lower score = better.** Final ranking uses weighted average across all 100 tests.

---

## Commands Reference

| Command | Description |
|---------|-------------|
| `--evaluate FILE` | Run optimizer on 100 test cases, compute score |
| `--score FILE` | Re-score saved solutions (without re-running optimizer) |
| `--validate FILE` | Check submission format before submitting |
| `--training` | Explore training data statistics |
| `--test-id N` | Run on single test case (for debugging) |
| `--save-solutions` | Export positions to JSON (use with --evaluate) |
| `--info` | Show scoring formula |
