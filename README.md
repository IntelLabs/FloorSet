# FloorSet

FloorSet is a dataset that contains 2 MILLION floorplan benchmark circuits. These circuits reflect real-world constraints and objectives of the Floorplanning problem at SoC and sub-system hierarchies, which is a crucial component of the physical design flow. This dataset contains synthetic fixed-outline floorplan layouts in PyTorch tensor format. FloorSet is composed of two datasets:
    1. FloorSet-Prime (1M layouts)
    2. FloorSet-Lite (1M layouts)

*The dataset has 1M training samples and 100 test samples, with hard constraints seen in modern design flows such as shape constraints, edge-affinity, grouping constraints, and pre-placement constraints. FloorSet is intended to spur fundamental research on large-scale constrained optimization problems and alleviates the core issue of reproducibility in modern ML driven solutions to such problems. FloorSet has the potential to be “the Floorplanning” benchmark for academic research community and can speed up research in this domain. All data in FloorSet in synthetically generated based on an algorithm by our design, with no external input.

**We're putting the finishing touches on on the code.  Stay tuned for an update here.


