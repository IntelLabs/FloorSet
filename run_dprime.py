import torch
from torch.utils.data import DataLoader
from prime_dataset import FloorplanDataset, floorplan_collate

def main():
    root = './'
    ds = FloorplanDataset(root)
    # Avoid shuffling to make file caching more effective
    dl = DataLoader(ds, batch_size=128, shuffle=False,
                    collate_fn=floorplan_collate)

    # A minibatch can contain floorplans with different number of blocks or pins
    # All tensors in a minibatch are padded to the maximum number of blocks and pins in the minibatch
    # The padding for the LongTensors is -1, for the bool tensors it is False
    for (area_target,  # bsz x (n_blocks-1)  . The area-targets for each block
         b2b_connectivity,  # bsz x n_blocks x n_blocks x edge-weight
         p2b_connectivity,  # bsz x n_blocks x n_pins x edge-weight
         pins_pos,  # bsz x n_pins x 2. The pins location
         placement_constraints) in dl:  # bsz x n_blocks x 5. See visualize.py for the interpretation of placement constraints
        print(
            f'area-target data: {area_target.size()}, pins_pos: {pins_pos.size()}, b2b_connectivity: {b2b_connectivity.size()},p2b_connectivity: {p2b_connectivity.size()}, placement_constraints: {placement_constraints.size()}')
        break


if __name__ == "__main__":
    main()
