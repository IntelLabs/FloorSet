import torch
from torch.utils.data import DataLoader

from prime_dataset import FloorplanDataset, floorplan_collate
from visualize import visualize_placement


def main():
    root = './'
    ds = FloorplanDataset(root)
    # Avoid shuffling to make file caching more effective
    dl = DataLoader(ds, batch_size=128, shuffle=False,
                    collate_fn=floorplan_collate)

    # A minibatch can contain floorplans with different number of blocks or pins
    # All tensors in a minibatch are padded to the maximum number of blocks and pins in the minibatch
    # The padding for the LongTensors is -1, for the bool tensors it is False
    for (tree_data,  # bsz x (n_blocks-1) x 3 . The B*Tree representation of the floorplans
         # bsz x n_blocks x 4. The size(:,:,:2) and the ground truth position(:,:,2:4) of the blocks
         block_sizes_pos,
         pins_pos,  # bsz x n_pins x 2. The pins location
         b2b_connectivity,  # bsz x n_blocks x n_blocks
         p2b_connectivity,  # bsz x n_blocks x n_pins
         edge_constraints) in dl:  # bsz x n_blocks x 2. See visualize.py for the interpretation of edge constraints
        print(
            f'tree data: {tree_data.size()}, block_sizes_pos: {block_sizes_pos.size()},pins_pos: {pins_pos.size()}, b2b_connectivity: {b2b_connectivity.size()},p2b_connectivity: {p2b_connectivity.size()}, edge_constraints: {edge_constraints.size()}')
        break

    # Visualize the first element in the dataset
    (tree_data, block_sizes_pos, pins_pos,
     b2b_connectivity, p2b_connectivity,
     edge_constraints) = ds[0]

    visualize_placement(block_sizes_pos[:, :2],
                        block_sizes_pos[:, 2:4],
                        pins_pos,
                        edge_constraints,
                        b2b_connectivity,
                        p2b_connectivity)


if __name__ == "__main__":
    main()
