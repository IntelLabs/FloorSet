import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch, Circle


# two bits per position
# bit 1: 0 - no specific position, 1: top: 2: bottom
# bit 2: 0 - no specific side, 1: left, 2: right
# edge-only constraints are single bit [1,0], [2,0]
# corner constraints have two bits, eg [1,1] is top left corner
# integer IDs for embedding matrices
EDGE_TO_ID = {
    "co": torch.LongTensor([0, 0]),  # no constraints
    "be": torch.LongTensor([2, 0]),  # bottom edge
    "le": torch.LongTensor([0, 1]),  # left edge
    "te": torch.LongTensor([1, 0]),  # top edge
    "re": torch.LongTensor([0, 2]),  # right edge
    "bl": torch.LongTensor([2, 1]),  # bottom-left corner
    "br": torch.LongTensor([2, 2]),  # bottom-right corner
    "tl": torch.LongTensor([1, 1]),  # top-left corner
    "tr": torch.LongTensor([1, 2]),  # top-right corner
}

ID_TO_EDGE = {tuple(v.tolist()): k for k, v in EDGE_TO_ID.items()}


CONSTR_TO_COLOR = {
    "co": "coral",  # no constraints
    "be": "goldenrod",  # bottom edge
    "le": "lightsteelblue",  # left edge
    "te": "deepskyblue",  # top edge
    "re": "yellowgreen",  # right edge
    "bl": "brown",  # bottom-left corner
    "br": "beige",  # bottom-right corner
    "tl": "mediumpurple",  # top-left corner
    "tr": "pink",  # top-right corner
}

CONSTR_TO_STR = {
    "coral": "nop",  # no constraints
    "goldenrod": "bottom",  # bottom edge
    "lightsteelblue": "left",  # left edge
    "deepskyblue": "top",  # top edge
    "yellowgreen": "right",  # right edge
    "brown": "bottom-left",  # bottom-left corner
    "beige": "bottom-right",  # bottom-right corner
    "mediumpurple": "top-left",  # top-left corner
    "pink": "top-right",  # top-right corner
}


def visualize_placement(block_sizes,  # n_blocks x 2
                        block_xy,  # n_blocks x 2
                        pins_xy,  # n_pins x 2
                        edge_constraints,  # n_blocks x 2
                        b2b_connectivity,  # n_blocks x n_blocks
                        p2b_connectivity):  # n_blocks x n_pins

    n_blocks = block_sizes.size(0)
    n_pins = pins_xy.size(0)

    block_centers = block_sizes / 2 + block_xy

    # We assume all blocks have been placed
    block_placed = torch.BoolTensor(n_blocks).fill_(True)

    p2b_connectivity_masked = torch.logical_and(
        p2b_connectivity, block_placed.unsqueeze(1))
    b2b_connectivity_masked = torch.logical_and(
        b2b_connectivity, (block_placed.unsqueeze(1) * block_placed.unsqueeze(0)))

    W, H = (block_xy + block_sizes).max(dim=0)[0]

    fig, ax2 = plt.subplots(1, 1)

    # visualizing the tree
    for bname in range(n_blocks):
        if not block_placed[bname]:
            continue
        x = block_xy[bname][0]
        y = block_xy[bname][1]
        w = block_sizes[bname][0]
        h = block_sizes[bname][1]
        constr_name = ID_TO_EDGE[tuple(edge_constraints[bname].numpy())]
        rect = Rectangle((x, y), w, h, fill=True,
                         facecolor=CONSTR_TO_COLOR[constr_name], edgecolor='black')
        ax2.add_patch(rect)
        ax2.annotate(str(bname), (x, y), color='black', fontsize=8)

    for pname in range(n_pins):
        x = pins_xy[pname][0]
        y = pins_xy[pname][1]

        circ = Circle((x, y), radius=1, color='g')
        ax2.add_patch(circ)

    for src_block, dst_block in b2b_connectivity_masked.nonzero():
        plt.plot((block_centers[src_block][0], block_centers[dst_block][0]),
                 (block_centers[src_block][1], block_centers[dst_block][1]), color='r', linewidth=0.1)

    for src_block, dst_pin in p2b_connectivity_masked.nonzero():
        plt.plot((block_centers[src_block][0], pins_xy[dst_pin][0]),
                 (block_centers[src_block][1], pins_xy[dst_pin][1]), color='b', linewidth=0.1)

    patches = [Patch(color=k, label=v) for k, v in CONSTR_TO_STR.items()]
    lgd = ax2.legend(handles=patches, loc='center right',
                     bbox_to_anchor=(1.3, 0.5))
    ax2.set_ylim(-0.02, H*(1.2))
    ax2.set_xlim(-0.02, W*(1.2))
    wspace = (W*H) - torch.prod(block_sizes, dim=1).sum()

    block_dist = (block_centers.unsqueeze(
        0) - block_centers.unsqueeze(1)).abs().sum(-1)
    hpwl_block = (block_dist * b2b_connectivity.long()).sum() // 4
    hpwl_pins = ((block_centers.unsqueeze(1) - pins_xy.unsqueeze(0)
                  ).abs().sum(-1) * p2b_connectivity.long()).sum() // 2

    plt.title(
        f'W*H = {W}*{H} = {W*H}, \n wspace fraction of area : {wspace/(W*H):.4f}, HPWL : {hpwl_pins + hpwl_block}')

    plt.show()
