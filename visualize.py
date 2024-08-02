import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch, Circle
import matplotlib.patches as patches
from shapely.geometry import Polygon, LineString, Point
import copy

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

def get_hard_color(constraint):
    # Define colors for each type of constraint
    colors = {
        'default': ("silver", "no constraint"),
        'boundary': ("olive", "boundary"),
        'fixed': ("violet", "fixed"),
        'preplaced': ("gray", "preplaced"),
        'group': ("red", "cluster"),
        'mib': ("darkgreen", "MIB")
    }

    # Determine the face color and label based on constraints
    if constraint[3]:
        return colors['group']
    elif constraint[0]:
        return colors['fixed']
    elif constraint[1]:
        return colors['preplaced']
    elif constraint[2]:
        return colors['mib']
    elif constraint[4] > 1:
        return colors['boundary']
    else:
        return colors['default']

def visualize_lite(fp_sol, b2b_connectivity, p2b_connectivity, pins_pos, placement_constraints, lind=0):
    fig, ax = plt.subplots()
    default_color = 'silver'
    edge_color = 'black'
    
    # Initialize dimensions
    W, H = 0, 0
    
    # Plot floorplan solution polygons
    all_poly_dict = {}
    for ind, elem in enumerate(fp_sol):
        x = elem[2]
        y = elem[3]
        w = elem[0]
        h = elem[1]
        polygon_list = [(x, y), (x+w, y), (x+w, y+h), (x, y+h), (x, y)]
        unpadded_polygon_list = [point for point in polygon_list if point != [-1.0, -1.0]]
        if len(unpadded_polygon_list) < 4:#ignore padded polygons that are dummy
            continue
        ##poly_elem = Polygon(elem.tolist())
        poly_elem = Polygon(unpadded_polygon_list)
        all_poly_dict[ind] = poly_elem
        hard_const = placement_constraints[ind]
        
        face_color, label_text = get_hard_color(hard_const)
        patch = patches.Polygon(
            list(poly_elem.exterior.coords),
            closed=True,
            fill=True,
            edgecolor=edge_color,
            facecolor=face_color,
            label=label_text,
            alpha=0.3
        )
        ax.add_patch(patch)
        
        llx, lly = poly_elem.bounds[0], poly_elem.bounds[1]
        urx, ury = poly_elem.bounds[2], poly_elem.bounds[3]
        W = max(W, urx)
        H = max(H, ury)
        
        ax.annotate(str(ind + 1), (llx, lly), fontsize=6)
    
    # Plot pin positions
    for pname in range(pins_pos.shape[0]):
        x, y = pins_pos[pname]
        circ = Circle((x, y), radius=1, color='g')
        ax.add_patch(circ)
    
    # Plot block-to-block (B2B) connectivity (0-index)
    for src_block, dst_block in b2b_connectivity[:, :2]:
        src_block, dst_block = int(src_block.item()), int(dst_block.item())
        if src_block != -1 and dst_block != -1:
            poly_elem1 = all_poly_dict[src_block] #Polygon(fp_sol[src_block].tolist())
            poly_elem2 = all_poly_dict[dst_block]#Polygon(fp_sol[dst_block].tolist())
            llx1, lly1 = poly_elem1.bounds[0], poly_elem1.bounds[1]
            llx2, lly2 = poly_elem2.bounds[0], poly_elem2.bounds[1]
            plt.plot((llx1, llx2), (lly1, lly2), color='r', linewidth=0.1)
    
    # # Plot pin-to-block (P2B) connectivity
    for src_block, dst_block in p2b_connectivity[:, :2]:
        src_block, dst_block = int(src_block.item()), int(dst_block.item())
        if src_block != -1 and dst_block != -1:
            poly_elem2 = all_poly_dict[dst_block]#Polygon(fp_sol[dst_block - 1].tolist())
            llx2, lly2 = poly_elem2.bounds[0], poly_elem2.bounds[1]
            plt.plot(
                (pins_pos[src_block][0], llx2),
                (pins_pos[src_block][1], lly2),
                color='b',
                linewidth=0.1
            )
    
    # Set plot limits and labels
    plt.xlim(0, W * 1.25)
    plt.ylim(0, H * 1.25)
    ax.set_aspect('equal', adjustable='box')
    plt.title('Baseline Layout ' + str(lind))
    
    # Create legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper right', title='Placement Constraints', fontsize=6)
    #plt.savefig('./images_lite/Layout_'+str(lind)+'.png')
    #plt.close()
    plt.show()



def visualize_prime(fp_sol, b2b_connectivity, p2b_connectivity, pins_pos, placement_constraints, lind=0):
    fig, ax = plt.subplots()
    default_color = 'silver'
    edge_color = 'black'
    
    # Initialize dimensions
    W, H = 0, 0
    
    # Plot floorplan solution polygons
    all_poly_dict = {}
    for ind, elem in enumerate(fp_sol):
        polygon_list = elem.tolist()
        unpadded_polygon_list = [point for point in polygon_list if point != [-1.0, -1.0]]
        if len(unpadded_polygon_list) < 4:#ignore padded polygons that are dummy
            continue
        ##poly_elem = Polygon(elem.tolist())
        poly_elem = Polygon(unpadded_polygon_list)
        all_poly_dict[ind] = poly_elem
        hard_const = placement_constraints[ind]
        
        face_color, label_text = get_hard_color(hard_const)
        patch = patches.Polygon(
            list(poly_elem.exterior.coords),
            closed=True,
            fill=True,
            edgecolor=edge_color,
            facecolor=face_color,
            label=label_text,
            alpha=0.3
        )
        ax.add_patch(patch)
        
        llx, lly = poly_elem.bounds[0], poly_elem.bounds[1]
        urx, ury = poly_elem.bounds[2], poly_elem.bounds[3]
        W = max(W, urx)
        H = max(H, ury)
        
        ax.annotate(str(ind + 1), (llx, lly), fontsize=6)
    
    # Plot pin positions
    for pname in range(pins_pos.shape[0]):
        x, y = pins_pos[pname]
        circ = Circle((x, y), radius=1, color='g')
        ax.add_patch(circ)
    
    # Plot block-to-block (B2B) connectivity (0-index)
    for src_block, dst_block in b2b_connectivity[:, :2]:
        src_block, dst_block = int(src_block.item()), int(dst_block.item())
        if src_block != -1 and dst_block != -1:
            poly_elem1 = all_poly_dict[src_block] #Polygon(fp_sol[src_block].tolist())
            poly_elem2 = all_poly_dict[dst_block]#Polygon(fp_sol[dst_block].tolist())
            llx1, lly1 = poly_elem1.bounds[0], poly_elem1.bounds[1]
            llx2, lly2 = poly_elem2.bounds[0], poly_elem2.bounds[1]
            plt.plot((llx1, llx2), (lly1, lly2), color='r', linewidth=0.3)
    
    # Plot pin-to-block (P2B) connectivity
    for src_block, dst_block in p2b_connectivity[:, :2]:
        src_block, dst_block = int(src_block.item()), int(dst_block.item())
        if src_block != -1 and dst_block != -1:
            poly_elem2 = all_poly_dict[dst_block]#Polygon(fp_sol[dst_block - 1].tolist())
            llx2, lly2 = poly_elem2.bounds[0], poly_elem2.bounds[1]
            plt.plot(
                (pins_pos[src_block][0], llx2),
                (pins_pos[src_block][1], lly2),
                color='b',
                linewidth=0.1
            )
    
    # Set plot limits and labels
    plt.xlim(0, W * 1.5)
    plt.ylim(0, H * 1.5)
    plt.title('Baseline Layout ' + str(lind))
    
    # Create legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper right', title='Placement Constraints', fontsize=6)
    #plt.savefig('./images_prime/Layout_'+str(lind)+'.png')
    #plt.close()
    plt.show()

def visualize_placement(block_sizes,  # n_blocks x 2
                        block_xy,  # n_blocks x 2
                        pins_xy,  # n_pins x 2
                        edge_constraints,  # n_blocks x 2
                        groups,  # n_blocks
                        tied_ar_ids,
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

        cluster = groups[bname].item()
        if cluster == 0:
            hatch = {}
        else:
            hatch = {'hatch' : ['/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*'][cluster-1]}

        edgecolor = 'red' if tied_ar_ids[bname] else 'black'
        rect = Rectangle((x, y), w, h, fill=True,
                         facecolor=CONSTR_TO_COLOR[constr_name], edgecolor=edgecolor,**hatch)
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
