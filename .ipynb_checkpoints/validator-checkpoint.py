import torch
from torch.utils.data import DataLoader
from prime_dataset import FloorplanDataset, floorplan_collate
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch, Circle
import matplotlib.patches as patches
from shapely.geometry import Polygon, LineString, Point
import copy
from visualize import get_hard_color, visualize_prime
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from prime_dataset import FloorplanDataset, floorplan_collate
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch, Circle
import matplotlib.patches as patches
from shapely.geometry import Polygon, LineString, Point
import copy
from visualize import get_hard_color, visualize_prime
from tqdm import tqdm

def estimate_cost(bdata, layout_index):


    #read baseline design (if it not donwloaded, need to spend time in donwloading for the first call)
    root = './'
    ds = FloorplanDataset(root)
    target_data = ds.__getitem__(layout_index)

    target_area_budgets = target_data['input'][0]
    target_b2b_edges = target_data['input'][1]
    target_p2b_edges = target_data['input'][2]
    target_pins_pos = target_data['input'][3]
    target_constraints = target_data['input'][4]
    target_metrics = target_data['label'][1]
    target_poly = target_data['label'][0]

    target_b2b_wl = target_metrics[-2].item()
    target_p2b_wl = target_metrics[-1].item()
    target_layout_area = target_metrics[0].item()

    if len(bdata) != len(target_poly):
        print('ERROR: incorrect number of polygons in the solution')
        exit

    #read solution file
    W = 0
    H = 0
    min_x, min_y, max_x, max_y = float('inf'), float('inf'), 0, 0

    #estimate area and validated target area budgets
    sol_area_budgets = torch.zeros_like(target_area_budgets)
    fp_sol = []
    ind  = 0
    for elem in bdata:#list with num_blocks polygons, in each sample
        poly_elem = Polygon(elem.tolist())
        coords_list = list(poly_elem.exterior.coords)
        poly_area = poly_elem.area
        sol_area_budgets[ind] = poly_area
        fp_sol.append(poly_elem)
        
        min_x, min_y, max_x, max_y = poly_elem.bounds
        if max_x > W:
            W = copy.copy(max_x)
        if max_y > H:
            H = copy.copy(max_y)
        ind += 1
        
    sol_area_cost = float(W*H)
    #needs to be zero or positive but not negative
    delta_budgets = sol_area_budgets -  target_area_budgets

    area_viol = torch.nonzero(delta_budgets < 0).squeeze()
    
    

    #estimate wirelength

    
    # # Calculate centroids using list comprehension and torch tensors
    centroids = torch.tensor(
        [list(Polygon(fp_sol[i]).centroid.coords[0]) for i in range(len(fp_sol))], dtype=torch.float32
    )
    
    # **Vectorized Calculation for b2b_wl**
    # Extract indices for b2b_edges
    b2b_indices_0 = target_b2b_edges[:, 0].long()
    b2b_indices_1 = target_b2b_edges[:, 1].long()
    
    if target_b2b_edges.shape[1] > 2:
        # If the third column exists, use it as b2b_weights
        b2b_weights = target_b2b_edges[:, 2]
    else:
        # If the third column does not exist, create a tensor of ones
        b2b_weights = torch.ones(target_b2b_edges.shape[0])
    
    # Calculate the differences in centroids for b2b_edges
    diff_x_b2b = torch.abs(centroids[b2b_indices_1, 0] - centroids[b2b_indices_0, 0])
    diff_y_b2b = torch.abs(centroids[b2b_indices_1, 1] - centroids[b2b_indices_0, 1])
    
    # Calculate weighted Manhattan distances for b2b_edges
    sol_b2b_wl = torch.sum((diff_x_b2b + diff_y_b2b) * b2b_weights).item()

    # Extract indices for p2b_edges
    p2b_indices_0 = target_p2b_edges[:, 0].long()
    p2b_indices_1 = target_p2b_edges[:, 1].long()

    if target_p2b_edges.shape[1] > 2:
        # If the third column exists, use it as p2b_weights
        p2b_weights = target_p2b_edges[:, 2]
    else:
        # If the third column does not exist, create a tensor of ones
        p2b_weights = torch.ones(target_p2b_edges.shape[0])

    # Get px and py from D[k][3] using indices 
    px_py = target_pins_pos[p2b_indices_0]
    px = px_py[:, 0]
    py = px_py[:, 1]
    
    # Calculate the differences for p2b_edges
    diff_x_p2b = torch.abs(centroids[p2b_indices_1, 0] - px)
    diff_y_p2b = torch.abs(centroids[p2b_indices_1, 1] - py)
    
    # Calculate weighted Manhattan distances for p2b_edges
    sol_p2b_wl = torch.sum((diff_x_p2b + diff_y_p2b) * p2b_weights).item()

    ##print('Statistics: ', sol_b2b_wl, sol_p2b_wl, sol_area_cost)
    print('area /wl difference:', sol_b2b_wl - target_b2b_wl, sol_p2b_wl - target_p2b_wl, sol_area_cost - target_layout_area)
    #print(target_area_budgets)
    #print(sol_area_budgets)

    print('Partition indices with area-budget violations:', area_viol.tolist())
    
    #estimate constraintc-cost
    fixed_const = target_constraints[:,0]
    preplaced_const = target_constraints[:,1]
    mib_const = target_constraints[:,2]
    clust_const = target_constraints[:,3]
    bound_const  = target_constraints[:,4]

    print(fixed_const, preplaced_const, mib_const)
    print(clust_const, bound_const)



