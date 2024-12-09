import tarfile
import os
import requests
from tqdm import tqdm
import urllib.request as ur

import glob
import torch
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

GBFACTOR = float(1 << 30)

def decide_download(url):
    d = ur.urlopen(url)
    size = int(d.info()["Content-Length"])/(2**30)
    # confirm if larger than 1GB
    if size > 1:
        return input("This will download %.2fGB. Will you proceed? (y/N) " % (size)).lower() == "y"
    else:
        return True

def download_dataset(root):
    url = 'https://huggingface.co/datasets/IntelLabs/FloorSet/resolve/main/PrimeTensorData.tar.gz'
    f_name = os.path.join(root, 'floorplan_primedata.tgz')
    if os.path.exists(f_name):
        file_size = os.path.getsize(f_name)/(2**30)
    if (not os.path.exists(f_name) and decide_download(url)) or (os.path.exists(f_name) and decide_download(url) and file_size < 13 ) :
        data = ur.urlopen(url)
        size = int(data.info()["Content-Length"])
        chunk_size = 1024*1024
        num_iter = int(size/chunk_size) + 2
        downloaded_size = 0
        with open(f_name, 'wb') as f:
            pbar = tqdm(range(num_iter))
            for i in pbar:
                chunk = data.read(chunk_size)
                downloaded_size += len(chunk)
                pbar.set_description("Downloaded {:.2f} GB".format(
                    float(downloaded_size)/(2**30)))
                f.write(chunk)
    else:
        print('Tar file already downloaded...')
    print("Downloaded floorplan data to", f_name)
    print("Unpacking. This may take a while")
    file = tarfile.open(f_name)
    file.extractall(root)
    file.close()
    os.remove(f_name)

def is_dataset_downloaded(root):
    return len(glob.glob(os.path.join(root, 'PrimeTensorData', 'config*'))) >= 100


def floorplan_collate(batch):
    area_target = [item['input'][0] for item in batch]
    b2b_connectivity = [item['input'][1] for item in batch]
    p2b_connectivity = [item['input'][2] for item in batch]
    pins_pos = [item['input'][3] for item in batch]
    placement_constraints = [item['input'][4] for item in batch]


    fp_sol = [item['label'][0] for item in batch]
    metrics_sol = [item['label'][1] for item in batch]


    def pad_polygons(sol):
        # Determine the maximum number of tensors in any list
        max_length = max(len(tensor_list) for tensor_list in sol)
        
        # Set target size for padding
        target_rows = 14
        target_cols = 2
        
        # List to store the padded tensors for each list
        all_group_padded_tensors = []
    
        # Iterate over each list of tensors in sol
        for tensor_list in sol:
            # List to store padded tensors within the current list
            padded_tensors = []
            
            # Pad each tensor to have target_rows
            for tensor in tensor_list:
                # Calculate the padding required
                pad_rows = target_rows - tensor.size(0)
                
                # Create the padding tuple (left, right, top, bottom)
                pad = (0, 0, 0, pad_rows)  # (left, right, top, bottom)
                
                # Pad the tensor using F.pad
                padded_tensor = F.pad(tensor, pad, value=-1)
                
                # Append the padded tensor to the list
                padded_tensors.append(padded_tensor)
            
            # If there are fewer tensors than max_length, pad the list
            while len(padded_tensors) < max_length:
                # Create a tensor of size (target_rows, target_cols) filled with -1
                empty_tensor = torch.full((target_rows, target_cols), -1)
                padded_tensors.append(empty_tensor)
            
            # Stack the padded tensors for the current list
            group_tensor = torch.stack(padded_tensors)
            
            # Append the group's tensor to the final list
            all_group_padded_tensors.append(group_tensor)
        
        # Stack all group tensors into a single tensor
        final_tensor = torch.stack(all_group_padded_tensors)
        
        return final_tensor
    
    # Pad polygon tensors
    def pad_polygons_old(sol):
        # Set the target size for padding (worst-case 14 edges)
        target_size = (14, 2)
        
        # List to store the padded tensors for each group
        group_padded_tensors = []
        
        # Iterate over each group of tensors in sol
        for tensor_list in sol:
            # List to store padded tensors within the current group
            padded_tensors = []
            
            # Pad each tensor in the current group
            for tensor in tensor_list:
                # Calculate the required padding
                pad_rows = target_size[0] - tensor.size(0)
                pad_cols = target_size[1] - tensor.size(1)
                
                # Ensure pad_cols is zero since all have the same column size
                assert pad_cols == 0, "Unexpected column mismatch."
                
                # Create the padding tuple (left, right, top, bottom)
                pad = (0, 0, 0, pad_rows)
                
                # Pad the tensor using F.pad
                padded_tensor = F.pad(tensor, pad, value=-1)
                
                # Append the padded tensor to the group's list
                padded_tensors.append(padded_tensor)
            
            # Stack the padded tensors for the current group
            group_tensor = torch.stack(padded_tensors)
            
            # Append the group's tensor to the final list
            group_padded_tensors.append(group_tensor)
        
        # Stack all group tensors into a single tensor
        stacked_tensor = torch.stack(group_padded_tensors)

        return stacked_tensor



    
    def pad_inputs(tens_list):
        ndims = tens_list[0].ndim
        max_dims = [max(x.size(dim) for x in tens_list)
                    for dim in range(ndims)]
        padded_tensors = []
        for tens in tens_list:
            padding_tuple = tuple(x for d in range(ndims)
                                  for x in (max_dims[d] - tens.size(d), 0))
            if tens.dtype == torch.bool:
                pad_value = False
            else:
                pad_value = -1
            padded_tensors.append(
                F.pad(tens, padding_tuple[::-1], value=pad_value))
        return torch.stack(padded_tensors)

    return list(map(pad_inputs, (area_target, b2b_connectivity, p2b_connectivity,
                                     pins_pos, placement_constraints))), [pad_polygons(fp_sol), torch.stack(metrics_sol)]
    


class FloorplanDatasetPrime(Dataset):
    def __init__(self, root):
        if not is_dataset_downloaded(root):
            download_dataset(root)
        self.all_input_files = []
        self.all_label_files = []
        partition_range = range(21, 121) #number of partitions in prime
        identifier_range = range(1, 11)  # Identifiers from 1 to 10
        for worker_idx in partition_range:
            config_dir = os.path.join(root, f'PrimeTensorData/config_{worker_idx}')
            # Collect data files within the specified identifier range
            for identifier in identifier_range:
                input_file_pattern = os.path.join(config_dir, f'primedata_{identifier}.pth')
                label_file_pattern = os.path.join(config_dir, f'primelabel_{identifier}.pth')
                if os.path.isfile(input_file_pattern):
                    self.all_input_files.append(input_file_pattern)
                if os.path.isfile(label_file_pattern):
                    self.all_label_files.append(label_file_pattern)
                    
        self.layouts_per_file = 1000
        self.cached_file_idx = -1

    def __len__(self):
        return len(self.all_input_files) * self.layouts_per_file

    def __getitem__(self, idx):
        file_idx, layout_idx = divmod(idx, self.layouts_per_file)
        if file_idx != self.cached_file_idx:
            self.cached_input_file_contents = torch.load(self.all_input_files[file_idx])
            self.cached_label_file_contents = torch.load(self.all_label_files[file_idx])
            self.cached_file_idx = file_idx
            
        area_target = self.cached_input_file_contents[layout_idx][0][:,0]
        placement_constraints = self.cached_input_file_contents[layout_idx][0][:,1:]
        b2b_connectivity = self.cached_input_file_contents[layout_idx][1]
        p2b_connectivity = self.cached_input_file_contents[layout_idx][2]
        pins_pos = self.cached_input_file_contents[layout_idx][3]


        fp_sol = self.cached_label_file_contents[layout_idx][1]
        metrics_sol = self.cached_label_file_contents[layout_idx][0]

        input_data = (area_target, b2b_connectivity, p2b_connectivity, pins_pos, placement_constraints)
        label_data = (fp_sol, metrics_sol)
        sample = {'input': input_data, 'label': label_data}
        return sample
        #return (area_target, b2b_connectivity, p2b_connectivity, pins_pos, placement_constraints)
