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
    size = int(d.info()["Content-Length"])/GBFACTOR
    # confirm if larger than 1GB
    if size > 1:
        return input("This will download %.2fGB. Will you proceed? (y/N) " % (size)).lower() == "y"
    else:
        return True


def download_dataset(root):
    url = 'https://huggingface.co/datasets/IntelLabs/FloorSet/resolve/main/LiteTensorData_v2.tar.gz'
    f_name = os.path.join(root, 'LiteTensorData_v2.tar.gz')
    if not os.path.exists(f_name) and decide_download(url):
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
                    float(downloaded_size)/GBFACTOR))
                f.write(chunk)
    print("Downloaded floorplan data to", f_name)
    print("Unpacking. This may take a while")
    file = tarfile.open(f_name)
    file.extractall(root)
    file.close()
    os.remove(f_name)


def is_dataset_downloaded(root):
    return len(glob.glob(os.path.join(root, 'floorset_lite', 'worker*'))) >= 100


class FloorplanDatasetLite(Dataset):
    def __init__(self, root):
        if not is_dataset_downloaded(root):
            download_dataset(root)
        self.all_files = []
        for worker_idx in range(100):
            self.all_files.extend(glob.glob(os.path.join(
                root, 'floorset_lite', f"worker_{worker_idx}/layouts*")))
        self.layouts_per_file = 112
        self.cached_file_idx = -1

    def __len__(self):
        return len(self.all_files) * self.layouts_per_file

    def __getitem__(self, idx):
        file_idx, layout_idx = divmod(idx, self.layouts_per_file)
        if file_idx != self.cached_file_idx:
            self.cached_input_file_contents = torch.load(self.all_files[file_idx])
            self.cached_file_idx = file_idx
            self.cached_layout_idx = layout_idx

        area_target = self.cached_input_file_contents[0][layout_idx][:,0]
        placement_constraints = self.cached_input_file_contents[0][layout_idx][:,1:]
        b2b_connectivity = self.cached_input_file_contents[1][layout_idx]
        p2b_connectivity = self.cached_input_file_contents[2][layout_idx]
        pins_pos = self.cached_input_file_contents[3][layout_idx]

        tree_sol = self.cached_input_file_contents[4][layout_idx]
        fp_sol = self.cached_input_file_contents[5][layout_idx]
        metrics_sol = self.cached_input_file_contents[6][layout_idx]

        input_data = (area_target, b2b_connectivity, p2b_connectivity, pins_pos, placement_constraints)
        label_data = (tree_sol, fp_sol, metrics_sol)
        sample = {'input': input_data, 'label': label_data}
        return sample

def floorplan_collate(batch):
    area_target = [item['input'][0] for item in batch]
    b2b_connectivity = [item['input'][1] for item in batch]
    p2b_connectivity = [item['input'][2] for item in batch]
    pins_pos = [item['input'][3] for item in batch]
    placement_constraints = [item['input'][4] for item in batch]

    tree_sol = [item['label'][0] for item in batch] 
    fp_sol = [item['label'][1] for item in batch]
    metrics_sol = [item['label'][2] for item in batch]

    def pad_to_largest(tens_list):

        ndims = tens_list[0].ndim
        # print('shape:', tens_list[0].shape)
        # if tens_list[0].shape[0] == 25:
        #     for l in range(len(tens_list)):
        #         if l > 510:
        #             print(l, tens_list[l].shape)
        #             print(tens_list[l])
            
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

    return list(map(pad_to_largest, (area_target, b2b_connectivity, p2b_connectivity, pins_pos, placement_constraints, tree_sol, fp_sol, metrics_sol)))
