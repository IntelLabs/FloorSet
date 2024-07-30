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
    url = 'https://huggingface.co/datasets/IntelLabs/FloorSet/resolve/main/floorset_lite.tgz'
    f_name = os.path.join(root, 'floorplan_data.tgz')
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


class FloorplanDataset(Dataset):
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
            self.cached_file_contents = list(
                torch.load(self.all_files[file_idx])[0])
            self.cached_file_contents[6] = self.cached_file_contents[6].repeat(
                112, 1)
            self.cached_file_contents[7] = self.cached_file_contents[6].repeat(
                112, 1)
            self.cached_file_idx = file_idx

        (tree_data, block_sizes_pos, pins_pos, b2b_connectivity, p2b_connectivity, edge_constraints,
         preplaced, fixed, tied_ar_ids, group_mask) = map(lambda x: x[layout_idx], self.cached_file_contents)

        return (tree_data, block_sizes_pos, pins_pos, b2b_connectivity, p2b_connectivity, edge_constraints,
                preplaced, fixed, tied_ar_ids, group_mask)


def floorplan_collate(all_fps):
    (tree_data, block_sizes_pos, pins_pos, b2b_connectivity, p2b_connectivity, edge_constraints,
     preplaced, fixed, tied_ar_ids, group_mask) = list(zip(*all_fps))

    def pad_to_largest(tens_list):

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

    return list(map(pad_to_largest, (tree_data, block_sizes_pos, pins_pos, b2b_connectivity, p2b_connectivity, edge_constraints,
                                     preplaced, fixed, tied_ar_ids, group_mask)))
