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
    (area_target, b2b_connectivity, p2b_connectivity, pins_pos,
     placement_constraints) = list(zip(*batch))
    
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

    return list(map(pad_to_largest, (area_target, b2b_connectivity, p2b_connectivity,
                                     pins_pos, placement_constraints)))

class FloorplanDataset(Dataset):
    def __init__(self, root):
        if not is_dataset_downloaded(root):
            download_dataset(root)
        self.all_files = []
        for worker_idx in range(21, 121):
            self.all_files.extend(glob.glob(os.path.join(
                root, 'PrimeTensorData', f"config_{worker_idx}/primedata*")))
        self.layouts_per_file = 1000
        self.cached_file_idx = -1

    def __len__(self):
        return len(self.all_files) * self.layouts_per_file

    def __getitem__(self, idx):
        file_idx, layout_idx = divmod(idx, self.layouts_per_file)
        if file_idx != self.cached_file_idx:
            self.cached_file_contents = torch.load(self.all_files[file_idx])
            self.cached_file_idx = file_idx

        area_target = self.cached_file_contents[layout_idx][0][:,0]
        placement_constraints = self.cached_file_contents[layout_idx][0][:,1:]
        b2b_connectivity = self.cached_file_contents[layout_idx][1]
        p2b_connectivity = self.cached_file_contents[layout_idx][2]
        pins_pos = self.cached_file_contents[layout_idx][3]

        return (area_target, b2b_connectivity, p2b_connectivity, pins_pos, placement_constraints)
