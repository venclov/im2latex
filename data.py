from os.path import join
import os

from torch.utils.data import Dataset
import torch


class Im2LatexDataset(Dataset):
    def __init__(self, data_dir, split, max_len):
        """args:
        data_dir: root dir storing the prepoccessed data
        split: train, validate or test
        """
        assert split in ["train", "validate", "test"]
        self.data_dir = data_dir
        self.split = split
        self.max_len = max_len
        # self.pairs = self._load_pairs()
        files_in_data_dir = os.listdir(self.data_dir)
        self.split_files = [file_name for file_name in files_in_data_dir if file_name.startswith(self.split)]

    def __getitem__(self, index):
        loaded_pair = torch.load(self.data_dir, self.split_files[index])
        a_pair = (loaded_pair[0], " ".join(loaded_pair[1].split()[:self.max_len]))
        return a_pair

    def __len__(self):
        return len(self.split_files)
