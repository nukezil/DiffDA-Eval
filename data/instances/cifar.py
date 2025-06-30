import os
from pathlib import Path
from torch.utils.data import Dataset
import torchvision.datasets as datasets


class CIFAR100Dataset(Dataset):
    def __init__(self, root, transform=None, split="train"):
        if split == "train":
            self.dset = datasets.ImageFolder(os.path.join(root, "CIFAR100/train-subset"), transform=transform)
        else:
            self.dset = datasets.CIFAR100(os.path.join(root, "CIFAR100"), train=False, transform=transform)
        if split == "train":
            self.targets = [item[1] for item in self.dset.imgs]
        self.split = split
        self.class_to_idx = self.dset.class_to_idx
        self.class_names = {idx: class_name for class_name, idx in self.class_to_idx.items()}

    def get_num_classes(self):
        return 100

    def __len__(self):
        return len(self.dset)

    def __getitem__(self, index):
        if self.split == "train":
            file_name = Path(self.dset.imgs[index][0]).stem
        else:
            file_name = ""
        img, target = self.dset[index]
        return {"x": img, "y": target, "class_name": self.class_names[target], "file_name": file_name}
