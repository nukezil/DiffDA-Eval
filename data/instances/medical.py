import os
from pathlib import Path
from torch.utils.data import Dataset
import torchvision.datasets as datasets


class MedicalDataset(Dataset):
    def __init__(self, cfg, transform=None, split="train"):
        root = cfg.data.root
        name, shot = str(cfg.data.dataset).split("-")
        if split == "train":
            self.dset = datasets.ImageFolder(os.path.join(root, f"{name}/shot{shot}/train"), transform=transform)
        else:
            self.dset = datasets.ImageFolder(os.path.join(root, f"{name}/shot{shot}/test"), transform=transform)
        self.targets = [item[1] for item in self.dset.imgs]
        self.class_to_idx = self.dset.class_to_idx
        self.class_names = {idx: class_name for class_name, idx in self.class_to_idx.items()}

    def get_num_classes(self):
        return len(self.class_to_idx)

    def __len__(self):
        return len(self.dset)

    def __getitem__(self, index):
        file_name = Path(self.dset.imgs[index][0]).stem
        img, target = self.dset[index]
        return {"x": img, "y": target, "class_name": self.class_names[target], "file_name": file_name}