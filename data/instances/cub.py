import os
from pathlib import Path
import pandas as pd
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader


class CUBDataset(Dataset):
    """
        CUB-200-2011 Dataset.
        <http://www.vision.caltech.edu/visipedia/CUB-200-2011.html>
        Reference: https://github.com/lvyilin/pytorch-fgvc-dataset/blob/master/cub2011.py
    """
    def __init__(self, root=None, transform=None, split="train"):
        self.root = os.path.join(root, "CUB_200_2011")
        self.transform = transform
        self.img_loader = default_loader
        assert split in ["train", "test"], "Invalid split name"
        self.split = split

        images = pd.read_csv(os.path.join(self.root, "images.txt"), sep=" ", names=["img_id", "filepath"])
        image_class_labels = pd.read_csv(os.path.join(self.root, "image_class_labels.txt"), sep=" ",
                                         names=["img_id", "target"])
        train_test_split = pd.read_csv(os.path.join(self.root, "train_test_split.txt"), sep=" ",
                                       names=["img_id", "is_training_img"])
        data = images.merge(image_class_labels, on="img_id")
        self.data = data.merge(train_test_split, on="img_id")

        class_names = pd.read_csv(os.path.join(self.root, "classes.txt"), sep=" ",
                                  names=["class_name"])
        self.class_names = [name.split(".")[1] for name in class_names["class_name"].tolist()]
        self.super_class = "bird"

        if self.split == "train":
            self.data = self.data[self.data.is_training_img == 1]
        elif self.split == "test":
            self.data = self.data[self.data.is_training_img == 0]
        else:
            raise ValueError("Invalid split name")
        self.targets = [t - 1 for t in self.data["target"].tolist()]
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}

    def __len__(self):
        return len(self.data)

    def get_num_classes(self):
        return len(self.class_names)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        path = os.path.join(self.root, "images", row.filepath)
        file_name = Path(path).stem
        img = self.img_loader(path)
        target = self.targets[index]
        if self.transform is not None:
            img = self.transform(img)

        return {"x": img, "y": target, "class_name": self.class_names[target], "file_name": file_name}