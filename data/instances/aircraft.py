import os
from pathlib import Path
import numpy as np
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader


class AircraftDataset(Dataset):
    """
        FGCV-Aircraft Dataset.
        <http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/>
        Reference: https://github.com/lvyilin/pytorch-fgvc-dataset/blob/master/aircraft.py
    """
    def __init__(self, root=None, transform=None, split="train", class_type="variant"):
        self.root = os.path.join(root, "fgvc-aircraft-2013b", "data")
        self.img_folder = os.path.join(self.root, "images")
        self.transform = transform
        self.img_loader = default_loader
        assert split in ["train", "val", "trainval", "test"], "Invalid split name"
        self.split = split
        assert class_type in ["variant", "family", "manufacturer"], "Invalid class type"
        self.class_type = class_type

        self.classes_file = os.path.join(self.root, f"images_{self.class_type}_{self.split}.txt")
        (image_ids, targets, classes, class_to_idx) = self.find_classes()
        samples = self.make_dataset(image_ids, targets)

        self.samples = samples
        self.targets = targets
        self.class_names = [process_string(c) for c in classes]
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        self.super_class = "aircraft"

    def __getitem__(self, index):
        path, target = self.samples[index]
        file_name = Path(path).stem
        img = self.img_loader(path)
        if self.transform is not None:
            img = self.transform(img)

        return {"x": img, "y": target, "class_name": self.class_names[target], "file_name": file_name}

    def __len__(self):
        return len(self.samples)

    def get_num_classes(self):
        return len(self.class_names)

    def find_classes(self):
        # read classes file, separating out image IDs and class names
        image_ids = []
        targets = []
        with open(self.classes_file, 'r') as f:
            for line in f:
                split_line = line.split(' ')
                image_ids.append(split_line[0])
                targets.append(' '.join(split_line[1:]))

        # index class names
        classes = np.unique(targets)
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        targets = [class_to_idx[c] for c in targets]
        # print(f"find {len(classes)} {self.class_type}")
        return image_ids, targets, classes, class_to_idx

    def make_dataset(self, image_ids, targets):
        assert (len(image_ids) == len(targets))
        images = []
        for i in range(len(image_ids)):
            item = (os.path.join(self.img_folder,
                                 '%s.jpg' % image_ids[i]), targets[i])
            images.append(item)
        return images

def process_string(s: str) -> str:
    s = s.rstrip('\n')
    s = s.replace(' ', '_')
    s = s.replace('/', '_')
    return s