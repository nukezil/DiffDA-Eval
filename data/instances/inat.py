import os
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader

def make_dataset(root, split):
    if not split in ["l_train", "l_train_val", "u_train_in", "u_train_out", "val", "test"]:
        raise ValueError("Unknown split name: {}".format(split))
    split_file = os.path.join(root, "splits", f"{split}.txt")
    if not os.path.exists(split_file):
        raise ValueError("Split file not found: {}".format(split_file))
    with open(split_file, "r") as f:
        lines = f.readlines()

    image_paths, targets = [], []
    for line in lines:
        parts = line.strip().split(" ")  # handle cases where there are spaces in path
        image_path, target = " ".join(parts[:-1]), int(parts[-1])
        image_paths.append(os.path.join(root, image_path))
        targets.append(target)
        assert os.path.exists(image_paths[-1]), "Image file not found: {}".format(image_paths[-1])

    return image_paths, targets


def parse_class_names(file_path, name):
    class_name_dict = {-1: "Unknown"}

    with open(file_path, "r") as f:
        lines = f.readlines()
    for line in lines:
        if name == "semi_aves":
            class_idx, class_name = line.strip().split(" ")
        elif name == "semi_fungi":
            class_name, class_idx = line.strip().split("_")
        else:
            raise ValueError("Unknown data name: {}".format(name))
        class_name_dict[int(class_idx)] = class_name

    return class_name_dict


class iNatDataset(Dataset):
    def __init__(self, root, name="semi_aves", transform=None, split="l_train_val"):
        self.root = os.path.join(root, name)
        self.transform = transform
        self.img_loader = default_loader
        self.images_paths, self.targets = make_dataset(self.root, split)
        self.class_name_dict = parse_class_names(os.path.join(self.root, "splits", f"{name}_class_names.txt"), name)
        if name == "semi_aves":
            self.super_class = "bird"
        elif name == "semi_fungi":
            self.super_class = "fungus"
        else:
            raise ValueError("Unknown data name: {}".format(name))

    def __getitem__(self, index):
        image_path, target = self.images_paths[index], self.targets[index]
        image = self.img_loader(image_path)

        if self.transform is not None:
            image = self.transform(image)

        return {'x': image, 'y': target, 'class_name': self.class_name_dict[target]}

    def __len__(self):
        return len(self.images_paths)

    def get_num_classes(self):
        return len(self.class_name_dict) - 1  # remove 'Unknown' class (-1)

    def get_class_names(self):
        class_names = [self.class_name_dict[i] for i in range(self.get_num_classes())]
        return class_names


def get_inat_dataset(dataset_name, root, split, transform):
    root = os.path.join(root, "semi_fgvc")

    # 处理 train 分割的特殊情况
    if split == "train":
        split = "l_train_val"

    return iNatDataset(root=root, name=dataset_name, transform=transform, split=split)
