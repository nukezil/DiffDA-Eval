import os
from pathlib import Path
import random
from tqdm import tqdm
from collections import defaultdict
import numpy as np
from omegaconf import OmegaConf
import math

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from .instances.aircraft import AircraftDataset
from .instances.cub import CUBDataset
from .instances.flowers102 import Flowers102Dataset
from .instances.inat import get_inat_dataset
from .instances.cifar import CIFAR100Dataset
from .instances.caltech101 import Caltech101Dataset
from .instances.imagenet100 import ImageNet100Dataset
from .instances.medical import MedicalDataset

class FewShotDataset(Dataset):
    def __init__(self, dset, seed=0, examples_per_class=-1):
        self.dset = dset
        self.all_targets = dset.targets
        self.all_indices = list(range(len(dset)))

        random.seed(seed)
        np.random.seed(seed)

        if examples_per_class > 0:
            target_to_indices = defaultdict(list)
            for i, target in enumerate(self.all_targets):
                target_to_indices[target].append(i)

            self.sub_indices = []
            for target, indices in target_to_indices.items():
                try:
                    sampled_indices = random.sample(indices, examples_per_class)
                except ValueError:
                    print(
                        f"Class {target}: Sample larger than population or is negative, use random.choices instead"
                    )
                    sampled_indices = random.choices(indices, k=examples_per_class)
                self.sub_indices.extend(sampled_indices)
        else:
            self.sub_indices = self.all_indices
        self.class_to_idx = self.dset.class_to_idx
        self.class_names = self.dset.class_names
        self.super_class = getattr(self.dset, "super_class", None)
    def __len__(self):
        return len(self.sub_indices)
    def __getitem__(self, index):
        data_dict = self.dset[self.sub_indices[index]]
        return data_dict

    def get_num_classes(self):
        return self.dset.get_num_classes()


class GeneratedDataset(Dataset):
    def __init__(self, cfg, transform, modified_class_to_idx=None):
        gen_task_path = os.path.join(cfg.generation.root_dir, cfg.data.dataset, cfg.generation.exp)
        print(f"Create GeneratedDataset from {gen_task_path}!")
        self.dset = datasets.ImageFolder(gen_task_path, transform=transform)
        self.transform = transform
        self.class_names = {idx: class_name for class_name, idx in self.dset.class_to_idx.items()}
        self.modified_class_to_idx = modified_class_to_idx  # modify to align with original dataset

    def get_num_classes(self):
        return len(self.class_names)

    def __len__(self):
        return len(self.dset)

    def __getitem__(self, index):
        file_name = Path(self.dset.imgs[index][0]).stem
        img, gen_target = self.dset[index]
        class_name = self.class_names[gen_target]
        if self.modified_class_to_idx:
            target = self.modified_class_to_idx[class_name]
        else:
            target = gen_target
        return {"x": img, "y": target, "class_name": class_name, "file_name": file_name}


class PartialConcatDataset(Dataset):
    def __init__(self, original_dset, generated_dset: GeneratedDataset, cfg):
        self.original_dset = original_dset
        self.num_original_samples = len(original_dset)
        self.generated_dset = generated_dset
        self.generated_root = os.path.join(cfg.generation.root_dir, cfg.data.dataset, cfg.generation.exp)

        self.generated_map = self._build_generated_map()
        self.partial_generated_indices = self._concat_datasets()

    def _build_generated_map(self):
        generated_map = {}
        print("TC: Building partial generated indices...")
        for idx, (path, _) in enumerate(self.generated_dset.dset.imgs):
            rel_path = os.path.relpath(path, self.generated_root)
            class_name, filename = os.path.split(rel_path)
            if "_augment_" not in filename:
                continue
            base_name = filename[:filename.rindex("_augment_")]
            key = (class_name, base_name)
            generated_map.setdefault(key, []).append(idx)
        return generated_map

    def _concat_datasets(self):
        partial_generated_indices = []
        for idx in tqdm(range(self.num_original_samples)):
            sample_dict = self.original_dset[idx]
            class_name = sample_dict["class_name"]
            file_name = sample_dict["file_name"]
            key = (class_name, file_name)
            if key in self.generated_map:
                partial_generated_indices.extend(self.generated_map[key])
        return partial_generated_indices

    def __len__(self):
        return self.num_original_samples + len(self.partial_generated_indices)

    def __getitem__(self, item):
        if item < self.num_original_samples:
            return self.original_dset[item]
        else:
            idx = self.partial_generated_indices[item - self.num_original_samples]
            return self.generated_dset[idx]


class PartialGeneratedDataset(Dataset):
    def __init__(self, original_dset, generated_dset: GeneratedDataset, cfg):
        self.original_dset = original_dset
        self.num_original_samples = len(original_dset)
        self.generated_dset = generated_dset
        self.generated_root = os.path.join(cfg.generation.root_dir, cfg.data.dataset, cfg.generation.exp)

        self.generated_map = self._build_generated_map()
        self.partial_generated_indices = self._concat_datasets()

    def _build_generated_map(self):
        generated_map = {}
        print("TR: Building partial generated indices...")
        for idx, (path, _) in enumerate(self.generated_dset.dset.imgs):
            rel_path = os.path.relpath(path, self.generated_root)
            class_name, filename = os.path.split(rel_path)
            if "_augment_" not in filename:
                continue
            base_name = filename[:filename.rindex("_augment_")]
            key = (class_name, base_name)
            generated_map.setdefault(key, []).append(idx)
        return generated_map

    def _concat_datasets(self):
        partial_generated_indices = []
        for idx in tqdm(range(self.num_original_samples)):
            sample_dict = self.original_dset[idx]
            class_name = sample_dict["class_name"]
            file_name = sample_dict["file_name"]
            key = (class_name, file_name)
            if key in self.generated_map:
                partial_generated_indices.extend(self.generated_map[key])
        return partial_generated_indices

    def __len__(self):
        return len(self.partial_generated_indices)

    def __getitem__(self, item):
        idx = self.partial_generated_indices[item - self.num_original_samples]
        return self.generated_dset[idx]


class LocalRandomReplaceDataset(Dataset):
    def __init__(self, original_dset, generated_dset: GeneratedDataset, cfg):
        self.original_dset = original_dset
        self.generated_dset = generated_dset
        self.generated_root = os.path.join(cfg.generation.root_dir, cfg.data.dataset, cfg.generation.exp)
        p = OmegaConf.select(cfg, "generation.replace_prob")
        if p is None:
            self.p = 0.5
        else:
            self.p = p

        self.generated_map = self._build_generated_map()

    def _build_generated_map(self):
        generated_map = {}
        num_total_generated = 0
        print("LRR: Building partial generated indices...")
        for idx, (path, _) in enumerate(self.generated_dset.dset.imgs):
            rel_path = os.path.relpath(path, self.generated_root)
            class_name, filename = os.path.split(rel_path)
            if "_augment_" not in filename:
                continue
            base_name = filename[:filename.rindex("_augment_")]
            key = (class_name, base_name)
            generated_map.setdefault(key, []).append(idx)
            num_total_generated += 1
        print(f"{num_total_generated} generated samples loaded!")
        return generated_map

    def __len__(self):
        return len(self.original_dset)

    def __getitem__(self, idx):
        sample_dict = self.original_dset[idx]
        class_name = sample_dict["class_name"]
        file_name = sample_dict["file_name"]
        key = (class_name, file_name)

        use_augmented = random.random() < self.p
        if use_augmented and key in self.generated_map:
            idx = random.choice(self.generated_map[key])
            sample_dict = self.generated_dset[idx]

        return sample_dict


class GlobalRandomReplaceDataset(Dataset):
    def __init__(self, original_dset, generated_dset: GeneratedDataset, cfg):
        self.original_dset = original_dset
        self.generated_dset = generated_dset
        self.generated_root = os.path.join(cfg.generation.root_dir, cfg.data.dataset, cfg.generation.exp)
        p = OmegaConf.select(cfg, "generation.replace_prob")
        if p is None:
            self.p = 0.5
        else:
            self.p = p
        print("GRR: GlobalRandomReplaceDataset Initialized!")

    def __len__(self):
        return len(self.original_dset)

    def __getitem__(self, idx):
        sample_dict = self.original_dset[idx]

        use_augmented = random.random() < self.p
        if use_augmented:
            augment_idx = random.choice(range(len(self.generated_dset)))
            sample_dict = self.generated_dset[augment_idx]

        return sample_dict

def extract_tgt_class(filename, class_names):
    import re
    # 排序：更长的类名优先匹配，避免截断
    sorted_class_names = sorted(class_names, key=lambda x: -len(x))

    # 记录所有匹配到的 class_name 和其位置
    matches = []
    for class_name in sorted_class_names:
        pattern = re.escape(class_name)
        for match in re.finditer(pattern, filename):
            matches.append((match.start(), class_name))

    if not matches:
        return None

    # 返回靠近末尾（start位置最大）的那个 class
    return max(matches, key=lambda x: x[0])[1]


class GlobalRandomMixDataset(Dataset):
    def __init__(self, original_dset, generated_dset: GeneratedDataset, cfg):
        self.original_dset = original_dset
        self.generated_dset = generated_dset
        self.generated_root = os.path.join(cfg.generation.root_dir, cfg.data.dataset, cfg.generation.exp)
        self.p = cfg.generation.replace_prob
        self.strength = cfg.generation.strength
        self.gamma = cfg.generation.gamma
        if isinstance(self.original_dset.class_names, dict):
            self.class_names = self.original_dset.class_names.values()
        elif isinstance(self.original_dset.class_names, list):
            self.class_names = self.original_dset.class_names
        else:
            raise ValueError("Unknown class_names type")

    def __len__(self):
        return len(self.original_dset)

    def __getitem__(self, idx):
        sample_dict = self.original_dset[idx]
        use_augmented = random.random() < self.p
        src_y = sample_dict["y"]
        src_class = sample_dict["class_name"]
        y_onehot = F.one_hot(torch.tensor(src_y), num_classes=len(self.class_names)).float()
        if use_augmented:
            augment_idx = random.choice(range(len(self.generated_dset)))
            augment_dict = self.generated_dset[augment_idx]
            tgt_class = extract_tgt_class(augment_dict["file_name"], self.class_names)
            tgt_y = self.original_dset.class_to_idx[tgt_class]
            tgt_weight = math.pow(self.strength, self.gamma)
            y_onehot[src_y] = 1 - tgt_weight
            y_onehot[tgt_y] = tgt_weight
            sample_dict["x"] = augment_dict["x"]
            sample_dict["class_name"] = f"{(1-tgt_weight):.2f}{src_class}+{tgt_weight:.2f}{tgt_class}"
            sample_dict["file_name"] = augment_dict["file_name"]
        sample_dict["y"] = y_onehot
        return sample_dict


class DiffMixConcatDataset(Dataset):
    def __init__(self, original_dset, generated_dset: GeneratedDataset, cfg):
        self.original_dset = original_dset
        self.generated_dset = generated_dset
        self.generated_root = os.path.join(cfg.generation.root_dir, cfg.data.dataset, cfg.generation.exp)
        self.strength = cfg.generation.strength
        self.gamma = cfg.generation.gamma
        if isinstance(self.original_dset.class_names, dict):
            self.class_names = self.original_dset.class_names.values()
        elif isinstance(self.original_dset.class_names, list):
            self.class_names = self.original_dset.class_names
        else:
            raise ValueError("Unknown class_names type")

    def __len__(self):
        return len(self.original_dset) + len(self.generated_dset)

    def __getitem__(self, idx):
        if idx < len(self.original_dset):
            sample_dict = self.original_dset[idx]
            src_y = sample_dict["y"]
            y_onehot = F.one_hot(torch.tensor(src_y), num_classes=len(self.class_names)).float()
        else:
            idx -= len(self.original_dset)
            sample_dict = self.generated_dset[idx]
            src_y = sample_dict["y"]
            src_class = sample_dict["class_name"]
            y_onehot = F.one_hot(torch.tensor(src_y), num_classes=len(self.class_names)).float()
            tgt_class = extract_tgt_class(sample_dict["file_name"], self.class_names)
            tgt_y = self.original_dset.class_to_idx[tgt_class]
            tgt_weight = math.pow(self.strength, self.gamma)
            y_onehot[src_y] = 1 - tgt_weight
            y_onehot[tgt_y] = tgt_weight
            sample_dict["x"] = sample_dict["x"]
            sample_dict["class_name"] = f"{(1 - tgt_weight):.2f}{src_class}+{tgt_weight:.2f}{tgt_class}"
            sample_dict["file_name"] = sample_dict["file_name"]
        sample_dict["y"] = y_onehot
        return sample_dict


def get_transformation(cfg, split='train'):
    if cfg.data.resolution == 224:
        resize_size = 256
        crop_size = 224
    elif cfg.data.resolution == 384:
        resize_size = 440
        crop_size = 384
    else:
        raise NotImplementedError

    if cfg.data.dataset.startswith("cifar"):
        DATASET_MEAN = (0.5071, 0.4867, 0.4408)
        DATASET_STD = (0.2675, 0.2565, 0.2761)
    else:
        DATASET_MEAN = (0.485, 0.456, 0.406)
        DATASET_STD = (0.229, 0.224, 0.225)

    if split == "train":
        # for training classifier
        transform = transforms.Compose([
            transforms.Resize((resize_size, resize_size)),
            transforms.RandomCrop(crop_size, padding=8),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(DATASET_MEAN, DATASET_STD),
        ])
    else:
        # for testing classifier
        transform = transforms.Compose([
            transforms.Resize((resize_size, resize_size)),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(DATASET_MEAN, DATASET_STD),
        ])

    return transform


def get_dataset(cfg, split="train", return_pure_image=False):
    if not return_pure_image:
        transform = get_transformation(cfg, split)
    else:
        transform = None
    if cfg.data.dataset == "Aircraft":
        dset = AircraftDataset(root=cfg.data.root, split=split, transform=transform)
    elif cfg.data.dataset == "Birds":
        dset = CUBDataset(root=cfg.data.root, split=split, transform=transform)
    elif cfg.data.dataset == "Flowers":
        dset = Flowers102Dataset(root=cfg.data.root, split=split, transform=transform)
    elif cfg.data.dataset == "iNatAves":
        dset = get_inat_dataset(dataset_name="semi_aves", root=cfg.data.root, split=split, transform=transform)
    elif cfg.data.dataset == 'iNatFungi':
        dset = get_inat_dataset(dataset_name="semi_fungi", root=cfg.data.root, split=split, transform=transform)
    elif cfg.data.dataset == 'CIFAR100':
        dset = CIFAR100Dataset(root=cfg.data.root, split=split, transform=transform)
    elif cfg.data.dataset == 'Caltech101':
        dset = Caltech101Dataset(root=cfg.data.root, split=split, transform=transform)
    elif cfg.data.dataset == 'ImageNet100':
        dset = ImageNet100Dataset(root=cfg.data.root, split=split, transform=transform)
    elif  cfg.data.dataset.startswith('Colon') or cfg.data.dataset.startswith('Blood') or cfg.data.dataset.startswith('Skin'):
        dset = MedicalDataset(cfg=cfg, split=split, transform=transform)
    else:
        raise NotImplementedError

    if split == "train" and cfg.data.examples_per_class > 0:
        dset = FewShotDataset(dset, seed=cfg.seed, examples_per_class=cfg.data.examples_per_class)

    return dset, transform


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test datasets")
    parser.add_argument("--dataset", type=str, default="Dogs")
    args = parser.parse_args()
    cfg = OmegaConf.create(
        {
            "data": {
                "dataset": args.dataset,
                "root": "/root/autodl-tmp/lzk/datasets",
                "resolution": 224,
                "examples_per_class": 5,
            },
            "task": "classification",
            "seed": 0,
        }
    )
    dset = get_dataset(cfg)
    print(len(dset))
    sample = dset[0]
    print(f"x.shape: {sample['x'].shape}, y: {sample['y']}, class_name: {sample['class_name']}")
