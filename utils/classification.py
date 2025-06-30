import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast
from torchvision.models import resnet50, vit_b_16, mobilenet_v3_large
from torch.utils.data import DataLoader
from data.wrapper import GeneratedDataset, PartialConcatDataset, PartialGeneratedDataset, \
    LocalRandomReplaceDataset, GlobalRandomReplaceDataset, GlobalRandomMixDataset

def perform_augmentation(cfg, original_dset, transform_train):
    from omegaconf import OmegaConf
    aug_method = OmegaConf.select(cfg, "generation.method")
    if aug_method is None:
        augmented_dataset = original_dset
    else:
        generated_dset = GeneratedDataset(cfg, transform_train, original_dset.class_to_idx)
        utilize_way = OmegaConf.select(cfg, "generation.utilize_way")
        if utilize_way is not None:
            # manually decide the utilize way for analysis
            if utilize_way == "TC":
                # Total Concat
                augmented_dataset = PartialConcatDataset(original_dset, generated_dset, cfg)
            elif utilize_way == "TR":
                # Total Replace
                augmented_dataset = PartialGeneratedDataset(original_dset, generated_dset, cfg)
            elif utilize_way == "LRR":
                # Local Random Replace
                augmented_dataset = LocalRandomReplaceDataset(original_dset, generated_dset, cfg)
            elif utilize_way == "GRR":
                # Global Random Replace
                augmented_dataset = GlobalRandomReplaceDataset(original_dset, generated_dset, cfg)
            else:
                raise NotImplementedError
        else:
            # decided by the generation method
            if aug_method in ["RealGuidance", "GIF"]:
                # concat
                if cfg.data.examples_per_class == -1 and not (cfg.data.dataset.startswith("Blood") or cfg.data.dataset.startswith("Skin")):
                    # just concat original_dset with full generated_dset
                    augmented_dataset = torch.utils.data.ConcatDataset([original_dset, generated_dset])
                else:
                    augmented_dataset = PartialConcatDataset(original_dset, generated_dset, cfg)
            elif aug_method in ["DAFusion", "DiffII"]:
                # randomly replace (local)
                augmented_dataset = LocalRandomReplaceDataset(original_dset, generated_dset, cfg)
            elif aug_method in ["DiffAug"]:
                # randomly replace (global)
                augmented_dataset = GlobalRandomReplaceDataset(original_dset, generated_dset, cfg)
            elif aug_method in ["DiffMix"]:
                # randomly mixup and replace (global)
                augmented_dataset = GlobalRandomMixDataset(original_dset, generated_dset, cfg)
            elif aug_method in ["DiffuseMix"]:
                # directly use generated dataset
                if cfg.data.examples_per_class == -1:
                    augmented_dataset = generated_dset
                else:
                    augmented_dataset = PartialGeneratedDataset(original_dset, generated_dset, cfg)
            else:
                raise NotImplementedError

    return augmented_dataset


def get_model(cfg, device="cuda"):
    if cfg.model.weights == 'None' or cfg.model.weights is None or cfg.model.weights == "scratch":
        weights = None
    else:
        weights = cfg.model.weights

    if cfg.model.arch == "resnet50":
        model = resnet50(weights=weights)
        model.fc = nn.Linear(in_features=model.fc.in_features, out_features=cfg.data.num_classes)
    elif cfg.model.arch == "vit_b_16":
        model = vit_b_16(weights=weights)
        model.heads.head = nn.Linear(in_features=model.heads.head.in_features, out_features=cfg.data.num_classes)
    elif cfg.model.arch == "mobilenet_v3_large":
        model = mobilenet_v3_large(weights=weights)
        model.classifier[3] = nn.Linear(in_features=model.classifier[3].in_features, out_features=cfg.data.num_classes)
    else:
        raise NotImplementedError("Model architecture not supported!")

    for param in model.parameters():
        param.requires_grad = True

    return model.to(device)


def get_dataloaders(train_dset, test_dset, cfg):
    train_loader = DataLoader(
        train_dset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=8
    )
    test_loader = DataLoader(
        test_dset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=8
    )
    return train_loader, test_loader


def get_optimizer(model, cfg):
    if cfg.training.optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=cfg.training.lr, momentum=cfg.training.momentum, weight_decay=cfg.training.weight_decay)
    elif cfg.training.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)
    else:
        raise NotImplementedError("Optimizer not supported!")
    return optimizer


def train_one_epoch(model, train_loader, optimizer, scaler, criterion, cutmix_or_mixup=None, device="cuda"):
    model.train()
    train_loss = 0.0
    correct = total = 0
    for batch_idx, batch in enumerate(train_loader):
        optimizer.zero_grad()
        imgs, targets = batch["x"], batch["y"]
        imgs, targets = imgs.to(device), targets.to(device)
        if cutmix_or_mixup is not None and random.random() < 0.5:
            imgs, targets = cutmix_or_mixup(imgs, targets)
        if scaler is not None:
            with autocast("cuda"):
                outputs = model(imgs)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(imgs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        if targets.ndim == 2:  # 判断是否为 soft label
            _, targets_hard = torch.max(targets, dim=1)
        else:
            targets_hard = targets  # 直接使用整数类别

        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == targets_hard).sum().item()
        train_loss += loss.item()

    train_loss /= len(train_loader)
    train_acc = 100 * correct / total

    return train_loss, train_acc


def test(model, test_loader, device="cuda"):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_loader:
            imgs, targets = batch["x"], batch["y"]
            imgs, targets = imgs.to(device), targets.to(device)
            outputs = model(imgs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    accuracy = 100 * correct / total
    return accuracy


class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            if target.dim() == 1:
                true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
            else:
                true_dist = true_dist + target * self.confidence
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))