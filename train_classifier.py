from datetime import datetime
import omegaconf
from omegaconf import OmegaConf
import argparse
import wandb
import torch.nn as nn
from torch.amp import GradScaler

from data.wrapper import get_dataset
from utils.classification import perform_augmentation, get_dataloaders, get_model, get_optimizer, LabelSmoothingLoss, train_one_epoch, test
from utils.misc import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/base.yaml")
    parser.add_argument("--data_root", type=str, default="/data/lzk/datasets")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="./out")
    parser.add_argument("--generation_exp", type=str, default=None)
    parser.add_argument("--exp_name_suffix", type=str, default=None)
    parser.add_argument("--tradition_augment", type=str, default=None)
    parser.add_argument("--replace_prob", type=float, default=None)
    parser.add_argument("--utilize_way", type=str, default=None)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    cfg.seed = args.seed
    set_all_seeds(cfg.seed)
    if args.data_root is not None:
        cfg.data.root = args.data_root

    # Override the path of generated samples
    if OmegaConf.select(cfg, "generation") is not None:
        if isinstance(cfg.generation.exp, str):
            pass
        elif isinstance(cfg.generation.exp, omegaconf.dictconfig.DictConfig):
            cfg.generation.exp = cfg.generation.exp[cfg.seed]
        else:
            raise ValueError(f"Unknown generation.exp: {cfg.generation.exp}")

    if args.generation_exp is not None:
        cfg.generation.exp = args.generation_exp

    if args.exp_name_suffix is not None:
        cfg.exp_name += f"_{args.exp_name_suffix}"

    use_mixup = False
    use_cutmix = False
    if args.tradition_augment is not None:
        cfg.exp_name += f"_{args.tradition_augment}"
        if args.tradition_augment == "MixUp":
            use_mixup = True
        elif args.tradition_augment == "CutMix":
            use_cutmix = True
        else:
            raise ValueError(f"Unknown args.tradition_augment: {args.tradition_augment}")

    if args.replace_prob is not None:
        cfg.generation.replace_prob = args.replace_prob
        cfg.exp_name += f"_p{args.replace_prob}"

    if args.utilize_way is not None:
        cfg.generation.utilize_way = args.utilize_way
        cfg.exp_name += f"_{args.utilize_way}"

    cfg.exp_name += f"_seed{cfg.seed}_{datetime.now().strftime('%m%d%H%M')}"
    cfg.exp_dir = os.path.join(args.output_dir, cfg.exp_name)
    if not os.path.exists(cfg.exp_dir):
        os.makedirs(cfg.exp_dir)
    print(f"Configuration Loaded! Exp {cfg.exp_name} starts!")

    os.environ["WANDB_MODE"] = "offline"
    run = wandb.init(
        project="GenDA-Bench",
        name=cfg.exp_name,
        config=OmegaConf.to_container(cfg, resolve=True)
    )

    original_dset, transform_train = get_dataset(cfg, split="train")
    train_dset = perform_augmentation(cfg, original_dset, transform_train)
    test_dset, _ = get_dataset(cfg, split="test")

    cutmix_or_mixup = None
    if use_mixup:
        from torchvision.transforms.v2 import MixUp
        cutmix_or_mixup = MixUp(alpha=1.0, num_classes=cfg.data.num_classes)
        print("Using MixUp augmentation!")
    if  use_cutmix:
        from torchvision.transforms.v2 import CutMix
        cutmix_or_mixup = CutMix(alpha=1.0, num_classes=cfg.data.num_classes)
        print("Using CutMix augmentation!")

    print(f"Datasets Initialized! Train with {len(train_dset)}; Test on {len(test_dset)} samples.")
    train_loader, test_loader = get_dataloaders(train_dset, test_dset, cfg)
    cfg.data.num_classes = original_dset.get_num_classes()

    model = get_model(cfg)
    optimizer = get_optimizer(model, cfg)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.training.num_epochs)
    scaler = GradScaler('cuda')
    # scaler = None

    if cfg.training.criterion == "LabelSmoothing":
        criterion = LabelSmoothingLoss(cfg.data.num_classes, 0.1)
    else:
        criterion = nn.CrossEntropyLoss()

    best_test_acc = 0
    for epoch in range(cfg.training.num_epochs):
        cur_lr = scheduler.get_last_lr()[0]
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, scaler, criterion,
                                                cutmix_or_mixup=cutmix_or_mixup, device="cuda")
        scheduler.step()
        test_acc = test(model, test_loader)
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            # torch.save(model.state_dict(), os.path.join(cfg.exp_dir, "best_model.pth"))
        run.log(
            {
                "train/loss": train_loss,
                "train/acc": train_acc,
                "test/acc": test_acc,
                "test/best_acc": best_test_acc,
                "lr": cur_lr,
            },
        )
        if (epoch + 1) % 4 == 0:
            print(f"[Train Epoch {epoch + 1}] Current Test Accuracy: {test_acc:.2f}%, Best Test Accuracy: {best_test_acc:.2f}%")
    run.finish()

if __name__ == "__main__":
    main()