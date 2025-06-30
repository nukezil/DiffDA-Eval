import os
from datetime import datetime
import omegaconf
from omegaconf import OmegaConf
import argparse
from tqdm import tqdm
import wandb

import torch.multiprocessing as mp
from torch.utils.data import Subset

from data.wrapper import get_dataset
from utils.misc import *
from augmentation import RealGuidanceGenerator, GIFGenerator, DiffuseMixGenerator, DAFusionGenerator,\
    DiffAugGenerator, DiffMixGenerator, DiffIIGenerator


import warnings
from diffusers.utils import logging as diffusers_logging
warnings.filterwarnings("ignore")
diffusers_logging.set_verbosity_error()


def process_subset(rank, train_dset, cfg):
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    print(f"Process {rank} using device {device} to augment {len(train_dset)} samples")

    run = wandb.init(
        project="DiffDA-Gen",
        group=cfg.exp_name,
        name=f"{cfg.exp_name}_{rank}",
        config=OmegaConf.to_container(cfg, resolve=True)
    )
    wandb.config.update({"gpu": rank})

    if cfg.generation.method == "GIF":
        augmentor = GIFGenerator(cfg, train_dset)
    elif cfg.generation.method == "RealGuidance":
        augmentor = RealGuidanceGenerator(cfg, train_dset)
    elif cfg.generation.method == "DiffuseMix":
        augmentor = DiffuseMixGenerator(cfg)
    elif cfg.generation.method == "DAFusion":
        augmentor = DAFusionGenerator(cfg, train_dset)
    elif cfg.generation.method == "DiffAug":
        augmentor = DiffAugGenerator(cfg, train_dset)
    elif cfg.generation.method == "DiffMix":
        augmentor = DiffMixGenerator(cfg, train_dset)
    elif cfg.generation.method == "DiffII":
        augmentor = DiffIIGenerator(cfg, train_dset)
    else:
        raise NotImplementedError

    gen_task_path = os.path.join(cfg.generation.root_dir, cfg.data.dataset, cfg.exp_name)  # path to store generated images
    link_path = os.path.join(cfg.generation.root_dir, cfg.data.dataset, cfg.exp_name_no_time)
    os.makedirs(gen_task_path, exist_ok=True)
    os.symlink(gen_task_path, link_path, target_is_directory=True)
    gen_class_paths = {}

    for sample_idx in tqdm(range(cfg.start_idx, len(train_dset)), disable=(cfg.num_gpus>1)):
        sample_dict = train_dset[sample_idx]
        if cfg.generation.method == "DiffMix":
            augment_images, grid_image, target_classes = augmentor.augment_one_sample(sample_dict)
        else:
            augment_images, grid_image = augmentor.augment_one_sample(sample_dict)
            target_classes = None

        class_name = sample_dict["class_name"]
        if class_name not in gen_class_paths:
            gen_class_path = os.path.join(gen_task_path, sample_dict["class_name"])
            gen_class_paths[class_name] = gen_class_path
            os.makedirs(gen_class_path, exist_ok=True)
            run.log({
                f"GridExample/{class_name}": wandb.Image(grid_image, caption=f"{class_name}")
            })
        gen_class_path = gen_class_paths[class_name]

        for aug_idx, aug_image in enumerate(augment_images):
            if target_classes is not None:
                suffix = f"{target_classes[aug_idx]}_{aug_idx}"
            else:
                suffix = f"augment_{aug_idx}"
            aug_image.save(os.path.join(gen_class_path, f"{sample_dict['file_name']}_{suffix}.png"))

        if ((sample_idx + 1) % 50  == 0) and (cfg.num_gpus > 1):
            print(f"Process {rank} has augmented {sample_idx + 1} samples")

    run.finish()

def multi_gpus_main(args):
    cfg = OmegaConf.load(args.config)
    cfg.num_gpus = args.num_gpus
    if args.seed is not None:
        cfg.seed = args.seed
    set_all_seeds(cfg.seed)
    if args.data_root is not None:
        cfg.data.root = args.data_root

    # Override generation parameters
    if args.strength is not None:
        cfg.generation.strength = args.strength
        cfg.exp_name += f"_strength{cfg.generation.strength}"

    # Override the path of finetuned weights
    # For Textural Inversion
    if OmegaConf.select(cfg, "generation.ti_embeds") is not None:
        if isinstance(cfg.generation.ti_embeds, str):
            pass
        elif isinstance(cfg.generation.ti_embeds, omegaconf.dictconfig.DictConfig):
            cfg.generation.ti_embeds = cfg.generation.ti_embeds[cfg.seed]
        else:
            raise ValueError(f"Unknown generation.ti_embeds: {cfg.generation.ti_embeds}")
    # For DreamBooth
    if OmegaConf.select(cfg, "generation.db_lora_weights") is not None:
        if isinstance(cfg.generation.db_lora_weights, str):
            pass
        elif isinstance(cfg.generation.db_lora_weights, omegaconf.dictconfig.DictConfig):
            cfg.generation.db_lora_weights = cfg.generation.db_lora_weights[cfg.seed]
        else:
            raise ValueError(f"Unknown generation.db_lora_weights: {cfg.generation.db_lora_weights}")

    timestamp = datetime.now().strftime('%m%d%H%M')
    if args.resume_task is not None:
        cfg.exp_name = args.resume_task
        print(f"Resuming {args.resume_task}, from {args.resume_from}-th sample")
    else:
        cfg.exp_name_no_time = cfg.exp_name + f"_seed{cfg.seed}"
        cfg.exp_name = cfg.exp_name_no_time + f"_{timestamp}"

    if args.resume_from is not None:
        cfg.start_idx = args.resume_from
    else:
        cfg.start_idx = 0

    cfg.exp_dir = os.path.join(args.output_dir, cfg.exp_name)
    if not os.path.exists(cfg.exp_dir):
        os.makedirs(cfg.exp_dir)
    print(f"Configuration Loaded! Exp {cfg.exp_name} starts!")

    train_dset, _ = get_dataset(cfg, split="train", return_pure_image=True)
    print(f"Datasets Initialized! Augmenting {len(train_dset)} samples by {cfg.generation.augment_ratio} times...")

    data_len = len(train_dset)
    indices = list(range(data_len))
    chunk_size = (data_len + args.num_gpus - 1) // args.num_gpus  # 向上取整，防止最后一份丢图
    subsets = [Subset(train_dset, indices[i * chunk_size: (i + 1) * chunk_size]) for i in range(args.num_gpus)]
    for subset in subsets:
        subset.class_names = train_dset.class_names

    mp.set_start_method("spawn", force=True)

    processes = []
    for rank in range(args.num_gpus):
        p = mp.Process(target=process_subset, args=(rank, subsets[rank], cfg))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/base.yaml")
    parser.add_argument("--data_root", type=str, default="/data/lzk/datasets")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="./out")  # for log files instead of generated images
    parser.add_argument("--resume_task", type=str, default=None)
    parser.add_argument("--resume_from", type=int, default=None)
    parser.add_argument("--num_gpus", type=int, default=1)
    # Override generation parameters
    parser.add_argument("--strength", type=float, default=None)
    args = parser.parse_args()

    multi_gpus_main(args)