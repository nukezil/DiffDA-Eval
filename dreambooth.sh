#!/bin/bash

# Check if DIFFDA_ROOT environment variable is set
if [ -z "$DIFFDA_ROOT" ]; then
    echo "Error: DIFFDA_ROOT environment variable is not set."
    echo "Please set it to the directory containing DiffDA-Eval, datasets, and ckpts."
    exit 1
fi

datasets=("CIFAR100" "Caltech101" "ImageNet100")

for dataset in "${datasets[@]}"; do
    CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes=2 finetuning/dreambooth.py \
    --pretrained_model_name_or_path="$DIFFDA_ROOT/ckpts/stable-diffusion-v1-5" \
    --data_root_dir="$DIFFDA_ROOT/datasets" \
    --dataset_name="$dataset" \
    --seed=0 \
    --examples_per_class=-1 \
    --resolution=512 \
    --train_batch_size=16 \
    --gradient_accumulation_steps=2 \
    --max_train_steps=1000 \
    --learning_rate=5e-6 \
    --scale_lr \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --mixed_precision="fp16" \
    --revision="fp16" \
    --gradient_checkpointing \
    --rank=8 \
    --ti_embed_path="$DIFFDA_ROOT/DiffDA-Eval/finetuned_generators/text_inversion/${dataset}_shot-1_seed0"
done

datasets=("Birds" "Aircraft")
seeds=(0 1 2)
shots=(1 5 10)


for dataset in "${datasets[@]}"; do
    for shot in "${shots[@]}"; do
        for seed in "${seeds[@]}"; do
            CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes=2 finetuning/dreambooth.py \
            --pretrained_model_name_or_path="$DIFFDA_ROOT/ckpts/stable-diffusion-v1-5" \
            --data_root_dir="$DIFFDA_ROOT/datasets" \
            --dataset_name="${dataset}-${shot}" \
            --seed="$seed" \
            --examples_per_class="-1" \
            --resolution=512 \
            --train_batch_size=16 \
            --gradient_accumulation_steps=2 \
            --max_train_steps=500 \
            --learning_rate=5e-6 \
            --scale_lr \
            --lr_scheduler="constant" \
            --lr_warmup_steps=0 \
            --mixed_precision="fp16" \
            --revision="fp16" \
            --gradient_checkpointing \
            --rank=8 \
            --ti_embed_path="$DIFFDA_ROOT/DiffDA-Eval/finetuned_generators/text_inversion/${dataset}-${shot}_shot-1_seed${seed}"
        done
    done
done