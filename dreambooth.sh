datasets=("CIFAR100" "Caltech101" "ImageNet100")

for dataset in "${datasets[@]}"; do
    CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes=2 finetuning/dreambooth.py \
    --pretrained_model_name_or_path="/path/to/your/ckpts/stable-diffusion-v1-5" \
    --data_root_dir="/path/to/your/datasets" \
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
    --ti_embed_path="/path/to/your/workspace/DataGen/finetuned_generators/text_inversion/${dataset}_shot-1_seed0" \
done

datasets=("Birds" "Aircraft")
seeds=(0 1 2)
shots=(1 5 10)


for dataset in "${datasets[@]}"; do
    for shot in "${shots[@]}"; do
        for seed in "${seeds[@]}"; do
            CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes=2 finetuning/dreambooth.py \
            --pretrained_model_name_or_path="/path/to/your/ckpts/stable-diffusion-v1-5" \
            --data_root_dir="/path/to/your/datasets" \
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
            --ti_embed_path="/path/to/your/workspace/DataGen/finetuned_generators/text_inversion/${dataset}-${shot}_shot-1_seed${seed}"
        done
    done
done

    