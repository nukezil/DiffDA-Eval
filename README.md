# DiffDA-Eval

**Official Codebase for our IJCV Paper**  
📄 _Diffusion-Based Data Augmentation for Image Recognition: A Systematic Analysis and Evaluation_
> **Abstract：** Diffusion-based data augmentation (DiffDA) has emerged as a promising approach to improving classification performance under data scarcity. However, existing works vary significantly in task configurations, model choices, and experimental pipelines, making it difficult to fairly compare methods or assess their effectiveness across different scenarios. Moreover, there remains a lack of systematic understanding of the full DiffDA workflow. In this work, we introduce UniDiffDA, a unified analytical framework that decomposes DiffDA methods into three core components: model fine-tuning, sample generation, and sample utilization. This perspective enables us to identify key differences among existing methods and clarify the overall design space. Building on this framework, we develop a comprehensive and fair evaluation protocol, benchmarking representative DiffDA methods across diverse low-data classification tasks. Extensive experiments reveal the relative strengths and limitations of different DiffDA strategies and offer practical insights into method design and deployment. All methods are re-implemented within a unified codebase, with full release of code and configurations to ensure reproducibility and to facilitate future research.

> **TL;DR:** We analyze diffusion-based data augmentation (DiffDA) methods under a unified framework and evaluate them using a unified codebase.

---
## 📚 Supported DiffDA Methods

| Method                                                  | Venue          | Our Implementation                                             |
|---------------------------------------------------------|----------------|----------------------------------------------------------------|
| [Real Guidance](https://arxiv.org/abs/2210.07574)       | _ICLR 2023_    | [augmentation/real_guidance.py](augmentation/real_guidance.py) |
| [GIF](https://arxiv.org/abs/2211.13976)                 | _NeurIPS 2023_ | [augmentation/gif.py](augmentation/gif.py)                     |
| [DiffuseMix](https://arxiv.org/abs/2405.14881)          | _CVPR 2024_    | [augmentation/diffuse_mix.py](augmentation/diffuse_mix.py)     |
| [DA-Fusion](https://openreview.net/forum?id=ZWzUA9zeAg) | _ICLR 2024_    | [augmentation/da_fusion.py](augmentation/da_fusion.py)         |
| [Diff-Aug](https://arxiv.org/abs/2403.19600)            | _CVPR 2024_    | [augmentation/diff_aug.py](augmentation/diff_aug.py)           |
| [Diff-Mix](https://arxiv.org/abs/2403.19600)            | _CVPR 2024_    | [augmentation/diff_mix.py](augmentation/diff_mix.py)           |
| [Diff-II](https://arxiv.org/abs/2408.16266)             | _CVPR 2025_    | [augmentation/diff_ii.py](augmentation/diff_ii.py)             |
               
**We gratefully acknowledge the authors of the above methods for making their code publicly available.**


## 🛠️ Environment Requirements

To ensure compatibility, please use the following package versions:

- `torch >= 2.3.0`
- `diffusers >= 0.33.1`
- `transformers >= 4.46.0`
- `peft >= 0.15.2`

## 📦 Code, Data, and Model Preparation

### Environment Configuration

This project uses environment variables for path configuration. Set the following environment variable before running any scripts:

```bash
export DIFFDA_ROOT=/path/to/your/root/directory
```


The root directory should contain:
- `DiffDA-Eval/` (this project)
- `datasets/` (dataset directory)
- `ckpts/` (model checkpoint directory)

Example directory structure:
```
/path/to/your/root/directory/
├─ DiffDA-Eval/
├─ datasets/
├─ ckpts/
```

### 1. Download Pre-split Datasets
Download the preprocessed dataset splits from [Hugging Face](https://huggingface.co/datasets/nukezil/DiffDA-Eval) and extract them to `$DIFFDA_ROOT/datasets`.

Directory structure after extraction:
```
$DIFFDA_ROOT/datasets/
├─ Blood/
├─ CIFAR100/
├─ CUB_200_2011/
...
```

### 2. Download Pretrained Diffusion Models

- **Stable Diffusion v1.5**  
  Download from [Hugging Face](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5)  
  Place it in: `$DIFFDA_ROOT/ckpts/stable-diffusion-v1-5/`

- **InstructPix2Pix**  
  Download from [Hugging Face](https://huggingface.co/timbrooks/instruct-pix2pix)  
  Place it in: `$DIFFDA_ROOT/ckpts/instruct-pix2pix/`

### 3. Configuration Files

All configuration files in the `configs/` directory should use paths relative to the environment variables. For example:

```yaml
data:
  root: ${DIFFDA_ROOT}/datasets  # Will be resolved at runtime

generation:
  sd_model: ${DIFFDA_ROOT}/ckpts/stable-diffusion-v1-5
  root_dir: ${DIFFDA_ROOT}/DiffDA-Eval/generated_images
```

The configuration system automatically resolves `${DIFFDA_ROOT}` placeholders to the actual value of the environment variable. This makes the configuration files portable across different systems without any modifications.


## 🚀 Running the Pipeline

The full DiffDA workflow consists of three stages:

### Stage 1: Model Fine-tuning

We fine-tune the base diffusion model using **Text Inversion** and **DreamBooth-LoRA**:

```bash
# Run Text Inversion
bash text_inversion.sh

# Run DreamBooth-LoRA
bash dreambooth.sh
```

### Stage 2: Sample Generation

Generate synthetic samples by loading the appropriate config file.  
For example, to generate samples on the **CUB (Birds)** dataset (shot-5 task) using the **Diff-Mix** method:

```bash
CUDA_VISIBLE_DEVICES=0 python generate_samples.py --config configs/generation/diff_mix/GEN_diff_mix_cub_shot5.yaml
```

### Stage 3: Classifier Training

Use both real and synthetic samples to train a classifier.  
For example, to train on **CUB shot-5** using samples generated by **Diff-Mix**:

```bash
CUDA_VISIBLE_DEVICES=0 python train_classifier.py --config configs/classification/diff_mix/CLS_diff_mix_cub_shot5.yaml
```

Additional configs for other methods and datasets are available under the `configs/` directory.