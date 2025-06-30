from einops import repeat
import random
import torch
import torchvision.transforms as transforms
import PIL
from diffusers import StableDiffusionImg2ImgPipeline, DPMSolverMultistepScheduler, DDIMScheduler


transform_img2img = transforms.Compose([transforms.Resize((512,512), interpolation=PIL.Image.BICUBIC),
                                        transforms.ToTensor()])

class DiffMixGenerator():
    """
        Implementation of [Diff-Mix] from
        Enhance Image Classification via Inter-Class Image Mixup with Diffusion Model (CVPR 2024)
    """
    def __init__(self, cfg, dset):
        self.cfg = cfg
        self.dset = dset
        self.super_class = getattr(dset, "super_class", None)
        self.pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
            cfg.generation.sd_model,
            torch_dtype=torch.float16,
            revision="fp16",
            safety_checker=None,
            feature_extractor=None
        ).to("cuda")
        self.pipeline.set_progress_bar_config(disable=True)

        self.load_ti_embeds()
        self.pipeline.load_lora_weights(cfg.generation.db_lora_weights, weight_name="pytorch_lora_weights.safetensors")
        print(f"DreamBooth Weights {self.cfg.generation.db_lora_weights} loaded!")

        if cfg.generation.scheduler == "DPMSolver++":
            self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(self.pipeline.scheduler.config)
        elif cfg.generation.scheduler == "DDIM":
            self.pipeline.scheduler = DDIMScheduler.from_config(self.pipeline.scheduler.config)
        else:
            raise ValueError("Scheduler not supported")

        self.generator = torch.Generator(device="cuda").manual_seed(cfg.seed)

    def load_ti_embeds(self):
        from safetensors.torch import load_file
        learned_embeds = load_file(f"{self.cfg.generation.ti_embeds}/learned_embeds.safetensors")["learned_embeds"]

        text_encoder = self.pipeline.text_encoder
        tokenizer = self.pipeline.tokenizer
        placeholder_tokens = [f"<{class_name}>" for class_name in self.dset.class_names]
        num_added_tokens = tokenizer.add_tokens(placeholder_tokens)

        if num_added_tokens != learned_embeds.shape[0]:
            raise ValueError(f"Loaded embeddings don't match this dataset!")

        text_encoder.resize_token_embeddings(len(tokenizer))
        placeholder_token_ids = tokenizer.convert_tokens_to_ids(placeholder_tokens)
        token_embeds = text_encoder.get_input_embeddings().weight.data
        for idx, token_id in enumerate(placeholder_token_ids):
            token_embeds[token_id] = learned_embeds[idx].clone()

        print(f"Textual Inversion Weights {self.cfg.generation.ti_embeds} loaded!")

    def get_prompt_template(self):
        if self.cfg.data.dataset.startswith("Blood"):
            return "a microscopic blood cell image of <{}>"
        elif self.cfg.data.dataset.startswith("Skin"):
            return "a dermatoscopic image of <{}>"
        else:
            return "a photo of a <{}>"

    def augment_one_sample(self, sample_dict):
        augment_ratio = self.cfg.generation.augment_ratio
        strength = self.cfg.generation.strength
        num_inference_steps = self.cfg.generation.num_inference_steps

        with torch.no_grad():
            sample_to_augment = transform_img2img(sample_dict["x"])
            init_images = repeat(sample_to_augment.unsqueeze(0), '1 ... -> b ...', b=augment_ratio)

            target_classes = []
            prompts_list = []
            for i in range(augment_ratio):
                target_class = random.choice(self.dset.class_names)
                target_classes.append(target_class)
                prompt_template = self.get_prompt_template()
                prompt = prompt_template.format(target_class)
                if self.super_class is not None:
                    prompt += f" {self.super_class}"
                prompts_list.append(prompt)

            prompts = prompts_list
            samples = self.pipeline(prompt=prompts,
                                           image=init_images,
                                           strength=strength,
                                           num_inference_steps=num_inference_steps,
                                           generator=self.generator
                                           )[0]

        return post_process(samples, target_classes, sample_to_augment, sample_dict["class_name"])

def post_process(samples, target_classes, sample_to_augment, source_class):
    grid_images = [transforms.ToPILImage()(sample_to_augment)]
    grid_images.extend(samples)
    grid_classes = [source_class]
    grid_classes.extend(target_classes)
    grid_to_show = create_image_grid_with_prompts(grid_images, grid_classes)
    return samples, grid_to_show, target_classes

def create_image_grid_with_prompts(images, prompts, image_size=(256, 256), padding=10, max_prompt_width=50):
    from PIL import Image, ImageDraw, ImageFont
    import textwrap
    images = [img.resize(image_size) for img in images]
    font = ImageFont.load_default()
    grid_width = image_size[0] * len(images)
    ascent, descent = font.getmetrics()
    line_height = ascent + descent
    max_lines = max(len(textwrap.wrap(p, width=max_prompt_width)) for p in prompts)
    text_block_height = line_height * max_lines + padding
    grid_height = image_size[1] + text_block_height
    grid_img = Image.new("RGB", (grid_width, grid_height), color="white")
    draw = ImageDraw.Draw(grid_img)
    for idx, (img, prompt) in enumerate(zip(images, prompts)):
        x = idx * image_size[0]
        y = 0
        grid_img.paste(img, (x, y))
        wrapped = textwrap.wrap(prompt, width=max_prompt_width)
        for i, line in enumerate(wrapped):
            text_x = x + 5
            text_y = image_size[1] + i * line_height
            draw.text((text_x, text_y), line, fill="black", font=font)
    return grid_img