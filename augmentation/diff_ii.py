import math
from einops import repeat
import random
import os
from collections import defaultdict
import pickle
from tqdm import tqdm
import torch
import torchvision.transforms as transforms
import PIL
from diffusers import StableDiffusionPipeline, DDIMInverseScheduler, AutoencoderKL, DDIMScheduler


transform_img2img = transforms.Compose([transforms.Resize((512,512), interpolation=PIL.Image.BICUBIC),
                                        transforms.ToTensor()])

SUFFIX_DICT = {
    "Aircraft": [
        "parked on the runway.",
        "flying in the sky with the landing gear down.",
        "landing with another plane in the background.",
        "on the runway at an airport.",
        "on the tarmac with mountains in the background.",
        "flying in the air with the landing gear down.",
        "parked in a hangar with the door open.",
        "flying in the sky with palm trees in the background.",
        "flying in the sky against a blue background.",
        "lined up on the runway at the airport."
    ],
    "Birds": [
        "standing on a tree branch.",
        "flying around flowers.",
        "standing on a post by the water.",
        "flying over water.",
        "standing on the ground.",
        "swimming in the water.",
        "sitting on a rock with a blue sky.",
        "perched on a branch in a tree.",
        "flying over water with wings spread.",
        "standing on a branch with tall grass in the background."
    ]
}

class DiffIIGenerator():
    """
        Implementation of [Diff-II] from
        nversion Circle Interpolation: Diffusion-based Image Augmentation for Data-scarce Classification (CVPR 2025)
    """
    def __init__(self, cfg, dset):
        self.cfg = cfg
        self.dset = dset
        self.class_to_indices = defaultdict(list)
        print("Preprocessing original dataset: mapping class to indices...")
        for idx in tqdm(range(len(dset))):
            sample_dict = dset[idx]
            self.class_to_indices[sample_dict["class_name"]].append(idx)
        self.suffix_list = SUFFIX_DICT.get(self.cfg.data.dataset, [""])
        self.super_class = getattr(dset, "super_class", "")
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            cfg.generation.sd_model,
            torch_dtype=torch.float16,
            revision="fp16",
            safety_checker=None,
            feature_extractor=None
        ).to("cuda")
        self.inversion_scheduler = DDIMInverseScheduler.from_pretrained(
            cfg.generation.sd_model, subfolder="scheduler"
        )
        self.ddim_scheduler = DDIMScheduler.from_pretrained(
            cfg.generation.sd_model, subfolder="scheduler"
        )
        self.pipeline.set_progress_bar_config(disable=True)

        self.load_ti_embeds()
        self.pipeline.load_lora_weights(cfg.generation.db_lora_weights, weight_name="pytorch_lora_weights.safetensors")
        print(f"DreamBooth Weights {self.cfg.generation.db_lora_weights} loaded!")

        self.get_inversion()


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

    def get_inversion(self):
        print("Getting or loading DDIM inversion latents...")
        pipe = self.pipeline
        vae = self.pipeline.vae
        pipe.scheduler = self.inversion_scheduler
        cache_dir = self.cfg.generation.inv_cache_dir
        inv_dir = os.path.join(cache_dir, self.cfg.data.dataset)
        os.makedirs(inv_dir, exist_ok=True)
        with torch.no_grad():
            for idx in tqdm(range(len(self.dset))):
                sample_dict = self.dset[idx]
                inv_name = f"{sample_dict['file_name']}.pkl"
                inv_path = os.path.join(inv_dir, inv_name)
                if os.path.exists(inv_path):
                    continue
                input_img = transform_img2img(sample_dict["x"]).unsqueeze(0).to(device="cuda", dtype=torch.float16)
                latents = image_to_latents(input_img, vae)
                text = f"a photo of a <{sample_dict['class_name']}> {self.super_class}"
                inv_latents, _ = pipe(prompt=text, negative_prompt="", guidance_scale=1.,
                                        width=input_img.shape[-1], height=input_img.shape[-2],
                                        output_type='latent', return_dict=False,
                                        num_inference_steps=200, latents=latents)
                with open(inv_path, "wb") as f:
                    pickle.dump(inv_latents, f)

    def augment_one_sample(self, sample_dict):
        augment_ratio = self.cfg.generation.augment_ratio
        inversion_step = self.cfg.generation.num_inference_steps
        split_ratio = self.cfg.generation.split_ratio
        condition_scale = 7.5

        pipe = self.pipeline
        vae = self.pipeline.vae
        pipe.scheduler = self.ddim_scheduler
        scheduler = pipe.scheduler
        tokenizer = pipe.tokenizer
        text_encoder = pipe.text_encoder
        unet = pipe.unet
        scheduler.set_timesteps(inversion_step)

        cache_dir = self.cfg.generation.inv_cache_dir
        inv_dir = os.path.join(cache_dir, self.cfg.data.dataset)
        z1_path = os.path.join(inv_dir, f"{sample_dict['file_name']}.pkl")
        z1 = pickle.load(open(z1_path, "rb"))

        samples = []
        prompts = []
        sampled_suffixes = random.sample(self.suffix_list, int(augment_ratio))
        for i in range(augment_ratio):
            text1 = f"a photo of a <{sample_dict['class_name']}>"
            text2 = f"a photo of a <{sample_dict['class_name']}> {sampled_suffixes[i]}"

            sample2 = self.dset[random.choice(self.class_to_indices[sample_dict["class_name"]])]
            z2_path = os.path.join(inv_dir, f"{sample2['file_name']}.pkl")
            z2 = pickle.load(open(z2_path, "rb"))
            ini_noise = rand_slerp(z1, z2)
            # print(f"Sample2: {sample2['file_name']}")
            with torch.no_grad():
                uncond_input = tokenizer([""], padding="max_length", max_length=tokenizer.model_max_length, truncation=True,
                                         return_tensors="pt")
                uncond_embeddings = text_encoder(uncond_input.input_ids.to(pipe.device))[0]
                text_input1 = tokenizer([text1], padding="max_length", max_length=tokenizer.model_max_length,
                                        truncation=True, return_tensors="pt")
                text_embeddings1 = text_encoder(text_input1.input_ids.to(pipe.device))[0]
                text_input2 = tokenizer([text2], padding="max_length", max_length=tokenizer.model_max_length,
                                        truncation=True, return_tensors="pt")
                text_embeddings2 = text_encoder(text_input2.input_ids.to(pipe.device))[0]
                text_embeddings_1 = torch.cat([uncond_embeddings, text_embeddings1])
                text_embeddings_2 = torch.cat([uncond_embeddings, text_embeddings2])

                latents = ini_noise * scheduler.init_noise_sigma
                for t in scheduler.timesteps:
                    if t < 1000 * (1 - split_ratio):
                        text_embeddings = text_embeddings_1
                    else:
                        text_embeddings = text_embeddings_2
                    latent_model_input = torch.cat([latents] * 2)
                    latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)
                    noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + condition_scale * (noise_pred_text - noise_pred_uncond)
                    latents = scheduler.step(noise_pred, t, latents).prev_sample
                image = latents_to_image(latents, vae)
            samples.append(image)
            prompts.append(text1)

        # return sample_dict["x"], samples, prompts
        return post_process(sample_dict["x"], samples, prompts)

def image_to_latents(x: torch.Tensor, vae: AutoencoderKL):
    x = 2. * x - 1.
    posterior = vae.encode(x).latent_dist
    latents = posterior.mean * 0.18215
    return latents


def latents_to_image(latents: torch.FloatTensor, vae: AutoencoderKL):
    latents = 1 / 0.18215 * latents
    image = vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1).squeeze()
    image = transforms.ToPILImage()(image)
    return image


def rand_slerp(z1, z2, eps=1e-6):
    cos_theta = torch.sum(z1 * z2) / (torch.norm(z1) * torch.norm(z2))
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
    theta = torch.acos(cos_theta)

    if torch.abs(theta) < eps:
        return z1.clone()

    T = 2 * math.pi / theta
    alpha = random.uniform(0, T)

    return (
            torch.sin((1 + alpha) * theta) / torch.sin(theta) * z1
            - torch.sin(alpha * theta) / torch.sin(theta) * z2
    )


def post_process(original_image, samples, prompts):
    grid_images = [original_image]
    grid_images.extend(samples)
    grid_prompts = [""]
    grid_prompts.extend(prompts)
    grid_to_show = create_image_grid_with_prompts(grid_images, grid_prompts)
    return samples, grid_to_show

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