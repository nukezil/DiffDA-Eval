import os
import random
import numpy as np
from einops import repeat
import torch
import torchvision.transforms as transforms
import PIL
from PIL import Image
from diffusers import StableDiffusionInstructPix2PixPipeline, DPMSolverMultistepScheduler, DDIMScheduler, EulerAncestralDiscreteScheduler

from utils.misc import create_image_grid

transform_img2img = transforms.Compose([transforms.Resize((256, 256), interpolation=PIL.Image.BICUBIC),
                                        transforms.ToTensor()])

class DiffuseMixGenerator():
    """
        Implementation of [Real-Guidance] from
        Is Synthetic Data from Generative Models Ready for Image Recognition (ICLR 2023)
    """
    def __init__(self, cfg):
        self.cfg = cfg
        self.pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            cfg.generation.sd_model,
            torch_dtype=torch.float16,
            revision="fp16",
            safety_checker=None,
            feature_extractor=None
        ).to("cuda")
        self.pipeline.set_progress_bar_config(disable=True)

        if cfg.generation.scheduler == "DPMSolver++":
            self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(self.pipeline.scheduler.config)
        elif cfg.generation.scheduler == "DDIM":
            self.pipeline.scheduler = DDIMScheduler.from_config(self.pipeline.scheduler.config)
        elif cfg.generation.scheduler == "EulerAncestral":
            self.pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(self.pipeline.scheduler.config)
        else:
            raise ValueError("Scheduler not supported")

        self.generator = torch.Generator(device="cuda").manual_seed(cfg.seed)
        self.fractal_images = Utils.load_fractal_images(cfg.generation.fractal_img_dir)

    def get_prompt_candidates(self):
        if self.cfg.data.dataset in ["Caltech101", "CIFAR100", "ImageNet100", "Aircraft", "Birds", "Dogs"]:
            style_list = ["autumn", "snowy", "watercolor art", "sunset", "rainbow",
                                 "aurora", "mosaic", "ukiyo-e", "a sketch with crayon"]
            prompt_candidates = [f"A transformed version of image into {style}" for style in style_list]
        else:
            raise NotImplementedError

        return prompt_candidates

    def augment_one_sample(self, sample_dict):
        # generation parameters for Real Guidance
        augment_ratio = self.cfg.generation.augment_ratio
        guidance_scale = self.cfg.generation.guidance_scale
        num_inference_steps = self.cfg.generation.num_inference_steps

        with torch.no_grad():
            sample_to_augment = transform_img2img(sample_dict["x"])
            original_image = transforms.ToPILImage()(sample_to_augment)
            init_images = repeat(sample_to_augment.unsqueeze(0), '1 ... -> b ...', b=augment_ratio)

            prompt_candidates = self.get_prompt_candidates()
            prompts_list = []
            for i in range(augment_ratio):
                prompts_list.append(random.choice(prompt_candidates))
            prompts = prompts_list
            samples = self.pipeline(prompt=prompts,
                                    image=init_images,
                                    num_images_per_prompt=1,
                                    num_inference_steps=num_inference_steps,
                                    guidance_scale=guidance_scale,
                                    generator=self.generator
                                    )[0]

            blended_images = []
            for i in range(len(samples)):
                augment_image = Utils.combine_images(original_image, samples[i])
                random_fractal_img = random.choice(self.fractal_images)
                blended_img = Utils.blend_images_with_resize(augment_image, random_fractal_img)
                blended_images.append(blended_img)

        return post_process(blended_images, original_image)

def post_process(samples, original_image):
    grid_list = [original_image]
    grid_list.extend(samples)
    grid_to_show = create_image_grid(grid_list, len(grid_list))
    return samples, grid_to_show

class Utils:
    @staticmethod
    def load_fractal_images(fractal_img_dir):
        fractal_img_paths = [os.path.join(fractal_img_dir, fname) for fname in os.listdir(fractal_img_dir) if fname.endswith(('.png', '.jpg', '.jpeg'))]
        return [Image.open(path).convert('RGB').resize((256, 256)) for path in fractal_img_paths]

    @staticmethod
    def blend_images_with_resize(base_img, overlay_img, alpha=0.20):
        overlay_img_resized = overlay_img.resize(base_img.size)
        base_array = np.array(base_img, dtype=np.float32)
        overlay_array = np.array(overlay_img_resized, dtype=np.float32)
        assert base_array.shape == overlay_array.shape and len(base_array.shape) == 3
        blended_array = (1 - alpha) * base_array + alpha * overlay_array
        blended_array = np.clip(blended_array, 0, 255).astype(np.uint8)
        blended_img = Image.fromarray(blended_array)
        return blended_img

    @staticmethod
    def combine_images(original_img, augmented_img, blend_width=20):
        width, height = original_img.size
        combine_choice = random.choice(['horizontal', 'vertical'])

        if combine_choice == 'vertical':  # Vertical combination
            mask = np.linspace(0, 1, blend_width).reshape(-1, 1)
            mask = np.tile(mask, (1, width))  # Extend mask horizontally
            mask = np.vstack([np.zeros((height // 2 - blend_width // 2, width)), mask,
                              np.ones((height // 2 - blend_width // 2 + blend_width % 2, width))])
            mask = np.tile(mask[:, :, np.newaxis], (1, 1, 3))

        else:
            mask = np.linspace(0, 1, blend_width).reshape(1, -1)
            mask = np.tile(mask, (height, 1))  # Extend mask vertically
            mask = np.hstack([np.zeros((height, width // 2 - blend_width // 2)), mask,
                              np.ones((height, width // 2 - blend_width // 2 + blend_width % 2))])
            mask = np.tile(mask[:, :, np.newaxis], (1, 1, 3))

        original_array = np.array(original_img, dtype=np.float32) / 255.0
        augmented_array = np.array(augmented_img, dtype=np.float32) / 255.0

        blended_array = (1 - mask) * original_array + mask * augmented_array
        blended_array = np.clip(blended_array * 255, 0, 255).astype(np.uint8)

        blended_img = Image.fromarray(blended_array)
        return blended_img

    @staticmethod
    def is_black_image(image):
        histogram = image.convert("L").histogram()
        return histogram[-1] > 0.9 * image.size[0] * image.size[1] and max(histogram[:-1]) < 0.1 * image.size[0] * \
            image.size[1]
