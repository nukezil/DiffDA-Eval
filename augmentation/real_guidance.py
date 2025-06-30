from einops import repeat
from omegaconf import OmegaConf
import torch
import torchvision.transforms as transforms
import PIL
from diffusers import StableDiffusionImg2ImgPipeline, DPMSolverMultistepScheduler, DDIMScheduler

from utils.misc import create_image_grid

transform_img2img = transforms.Compose([transforms.Resize((512,512), interpolation=PIL.Image.BICUBIC),
                                        transforms.ToTensor()])

class RealGuidanceGenerator():
    """
        Implementation of [Real-Guidance] from
        Is Synthetic Data from Generative Models Ready for Image Recognition (ICLR 2023)
    """
    def __init__(self, cfg, dset):
        self.cfg = cfg
        self.dset = dset
        self.pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
            cfg.generation.sd_model,
            torch_dtype=torch.float16,
            revision="fp16",
            safety_checker=None,
            feature_extractor=None
        ).to("cuda")
        self.pipeline.set_progress_bar_config(disable=True)

        self.ti_flag = False
        if OmegaConf.select(cfg, "generation.ti_embeds") is not None:
            self.load_ti_embeds()
            self.ti_flag = True
        if OmegaConf.select(cfg, "generation.db_lora_weights") is not None:
            self.pipeline.load_lora_weights(cfg.generation.db_lora_weights,
                                            weight_name="pytorch_lora_weights.safetensors")
            print(f"Dreambooth Weights {cfg.generation.db_lora_weights} loaded!")

        if cfg.generation.scheduler == "DPMSolver++":
            self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(self.pipeline.scheduler.config)
        elif cfg.generation.scheduler == "DDIM":
            self.pipeline.scheduler = DDIMScheduler.from_config(self.pipeline.scheduler.config)
        else:
            raise ValueError("Scheduler not supported")

        self.generator = torch.Generator(device="cuda").manual_seed(cfg.seed)

    def get_prompt_template(self):
        if self.ti_flag:
            prompt_template = "a photo of <{}>"
        else:
            prompt_template = "a photo of {}"
        return prompt_template

    def augment_one_sample(self, sample_dict):
        # generation parameters for Real Guidance
        augment_ratio = self.cfg.generation.augment_ratio
        strength = self.cfg.generation.strength
        num_inference_steps = self.cfg.generation.num_inference_steps

        with torch.no_grad():
            sample_to_augment = transform_img2img(sample_dict["x"])
            init_images = repeat(sample_to_augment.unsqueeze(0), '1 ... -> b ...', b=augment_ratio)

            prompt_template = self.get_prompt_template()
            prompts_list = []
            for i in range(augment_ratio):
                prompts_list.append(prompt_template.format(sample_dict["class_name"]))
            prompts = prompts_list
            samples = self.pipeline(prompt=prompts,
                                           image=init_images,
                                           strength=strength,
                                           num_inference_steps=num_inference_steps,
                                           generator=self.generator
                                           )[0]

        return post_process(samples, sample_to_augment)

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

def post_process(samples, sample_to_augment):
    grid_list = [transforms.ToPILImage()(sample_to_augment)]
    grid_list.extend(samples)
    grid_to_show = create_image_grid(grid_list, len(grid_list))
    return samples, grid_to_show