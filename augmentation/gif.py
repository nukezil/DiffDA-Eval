import random
import gc
from einops import repeat
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import torchvision.transforms as transforms
import PIL
import clip
from diffusers import StableDiffusionImg2ImgPipeline, DPMSolverMultistepScheduler, DDIMScheduler
from utils.misc import create_image_grid
from utils.imresize import imresize

transform_augment = transforms.Compose([transforms.Resize((512,512), interpolation=PIL.Image.BICUBIC),
                                        transforms.RandomRotation(15,),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor()])

class GIFGenerator():
    """
        Implementation of [GIF-SD] from
        Expanding Small-Scale Datasets with Guided Imagination (NeurIPS 2023)
    """
    def __init__(self, cfg, dset):
        self.cfg = cfg
        self.pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
            cfg.generation.sd_model,
            torch_dtype=torch.float16,
            revision="fp16",
            safety_checker=None,
            feature_extractor=None
        ).to("cuda")
        self.pipeline.set_progress_bar_config(disable=True)

        self.dset = dset  # transform should be None (sample["x"] is PIL.Image)
        self.is_medical_dataset = self.cfg.data.dataset.startswith("Blood") or self.cfg.data.dataset.startswith("Skin")
        if self.is_medical_dataset:
            # For medical datasets, use finetuned SD model
            self.load_ti_embeds()
            self.pipeline.load_lora_weights(cfg.generation.db_lora_weights,
                                            weight_name="pytorch_lora_weights.safetensors")
            print(f"DreamBooth Weights {self.cfg.generation.db_lora_weights} loaded!")


        if cfg.generation.scheduler == "DPMSolver++":
            self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(self.pipeline.scheduler.config)
        elif cfg.generation.scheduler == "DDIM":
            self.pipeline.scheduler = DDIMScheduler.from_config(self.pipeline.scheduler.config)
        else:
            raise ValueError("Scheduler not supported")


        self.clip_model, self.clip_preprocess, self.text_classifier = self.prepare_clip_model(dset.class_names)

        self.generator = torch.Generator(device="cuda").manual_seed(cfg.seed)

    def prepare_clip_model(self, class_names):
        clip_model, clip_preprocess = clip.load(self.cfg.generation.clip_model, device="cuda")
        text_descriptions = [f"This is a photo of a {label}" for label in class_names]
        text_tokens = clip.tokenize(text_descriptions).cuda()
        text_features = clip_model.encode_text(text_tokens).float()
        text_features /= text_features.norm(dim=-1, keepdim=True)
        text_classifier = text_features
        return clip_model, clip_preprocess, text_classifier,

    def get_prompt_template(self):
        if self.cfg.data.dataset in ["Caltech101", "CIFAR100", "ImageNet100", "Aircraft", "Birds", "Dogs"]:
            domain_list = ["an image of", "a real-world photo of", "a cartoon image of", "an oil painting of",
                           "a sketch of"]
            adjective_list = ["", "colorful", "stylized", "high-contrast", "low-contrast", "posterized", "solarized",
                              "sheared", "bright", "dark"]
            return domain_list, adjective_list
        elif self.cfg.data.dataset.startswith("Blood"):
            prompt_template = "a microscopic blood cell image of <{}>"
            return prompt_template
        elif self.cfg.data.dataset.startswith("Skin"):
            prompt_template = "a dermatoscopic image of <{}>"
            return prompt_template
        else:
            raise NotImplementedError

    def augment_one_sample(self, sample_dict):
        # generation parameters for GIF
        augment_ratio = self.cfg.generation.augment_ratio
        strength = self.cfg.generation.strength
        num_inference_steps = self.cfg.generation.num_inference_steps
        constraint_value = self.cfg.generation.constraint_value

        with torch.no_grad():
            original_sample = self.clip_preprocess(sample_dict["x"])
            sample_to_augment = transform_augment(sample_dict["x"])
            init_images = repeat(sample_to_augment.unsqueeze(0), '1 ... -> b ...', b=augment_ratio)

            if not self.is_medical_dataset:
                domain_list, adjective_list = self.get_prompt_template()
            else:
                prompt_template = self.get_prompt_template()

            prompts_list = []
            for i in range(augment_ratio):
                if not self.is_medical_dataset:
                    prompts_list.append("{} {} {}".format(random.choice(domain_list), random.choice(adjective_list),
                                                      sample_dict["class_name"]))
                else:
                    prompts_list.append(prompt_template.format(sample_dict["class_name"]))

            prompts = prompts_list
            sample_latents = self.pipeline(prompt=prompts,
                                           image=init_images,
                                           strength=strength,
                                           num_inference_steps=num_inference_steps,
                                           output_type="latent",
                                           generator=self.generator
                                           )[0]

            original_inputs = original_sample.unsqueeze(0).cuda()
            image_original_features = self.clip_model.encode_image(original_inputs).float().detach()
            image_original_features = image_original_features / image_original_features.norm(dim=-1, keepdim=True)
            original_outputs = (100.0 * image_original_features @ self.text_classifier.T)
            original_output_prob = original_outputs.softmax(dim=-1)
            original_output_index = original_output_prob.argmax(dim=1)
            original_output_entropy = -(original_output_prob * torch.log(original_output_prob)).sum(dim=1)

        samples_temp = sample_latents.clone()
        channel_noise_dim = samples_temp.shape[1]
        channel_noise = Variable(torch.rand([augment_ratio, channel_noise_dim, 1, 1]), requires_grad=True).cuda()
        channel_noise_bias = Variable(torch.zeros([augment_ratio, channel_noise_dim, 1, 1]).data.normal_(0, 1), requires_grad=True).cuda()
        samples_temp = samples_temp * (1 + channel_noise) + channel_noise_bias

        samples_temp = 1 / self.pipeline.vae.config.scaling_factor * samples_temp
        x_samples = self.pipeline.vae.decode(samples_temp.to(torch.float16), return_dict=False)[0]
        x_samples = (x_samples / 2 + 0.5).clamp(0, 1)

        augmented_inputs1 = imresize(x_samples, sizes=(224, 224))
        image_augmented_features = self.clip_model.encode_image(augmented_inputs1).float()
        image_augmented_features = image_augmented_features / image_augmented_features.norm(dim=-1, keepdim=True)
        augmented_output_prob = (100.0 * image_augmented_features @ self.text_classifier.T).softmax(dim=-1)
        augmented_output_entropy = -(augmented_output_prob * torch.log(augmented_output_prob)).sum(dim=1)

        # Class-maintained informativeness
        delta_entropy = augmented_output_entropy - original_output_entropy
        delta_entropy = delta_entropy.mean()
        original_output_index1 = original_output_index.expand(augmented_output_prob.size(0), 1)
        augmented_output_prob_top1 = torch.gather(augmented_output_prob, 1, original_output_index1).mean()

        # Sample diversity
        kl_loss = nn.KLDivLoss(reduction="batchmean", log_target=True)
        mean_image_augmented_features = torch.mean(image_augmented_features, 0, keepdim=True).repeat(
            [augment_ratio, 1])
        image_augmented_features = F.log_softmax(image_augmented_features, dim=1)
        target_mean_image_augmented_features = F.log_softmax(mean_image_augmented_features, dim=1)
        divergence = kl_loss(image_augmented_features, target_mean_image_augmented_features)

        score = delta_entropy + augmented_output_prob_top1 + divergence
        (channel_noise_grad, channel_noise_bias_grad) = torch.autograd.grad(score, [channel_noise, channel_noise_bias])
        channel_noise.data.add_(0.1 * torch.sign(channel_noise_grad))
        channel_noise_bias.data.add_(0.1 * torch.sign(channel_noise_bias_grad))

        del samples_temp, x_samples, augmented_inputs1, image_augmented_features, augmented_output_prob, augmented_output_entropy, delta_entropy, augmented_output_prob_top1, divergence, channel_noise_grad, channel_noise_bias_grad
        gc.collect()
        torch.cuda.empty_cache()

        with torch.no_grad():
            samples_temp = sample_latents.clone()
            sample_latents = sample_latents * (1 + channel_noise.to(torch.float16)) + channel_noise_bias.to(torch.float16)
            linfball_proj(samples_temp, constraint_value, sample_latents, in_place=True)
            sample_latents = 1 / self.pipeline.vae.config.scaling_factor * sample_latents
            x_samples = self.pipeline.vae.decode(sample_latents.to(torch.float16), return_dict=False)[0]
            x_samples = (x_samples / 2 + 0.5).clamp(0, 1)

        return post_process(x_samples.cpu(), sample_to_augment)

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

def post_process(x_samples, samples_to_augment):
    augment_samples = []
    for x_sample in x_samples:
        augment_samples.append(transforms.ToPILImage()(x_sample))
    grid_list = [transforms.ToPILImage()(samples_to_augment)]
    grid_list.extend(augment_samples)
    grid_to_show = create_image_grid(grid_list, len(grid_list))
    return augment_samples, grid_to_show

def tensor_clamp(t, min, max, in_place=True):
    if not in_place:
        res = t.clone()
    else:
        res = t
    idx = res.data < min
    res.data[idx] = min[idx]
    idx = res.data > max
    res.data[idx] = max[idx]
    return res

def linfball_proj(center, radius, t, in_place=True):
    return tensor_clamp(t, min=center - radius, max=center + radius, in_place=in_place)