from typing import List
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)
import os
from diffusers.schedulers import *

# setup noise schedulers
schedulers_names = [
    "DDIM",
    "PNDM",
    "K-LMS linear",
    "K-LMS scaled",
]
schedulers_cls = [
    DDIMScheduler,
    PNDMScheduler,
    LMSDiscreteScheduler,
    LMSDiscreteScheduler,
]
# default PNDM parameters
schedulers_args = [
    dict(),
    {
        "beta_end": 0.012,
        "beta_schedule": "scaled_linear",
        "beta_start": 0.00085,
        "num_train_timesteps": 1000,
        "skip_prk_steps": True,
    },
    dict(),
    dict(beta_schedule="scaled_linear"),
]
# scheduler_name -> (scheduler_class, scheduler_args)
schedulers = dict(zip(schedulers_names, zip(schedulers_cls, schedulers_args)))


def dummy_checker(images, **kwargs):
    return images, False


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


TOKEN = os.environ.get("TOKEN", None)
fp16 = bool(os.environ.get("FP16", True))
if TOKEN is None:
    raise Exception(
        "Unable to read huggingface token! Make sure to get your token here https://huggingface.co/settings/tokens and set the corresponding env variable with `docker run --env TOKEN=<YOUR_TOKEN>`"
    )

print("Loading model..")
from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline

# If you are limited by GPU memory and have less than 10GB of GPU RAM available, please make sure to load the StableDiffusionPipeline in float16 precision
pipe: StableDiffusionPipeline = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    revision="fp16" if fp16 else None,
    torch_dtype=torch.float16 if fp16 else None,
    use_auth_token=TOKEN,
)
safety: StableDiffusionSafetyChecker = pipe.safety_checker
if torch.cuda.is_available():
    print(f"Moving model to {torch.cuda.get_device_name()}..")
    pipe.to("cuda:0")
print("Ready!")


def inference(
    prompt,
    num_images=1,
    num_inference_steps=50,
    height=512,
    width=512,
    guidance_scale=7,
    seed=None,
    nsfw_filter=False,
    noise_scheduler=None,
):
    # for repeatable results
    generator = (
        torch.Generator("cuda").manual_seed(seed)
        if seed is not None and seed > 0
        else None
    )

    if nsfw_filter:
        pipe.safety_checker = safety
    else:
        pipe.safety_checker = dummy_checker

    # set noise scheduler for inference
    if noise_scheduler is not None and noise_scheduler in schedulers:
        scls, skwargs = schedulers[noise_scheduler]
        pipe.scheduler = scls(**skwargs)

    prompt = [prompt] * num_images
    # number of denoising steps run during inference (the higher the better)
    with torch.autocast("cuda"):
        images: List[Image.Image] = pipe(
            prompt,
            num_inference_steps=num_inference_steps,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            generator=generator,
        )["sample"]
    # image.show()

    return images


if __name__ == "__main__":
    images = inference(input("Input prompt:"))
    grid = image_grid(images, rows=1, cols=1)
    grid.show()
