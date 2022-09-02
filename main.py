from typing import List
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)
import os
from utils import StableDiffusionMultiProcessing, get_gpu_setting, dummy_checker
from schedulers import schedulers


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

# create and move model to GPU(s), defaults to GPU 0
multi, devices = get_gpu_setting(os.environ.get("DEVICES", "0"))
# If you are limited by GPU memory and have less than 10GB of GPU RAM available, please make sure to load the StableDiffusionPipeline in float16 precision
kwargs = dict(
    pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4",
    revision="fp16" if fp16 else None,
    torch_dtype=torch.float16 if fp16 else None,
    use_auth_token=TOKEN,
)

if multi:
    # "data parallel", replicate the model on multiple gpus, each is handled by a different process
    pipe: StableDiffusionMultiProcessing = StableDiffusionMultiProcessing.from_pretrained(
        devices, **kwargs
    )
    safety: StableDiffusionSafetyChecker = None
else:
    pipe: StableDiffusionPipeline = StableDiffusionPipeline.from_pretrained(**kwargs)
    safety: StableDiffusionSafetyChecker = pipe.safety_checker
    if len(devices):
        pipe.to(f"cuda:{devices[0]}")


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
        if multi:
            pipe.scheduler = noise_scheduler
        else:
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
