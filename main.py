from typing import List
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)
import os


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


# TODO config file for token, fp16 loading, gpu
TOKEN = os.environ.get("TOKEN", None)  # "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"
fp16 = bool(os.environ.get("FP16", True))
if TOKEN is None:
    raise Exception(
        "Unable to read huggingface token! Make sure to get your token here https://huggingface.co/settings/tokens and set the corresponding env variable with `docker run --env TOKEN=<YOUR_TOKEN>`"
    )

print("Loading model..")
from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline
pipe: StableDiffusionPipeline = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    revision="fp16" if fp16 else None,
    torch_dtype=torch.float16 if fp16 else None,
    use_auth_token=TOKEN,
)
safety: StableDiffusionSafetyChecker = pipe.safety_checker
if torch.cuda.is_available():
    print(f"Moving model to {torch.cuda.get_device_name()}..")
    pipe.to("cuda")
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
):
    # for repeatable results
    generator = (
        torch.Generator("cuda").manual_seed(seed)
        if seed is not None and seed > 0
        else None
    )

    # If you are limited by GPU memory and have less than 10GB of GPU RAM available, please make sure to load the StableDiffusionPipeline in float16 precision
    if nsfw_filter:
        pipe.safety_checker = safety
    else:
        pipe.safety_checker = dummy_checker

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
