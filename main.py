from typing import List
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)
import os
from utils import ModelParts2GPUsAssigner, get_gpu_setting, dummy_checker, remove_nsfw
from parallel import StableDiffusionModelParallel, StableDiffusionMultiProcessing
from schedulers import schedulers


TOKEN = os.environ.get("TOKEN", None)
TORCH_COMPILE = os.environ.get("TORCH_COMPILE", None)
# TODO change from UI and require reload
MODEL_ID = os.environ.get("MODEL_ID", "stabilityai/stable-diffusion-2-base")

fp16 = bool(int(os.environ.get("FP16", 1)))
# MP = bool(int(os.environ.get("MODEL_PARALLEL", 0)))
MP = False # disabled
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
    pretrained_model_name_or_path=MODEL_ID,
    revision="fp16" if fp16 else None,
    torch_dtype=torch.float16 if fp16 else None,
    use_auth_token=TOKEN,
    requires_safety_checker=False
)

model_ass = None
# single-gpu multiple models currently disabled
if MP and len(devices)>1:
    # setup for model parallel: find model parts->gpus assignment
    print(
        f"Looking for a valid assignment in which to split model parts to device(s): {devices}"
    )
    ass_finder = ModelParts2GPUsAssigner(devices)
    model_ass = ass_finder()
    if not len(model_ass):
        raise Exception(
            "Unable to find a valid assignment of model parts to GPUs! This could be bad luck in sampling!"
        )
    print("Assignments:", model_ass)

if multi:
    # DataParallel: one process *per GPU* (each has a copy of the model)
    # ModelParallel: one process *per model*, each model (possibly) on multiple GPUs
    n_procs = len(devices) if not MP else len(model_ass)
    pipe = StableDiffusionMultiProcessing.from_pretrained(
        n_procs, devices, model_parallel_assignment=model_ass, **kwargs
    )
else:
    pipe: StableDiffusionPipeline = StableDiffusionPipeline.from_pretrained(
        **kwargs
    )
    # remove safety checker so it doesn't use up GPU memory
    safety, safety_extractor = remove_nsfw(pipe)
    if len(devices):
        pipe.to(f"cuda:{devices[0]}")
    if TORCH_COMPILE:
        pipe.unet = torch.compile(pipe.unet)


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
    low_vram=False,
    noise_scheduler=None,
):
    # for repeatable results; tensor generated on cpu for model parallel
    if multi:
        # generator cant be pickled
        # NOTE fixed seed with multiples gpus should be different for each one but fixed!
        generator = seed
    else:
        generator = (
            torch.Generator(f"cuda:{devices[0]}" if not MP else "cpu").manual_seed(seed)
            if seed is not None and seed > 0
            else None
        )

    if nsfw_filter:
        if multi:
            pipe.safety_checker = None
        else:
            pipe.safety_checker = safety.to(f"cuda:{devices[0]}")
            pipe.feature_extractor = safety_extractor
    else:
        if multi:
            pipe.safety_checker = dummy_checker
        else:
            # remove safety network from gpu
            remove_nsfw(pipe)
    
    if low_vram:
        # needed on 16GB RAM 768x768 fp32
        pipe.enable_attention_slicing()
    else:
        pipe.disable_attention_slicing()

    # set noise scheduler for inference
    # TODO nsfw not working
    if noise_scheduler is not None and noise_scheduler in schedulers:
        if multi:
            pipe.scheduler = noise_scheduler
        else:
            # load scheduler from pre-trained config
            s = getattr(schedulers[noise_scheduler], "from_config")(pipe.scheduler.config)
            pipe.scheduler = s

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
        )["images"]
    # image.show()

    return images


if __name__ == "__main__":
    from utils import image_grid

    images = inference(input("Input prompt:"))
    grid = image_grid(images, rows=1, cols=1)
    grid.show()
