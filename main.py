from typing import List, Union
from diffusers import StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion import (
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
)
import torch
from PIL import Image
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)
import os
from utils import ModelParts2GPUsAssigner, get_gpu_setting, dummy_checker, remove_nsfw
from parallel import StableDiffusionModelParallel, StableDiffusionMultiProcessing
from schedulers import schedulers
import numpy as np

TOKEN = os.environ.get("TOKEN", None)
MODEL_ID = os.environ.get("MODEL_ID", "stabilityai/stable-diffusion-2-base")

fp16 = bool(int(os.environ.get("FP16", 1)))
# MP = bool(int(os.environ.get("MODEL_PARALLEL", 0)))
MP = False  # disabled
MIN_INPAINT_MASK_PERCENT = 0.1

from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline
# FIXME devices=0,1 causes cuda error on memory access..?
# create and move model to GPU(s), defaults to GPU 0
multi, devices = get_gpu_setting(os.environ.get("DEVICES", "0"))
# If you are limited by GPU memory and have less than 10GB of GPU RAM available, please make sure to load the StableDiffusionPipeline in float16 precision
kwargs = dict(
    pretrained_model_name_or_path=MODEL_ID,
    revision="fp16" if fp16 else None,
    torch_dtype=torch.float16 if fp16 else None,
    use_auth_token=TOKEN,
    requires_safety_checker=False,
)

pipe, safety, safety_extractor = None, None, None

def load_pipeline(model_or_path, devices: List[int]):
    global pipe, safety, safety_extractor
    if pipe is not None and pipe._pipe_name == model_or_path:
        # avoid re-loading same model
        return

    model_ass = None
    print(f"Loading {model_or_path} from disk..")
    kwargs["pretrained_model_name_or_path"] = model_or_path
    # single-gpu multiple models currently disabled
    if MP and len(devices) > 1:
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

    if multi and pipe is not None:
        # avoid re-creating processes in multi-gpu mode, have them reload a different model
        pipe.reload_model(model_or_path)
    elif multi:
        # DataParallel: one process *per GPU* (each has a copy of the model)
        # ModelParallel: one process *per model*, each model (possibly) on multiple GPUs
        n_procs = len(devices) if not MP else len(model_ass)
        pipe = StableDiffusionMultiProcessing.from_pretrained(
            n_procs, devices, model_parallel_assignment=model_ass, **kwargs
        )
    else:
        pipe = StableDiffusionPipeline.from_pretrained(**kwargs)
        # remove safety checker so it doesn't use up GPU memory
        safety, safety_extractor = remove_nsfw(pipe)
        if len(devices):
            pipe.to(f"cuda:{devices[0]}")
    
    pipe._pipe_name = model_or_path
    print("Model Loaded!")

load_pipeline(MODEL_ID, devices)


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
    inv_strenght=0.0,
    input_image=None,
    input_sketch=None,
    masked_image=None
):
    prompt = [prompt] * num_images
    input_kwargs = dict(
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        height=height,
        width=width,
        guidance_scale=guidance_scale,
        generator=None,
    )
    # input sketch has priority over input image
    if input_sketch is not None:
        input_image = input_sketch

    # Img2Img: to avoid re-loading the model, we ""cast"" the pipeline
    if input_image is not None:
        input_image = input_image.resize((width, height)) 
        # image guided generation
        if multi:
            pipe.change_pipeline_type("img2img")
        else:
            pipe.__class__ = StableDiffusionImg2ImgPipeline
        # TODO negative prompt?
        input_kwargs["init_image"] = [input_image] * num_images
        input_kwargs["strength"] = 1.0 - inv_strenght
    elif masked_image is not None:
        # TODO load inpainting model
        # resize to specified shape
        masked_image = {
            k: v.convert("RGB").resize((width, height)) for k, v in masked_image.items()
        }

        # to do image inpainting, we must provide a big enough mask
        if np.count_nonzero(masked_image["mask"].convert("1")) < (
            width * height * MIN_INPAINT_MASK_PERCENT
        ):
            # FIXME error handling
            raise Exception("ERROR: mask is too small!")
        if multi:
            pipe.change_pipeline_type("inpaint")
        else:
            pipe.__class__ = StableDiffusionInpaintPipeline
        # TODO extra fields? does this weak copy work here??
        # input_kwargs["prompt"] = "portrait of a man selling paintings with passion"
        input_kwargs["image"] = masked_image["image"]
        input_kwargs["mask_image"] = masked_image["mask"] 
    elif multi:
        # default mode
        pipe.change_pipeline_type("text")
    else:
        pipe.__class__ = StableDiffusionPipeline

    # for repeatable results; tensor generated on cpu for model parallel
    if multi:
        # generator cant be pickled
        # NOTE fixed seed with multiples gpus will be different for each one but fixed!
        input_kwargs["generator"] = seed
    elif seed is not None and seed > 0:
        input_kwargs["generator"] = torch.Generator(
            f"cuda:{devices[0]}" if not MP else "cpu"
        ).manual_seed(seed)

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
    if noise_scheduler is not None and noise_scheduler in schedulers:
        if multi:
            pipe.scheduler = noise_scheduler
        else:
            # load scheduler from pre-trained config
            s = getattr(schedulers[noise_scheduler], "from_config")(
                pipe.scheduler.config
            )
            pipe.scheduler = s

    # number of denoising steps run during inference (the higher the better)
    with torch.autocast("cuda"):
        images: List[Image.Image] = pipe(**input_kwargs)["images"]
    return images


if __name__ == "__main__":
    from utils import image_grid

    images = inference(input("Input prompt:"))
    grid = image_grid(images, rows=1, cols=1)
    grid.show()
