from typing import List, Union
import torch
from PIL import Image
import os
from utils import ModelParts2GPUsAssigner, get_gpu_setting
from parallel import StableDiffusionModelParallel, StableDiffusionMultiProcessing
import numpy as np
from sb import DiffusionModel

# read env variables
TOKEN = os.environ.get("TOKEN", None)
MODEL_ID = os.environ.get("MODEL_ID", "stabilityai/stable-diffusion-2-base")

# If you are limited by GPU memory (e.g <10GB VRAM), please make sure to load in fp16 precision
fp16 = bool(int(os.environ.get("FP16", 1)))
# MP = bool(int(os.environ.get("MODEL_PARALLEL", 0)))
MP = False  # disabled
MIN_INPAINT_MASK_PERCENT = 0.1

# FIXME devices=0,1 causes cuda error on memory access..?
IS_MULTI, DEVICES = get_gpu_setting(os.environ.get("DEVICES", "0"))

# TODO docs
def init_pipeline(model_or_path=MODEL_ID, devices: List[int]=DEVICES)->Union[DiffusionModel, StableDiffusionMultiProcessing]:
    kwargs = dict(
        pretrained_model_name_or_path=model_or_path,
        revision="fp16" if fp16 else None,
        torch_dtype=torch.float16 if fp16 else None,
        use_auth_token=TOKEN,
        requires_safety_checker=False,
    )
    model_ass = None
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

    # TODO move logic
    # if multi and pipe is not None:
        # avoid re-creating processes in multi-gpu mode, have them reload a different model
        # pipe.reload_model(model_or_path)
    if IS_MULTI:
        # DataParallel: one process *per GPU* (each has a copy of the model)
        # ModelParallel: one process *per model*, each model (possibly) on multiple GPUs
        n_procs = len(devices) if not MP else len(model_ass)
        pipe = StableDiffusionMultiProcessing.from_pretrained(
            n_procs, devices, model_parallel_assignment=model_ass, **kwargs
        )
    else:
        pipe = DiffusionModel.from_pretrained(**kwargs)
        if len(devices):
            pipe.to(f"cuda:{devices[0]}")

    return pipe


def inference(
    pipe: DiffusionModel,
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
    masked_image=None,
):
    prompt = [prompt] * num_images
    input_kwargs = dict(
        inference_type = "text",
        prompt=prompt,
        # number of denoising steps run during inference (the higher the better)
        num_inference_steps=num_inference_steps,
        height=height,
        width=width,
        guidance_scale=guidance_scale,
        # NOTE seed with multiples gpus will be different for each one but fixed!
        generator=seed,
    )
    # input sketch has priority over input image
    if input_sketch is not None:
        input_image = input_sketch

    # TODO batch images by providing a torch tensor
    if input_image is not None:
        # image guided generation
        input_image = input_image.resize((width, height))
        # TODO negative prompt?
        input_kwargs["init_image"] = input_image
        input_kwargs["strength"] = 1.0 - inv_strenght
        input_kwargs["inference_type"] = "img2img"
    elif masked_image is not None:
        # resize to specified shape
        masked_image = {
            k: v.convert("RGB").resize((width, height)) for k, v in masked_image.items()
        }

        # to do image inpainting, we must provide a big enough mask
        if np.count_nonzero(masked_image["mask"].convert("1")) < (
            width * height * MIN_INPAINT_MASK_PERCENT
        ):
            raise ValueError("Mask is too small. Please paint-over a larger area")
        input_kwargs["image"] = masked_image["image"]
        input_kwargs["mask_image"] = masked_image["mask"]
        input_kwargs["inference_type"] = "inpaint"

    pipe.set_nsfw(nsfw_filter)

    # needed on 16GB RAM 768x768 fp32
    pipe.enable_attention_slicing("auto" if low_vram else None)

    # set noise scheduler for inference
    if noise_scheduler is not None:
        pipe.scheduler = noise_scheduler

    with torch.autocast("cuda"):
        images: List[Image.Image] = pipe(**input_kwargs)["images"]
    return images


if __name__ == "__main__":
    from utils import image_grid

    images = inference(input("Input prompt:"))
    grid = image_grid(images, rows=1, cols=1)
    grid.show()
