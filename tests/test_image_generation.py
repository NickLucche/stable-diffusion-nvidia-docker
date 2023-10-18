from main import load_pipeline, inference
from diffusers import StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion import (
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
)
import pytest
from typing import List
from PIL import Image
import torch
import numpy as np

PROMPT = "A starry night"


@pytest.fixture
def txt2img() -> StableDiffusionPipeline:
    pipe = StableDiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-base"
    )
    if torch.cuda.is_available():
        pipe.to(torch.device("cuda"))
    return pipe


def requires_cuda(func):
    def wrapper(*args, **kwargs):
        if not torch.cuda.is_available():
            pytest.skip("This test needs a GPU")
        return func(*args, **kwargs)

    return wrapper


# these tests have to run on cpu..
def test_txt2img(txt2img: StableDiffusionPipeline):
    input_kwargs = dict(
        prompt=PROMPT,
        num_inference_steps=3,
        height=512,
        width=512,
        generator=None,
    )
    images: List[Image.Image] = txt2img(**input_kwargs)["images"]
    assert images[0].size == (512, 512)


@requires_cuda
def test_txt2img_pipeline():
    load_pipeline("stabilityai/stable-diffusion-2-base", [0])
    images = inference(
        PROMPT, num_images=1, num_inference_steps=3, height=512, width=512
    )
    assert len(images) == 1 and images[0].size == (512, 512)


@requires_cuda
def test_img2img_pipeline():
    load_pipeline("stabilityai/stable-diffusion-2-base", [0])
    image = Image.open("./assets/0.png")
    images = inference(
        PROMPT,
        num_images=1,
        num_inference_steps=3,
        height=512,
        width=512,
        input_image=image,
        inv_strenght=0.5
    )
    assert len(images) == 1 and images[0].size == (512, 512)


@requires_cuda
def test_imginpainting_pipeline():
    load_pipeline("stabilityai/stable-diffusion-2-inpainting", [0])
    image = Image.open("./assets/0.png")
    # mask image
    mask = np.array(image)
    mask[:, : image.size[0] // 2] = 0
    mask = Image.fromarray(mask)
    images = inference(
        PROMPT,
        num_images=1,
        num_inference_steps=3,
        height=512,
        width=512,
        masked_image={"image": image, "mask": mask},
    )
    assert len(images) == 1 and images[0].size == (512, 512)
    # masked part more diverse than the "fixed" one
    res, source = np.array(images[0]), np.array(image)
    assert (source[:, : image.size[0] // 2] - res[:, : image.size[0] // 2]).sum() < (source[:, image.size[0] // 2:] - res[:, image.size[0] // 2:]).sum()