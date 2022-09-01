from typing import Tuple, List
import torch
import torch.nn as nn
from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline
from multiprocessing import Pool
import numpy as np
from PIL import Image


def get_gpu_setting(env_var: str) -> Tuple[bool, List[int]]:
    if not torch.cuda.is_available():
        print("GPU not detected! Make sure you have a GPU to reduce inference time!")
        return False, []
    # reads user input, returns multi_gpu flag and gpu id(s)
    n = torch.cuda.device_count()
    if env_var == "all":
        gpus = list(range(n))
    elif "," in env_var:
        gpus = [int(gnum) for gnum in env_var.split(",") if int(gnum) < n]
    else:
        gpus = [int(env_var)]
    assert len(
        gpus
    ), f"Make sure to provide valid device ids! You have {n} GPU(s), you can specify the following values: {list(range(n))}"
    return len(gpus) > 1, gpus


def create_cuda_model(device_id: int, kwargs) -> StableDiffusionPipeline:
    return StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4", **kwargs
    ).to(f"cuda:{device_id}")


def model_forward(model: StableDiffusionPipeline, prompts: List[str], kwargs):
    if not len(prompts):
        return []
    images: List[Image.Image] = model(prompts, **kwargs)["sample"]
    return images


class StableDiffusionContainer(object):
    def __init__(self, pipelines: List[StableDiffusionPipeline]) -> None:
        self.pipes: List[StableDiffusionPipeline] = pipelines

    def __setattr__(self, name: str, val) -> None:
        for pipe in self.pipes:
            setattr(pipe, name, val)

    def __call__(self, prompts, **kwargs):
        # run inference on different processes, each handles a model on a different GPU (split load evenly)
        prompts = [list(p) for p in np.array_split(prompts)]
        res = self.pool.starmap(
            model_forward, [(p, prompts[i], kwargs) for i, p in enumerate(self.pipes)]
        )
        # mimic interface
        return {"sample": np.concatenate(res)}

    @classmethod
    def from_pretrained(cls, devices: List[int], **kwargs):
        cls.pool = Pool(len(devices))
        # create models and move them to correspoding device

        models = cls.pool.starmap(
            create_cuda_model, [(d_id, kwargs) for d_id in devices]
        )
        return cls(models)

    def __del__(self):
        self.pool.terminate()

    # mimic interface
    def safety_checker(self):
        return self.pipes[0].safety_checker
