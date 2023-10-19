from typing import Any, List
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
from transformers import CLIPFeatureExtractor

class DiffusionModel:

    def __init__(self, pipe: StableDiffusionPipeline=None) -> None:
        self.pipe: StableDiffusionPipeline = pipe
        self._safety: StableDiffusionSafetyChecker = None
        self._safety_extractor: CLIPFeatureExtractor = None
        self._pipe_name = ""
        self._device = None
    
    @classmethod
    def from_pretrained(model_or_path, **kwargs):
        return DiffusionModel()._load_pipeline(model_or_path, 0, **kwargs)

    def _load_pipeline(self, model_or_path, **kwargs):
        if self.pipe is not None and self._pipe_name == model_or_path:
            # avoid re-loading same model
            return

        print(f"Loading {model_or_path} from disk..")
        self.pipe = StableDiffusionPipeline.from_pretrained(pretrained_model_name_or_path=model_or_path,**kwargs)
        # remove safety checker so it doesn't use up GPU memory (by default)
        self._safety, self._safety_extractor = remove_nsfw(self.pipe)

        self._pipe_name = model_or_path
        print("Model Loaded!")
        return self
    
    def __call__(self, inference_type: str, prompt: str, *args: Any, **kwds: Any) -> Any:
        # NOTE: to avoid re-loading the model, we ""cast"" the pipeline
        if inference_type == "text":
            self.pipe.__class__ = StableDiffusionPipeline
        elif inference_type == "img2img":
            self.pipe.__class__ = StableDiffusionImg2ImgPipeline
        elif inference_type == "inpaint":
            self.pipe.__class__ = StableDiffusionInpaintPipeline
    
    def reload_model(self, model_or_path: str, **kwargs):
        # this is separated from __call__ hoping that we can get a single model that can do inpainting and img2img without reloading
        return self._load_pipeline(model_or_path, **kwargs).to(self._device) # maintain device!

    def to(self, device: Union[torch.device, str]):
        self.pipe.to(device)
        self._device = device
        return self