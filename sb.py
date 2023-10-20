from typing import Any, List, Optional, Union
from diffusers.pipelines.stable_diffusion import (
    StableDiffusionPipeline,
    StableDiffusionPipelineOutput,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
)
import torch
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)
from utils import remove_nsfw
from schedulers import schedulers
from transformers import CLIPFeatureExtractor


class DiffusionModel:
    def __init__(self, pipe: StableDiffusionPipeline = None) -> None:
        self.pipe: StableDiffusionPipeline = pipe
        self._safety: StableDiffusionSafetyChecker = None
        self._safety_extractor: CLIPFeatureExtractor = None
        self._pipe_name = ""
        self._device = torch.cpu

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        return cls()._load_pipeline(pretrained_model_name_or_path, **kwargs)

    def _load_pipeline(self, model_or_path, **kwargs):
        if self.pipe is not None and self._pipe_name == model_or_path:
            # avoid re-loading same model
            return

        print(f"Loading {model_or_path} from disk..")
        self.pipe = StableDiffusionPipeline.from_pretrained(
            pretrained_model_name_or_path=model_or_path, **kwargs
        )
        # remove safety checker so it doesn't use up GPU memory (by default)
        self._safety, self._safety_extractor = remove_nsfw(self.pipe)

        self._pipe_name = model_or_path
        print("Model Loaded!")
        return self

    def __call__(
        self, inference_type: str, *args: Any, **kwargs: Any
    ) -> StableDiffusionPipelineOutput:
        # NOTE: to avoid re-loading the model, we ""cast"" the pipeline
        if inference_type == "text":
            self.pipe.__class__ = StableDiffusionPipeline
        elif inference_type == "img2img":
            self.pipe.__class__ = StableDiffusionImg2ImgPipeline
        elif inference_type == "inpaint":
            self.pipe.__class__ = StableDiffusionInpaintPipeline
        # generator cant be pickled for multiprocessing, provide a coherent interface
        if kwargs.get("generator", None) is not None and kwargs["generator"] > 0:
            kwargs["generator"] = torch.Generator(self._device).manual_seed(
                kwargs["generator"]
            )
        else:
            kwargs.pop("generator", None)  # ignore seed < 0

        return self.pipe(*args, **kwargs)

    def reload_model(self, model_or_path: str, **kwargs):
        # this is separated from __call__ hoping that we can get a single model that can do inpainting and img2img without reloading
        return self._load_pipeline(model_or_path, **kwargs).to(
            self._device
        )  # maintain device!

    def to(self, device: Union[torch.device, str]):
        self.pipe.to(device)
        self._device = device
        return self

    def set_nsfw(self, nsfw: bool):
        if nsfw:
            # re- instatiate safety checkers
            self.pipe.safety_checker = self._safety_checker.to(self._device)
            self.pipe.feature_extractor = self._safety_extractor
        else:
            # ignore return value, we already have the safety network
            remove_nsfw(self.pipe)

    # mimic interface
    @property
    def scheduler(self):
        return self.pipe.scheduler

    @scheduler.setter
    def scheduler(self, scheduler: str):
        if self.scheduler.__class__.__name__ == schedulers[scheduler].__name__:
            # avoid re-setting same scheduler
            pass
        elif scheduler is not None and scheduler in schedulers:
            print(f"Setting noise scheduler to {scheduler}")
            # TODO use a default config instead of self.pipe.scheduler.config?
            s = getattr(schedulers[scheduler], "from_config")(
                self.pipe.scheduler.config
            )
            self.pipe.scheduler = s
        else:
            raise ValueError(f"Invalid Scheduler {scheduler}!")

    def enable_attention_slicing(self, slice_size: Optional[Union[str, int]] = "auto"):
        # TODO this can be further pushed
        # when slice_size is None, this is disabled
        return self.pipe.enable_attention_slicing(slice_size)
