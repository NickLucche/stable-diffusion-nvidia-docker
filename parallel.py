from ctypes import Union
import numpy as np
from PIL import Image
import torch
import torch.multiprocessing as mp
from transformers import CLIPConfig
from schedulers import schedulers
import pickle
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)
from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline
from utils import ToGPUWrapper, dummy_checker, dummy_extractor
from typing import Any, Dict, List

## Data Parallel: each process handles a copy of the model, executed on a different device ##
def cuda_inference_process(
    device_id: int, in_q: mp.Queue, out_q: mp.Queue, model_kwargs: Dict[Any, Any]
):
    """Code executed by the torch.multiprocessing process, handling inference on device `device_id`.
    It's a simple loop in which the worker pulls data from a shared input queue, and puts result
    into an output queue.
    """
    mp_ass: Dict[int, int]= model_kwargs.pop("mp_assignment", None)
    try:
        print(
            f"Creating and moving model to cuda:{device_id} ({torch.cuda.get_device_name(device_id)}).."
        )
        if mp_ass is None:
            model: StableDiffusionPipeline = StableDiffusionPipeline.from_pretrained(
                **model_kwargs
            ).to(f"cuda:{device_id}")
        else:
            model = StableDiffusionModelParallel.from_pretrained(**model_kwargs)
        # disable nsfw filter by default
        model.safety_checker = dummy_checker
        # create nsfw clip filter so we can re-set it if needed
        # TODO perhaps we can skip this if no one cares about nsfw
        safety_checker = StableDiffusionSafetyChecker(
            CLIPConfig(**model_kwargs.pop("clip_config"))
        )
        out_q.put(True)
    except:
        out_q.put(False)
        return
    # inference loop
    while True:
        # get prompt
        prompts, kwargs = in_q.get()
        if type(prompts) is not list:
            # special commands
            if prompts == "quit":
                break
            elif prompts == "safety_checker":
                # safety checker needs to be moved to GPU (it can cause crashes)
                if kwargs == "clip":
                    model.safety_checker = safety_checker.to(f"cuda:{device_id}")
                else:
                    safety_checker = safety_checker.to(f"cpu")
                    model.safety_checker = dummy_checker
            elif prompts == "scheduler":
                scls, skwargs = schedulers[kwargs]
                model.scheduler = scls(**skwargs)
            continue
        if not len(prompts):
            images = []
        else:
            # actual inference
            # print("Inference", prompts, kwargs, model.device)
            with torch.autocast("cuda"):
                # images = [Image.fromarray(np.random.randn(512, 512, 3))]
                images: List[Image.Image] = model(prompts, **kwargs)["sample"]
        out_q.put(images)


# class that handles multi-gpu models, mimicking original interface
class StableDiffusionMultiProcessing(object):
    def __init__(self, devices: List[int]) -> None:
        self.devices = devices
        self.n = len(devices)
        self._safety_checker = "dummy"
        self._scheduler = "PNDM"

    def _send_cmd(self, k1, k2, wait_ack=True):
        # send a cmd to all processes (put item in queue)
        for i in range(self.n):
            self.q.put((k1[i], k2[i]))
        # and wait for its completion
        res = []
        if wait_ack:
            for _ in range(self.n):
                res.append(self.outq.get())
        return res

    def __call__(self, prompts, **kwargs):
        # run inference on different processes, each handles a model on a different GPU (split load evenly)
        # print("prompts!", prompts)
        # FIXME when n_prompts < n, unused processes get an empty list as input, so we can always wait all processes
        prompts = [list(p) for p in np.array_split(prompts, self.n)]
        # request inference and block for result
        res = self._send_cmd(prompts, [kwargs] * self.n)
        # mimic interface
        return {"sample": [img for images in res for img in images]}

    @classmethod
    def from_pretrained(
        cls, devices: List[int], **kwargs
    ) -> "StableDiffusionMultiProcessing":
        # create communication i/o "channels"
        cls.q = mp.Queue()
        cls.outq = mp.Queue()
        # load nsfw filter CLIP configuration
        with open("./clip_config.pickle", "rb") as f:
            d = pickle.load(f)
        kwargs["clip_config"] = d
        # create models in their own process and move them to correspoding device
        cls._procs: List[mp.Process] = []
        for d in devices:
            p = mp.Process(
                target=cuda_inference_process,
                args=(d, cls.q, cls.outq, kwargs),
                daemon=False,
            )
            p.start()
            cls._procs.append(p)

        # wait until you move all models to their respective gpu (consistent with single mode)
        for _ in range(len(devices)):
            d = cls.outq.get()
            assert d
        # cls.pipes: List[StableDiffusionPipeline] = models
        return cls(devices)

    def __del__(self):
        # exit and join condition
        for _ in range(self.n):
            self.q.put(("quit", ""))
        for p in self._procs:
            p.join()

    def __len__(self):
        return self.n

    # mimic interface
    @property
    def safety_checker(self):
        return self._safety_checker

    @safety_checker.setter
    def safety_checker(self, value):
        # value=None->set filter, o/w set nsfw filter off
        nsfw_on = value is None
        # switch nsfw on, otherwise don't bother re-setting on processes
        if self.safety_checker == "dummy" and nsfw_on:
            self._safety_checker == "clip"
            self._send_cmd(
                ["safety_checker"] * self.n, ["clip"] * self.n, wait_ack=False
            )
        elif self.safety_checker == "clip" and not nsfw_on:
            self._safety_checker == "dummy"
            self._send_cmd(
                ["safety_checker"] * self.n, ["dummy"] * self.n, wait_ack=False
            )

    @property
    def scheduler(self):
        return self._scheduler

    @scheduler.setter
    def scheduler(self, value):
        # avoid re-setting if already set
        if self.scheduler == value or value not in schedulers:
            return
        self._scheduler = value
        self._send_cmd(["scheduler"] * self.n, [value] * self.n, wait_ack=False)


from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer

from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler


class StableDiffusionModelParallel(StableDiffusionPipeline):
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: Union[DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler],
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPFeatureExtractor,
        part_to_device: Dict[int, int],
    ):
        """
        Model can be split into 4 main components:
            - unet_encoder (downblocks to middle block)
            - unet_decoder (up_blocks+)
            - text_encoder
            - vae
        This class handles the components of a model that are split among multiple GPUs,
        taking care of moving tensors and Modules to the right devices: e.g.
        unet_encoder GPU_0 -> unet_decoder GPU_1 -> text_encoder GPU_1 -> vae GPU_0.
        Result is eventually moved back to CPU at the end of each foward call.
        """
        super().__init__(
            vae,
            text_encoder,
            tokenizer,
            unet,
            scheduler,
            safety_checker,
            feature_extractor,
        )
        # TODO do in `to()`?
        # move each component onto the specified device
        self.vae = ToGPUWrapper(self.vae, part_to_device[3])
        self.text_encoder = ToGPUWrapper(self.text_encoder, part_to_device[2])

        # move unet, requires a bit more work as it is chunked further into multiple parts
        # move encoder
        for layer in [
            "time_proj",
            "time_embedding",
            "conv_in",
            "down_blocks",
            "mid_block",
        ]:
            module = getattr(self.unet, layer)
            setattr(self.unet, layer, ToGPUWrapper(module, part_to_device[0]))

        # move decoder
        for layer in ["up_blocks", "conv_norm_out", "conv_act", "conv_out"]:
            module = getattr(self.unet, layer)
            setattr(self.unet, layer, ToGPUWrapper(module, part_to_device[1]))

        self.feature_extractor = dummy_extractor
        self.safety_checker = dummy_checker

    @property
    def device(self) -> torch.device:
        # NOTE this overrides super so we can handle all tensors devices manually, all `to(self.device)`
        # done in the forward pass become a no-op
        return None
