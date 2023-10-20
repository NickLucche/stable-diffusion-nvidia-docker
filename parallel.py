import numpy as np
from PIL import Image
import torch
import torch.multiprocessing as mp
import torch.nn as nn
from transformers import CLIPConfig
from schedulers import schedulers
import pickle
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)
from diffusers.pipelines.stable_diffusion import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
)
from utils import ToGPUWrapper, dummy_checker, dummy_extractor, remove_nsfw
from typing import Any, Dict, List, Optional, Union
import random
from sb import DiffusionModel

## Data Parallel: each process handles a copy of the model, executed on a different device ##
## +Model Parallel: model components are (potentially) scattered across different devices, each model handled by a process ##
def cuda_inference_process(
    worker_id: int,
    devices: List[torch.device],
    in_q: mp.Queue,
    out_q: mp.Queue,
    model_kwargs: Dict[Any, Any],
):
    """Code executed by the torch.multiprocessing process, handling inference on device `device_id`.
    It's a simple loop in which the worker pulls data from a shared input queue, and puts result
    into an output queue.
    """
    # wont work in pytorch 1.12 https://github.com/pytorch/pytorch/issues/80876
    # os.environ["CUDA_VISIBLE_DEVICES"]=str(device_id)
    mp_ass: Dict[int, int] = model_kwargs.pop("model_parallel_assignment", None)
    # each worker gets a different starting seed so they can be fixed and yet produce different results
    worker_seed = random.randint(0, int(2**32 - 1))
    # TODO replace with custom `StableDiffusion` model, single process == multi-process
    try:
        if mp_ass is None:
            # TODO should we make sure we're downloading the model only once?
            device_id = devices[worker_id]
            print(
                f"Creating and moving model to cuda:{device_id} ({torch.cuda.get_device_name(device_id)}).."
            )
            model: DiffusionModel = DiffusionModel.from_pretrained(**model_kwargs).to(
                f"cuda:{device_id}"
            )
        else:
            mp_ass = mp_ass[worker_id]
            print("Model parallel worker component assignment:", mp_ass)
            print(f"Creating and moving model parts to respective devices..")
            model = StableDiffusionModelParallel.from_pretrained(**model_kwargs).to(
                mp_ass
            )
            # TODO add for model parallel, but likely to refactor too
            # safety_checker, safety_extr = remove_nsfw(model)
            # create nsfw clip filter so we can re-set it if needed
            # safety_checker = StableDiffusionSafetyChecker(
            #     CLIPConfig(**model_kwargs.pop("clip_config"))
            # )
        out_q.put(True)
    except Exception as e:
        print(e)
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
            elif prompts == "safety_checker" and mp_ass is not None:
                # TODO
                raise NotImplementedError()
            elif prompts == "safety_checker":
                # safety checker is also be moved to GPU (it can cause crashes) when 'clip' is passed
                model.set_nsfw(kwargs == "clip")
            elif prompts == "scheduler":
                model.scheduler = kwargs
            elif prompts == "low_vram":
                model.enable_attention_slicing(kwargs)
            elif prompts == "reload_model":
                print(f"Worker {device_id}- Reloading model from disk..")
                model_path_or_id = kwargs
                model = model.reload_model(model_path_or_id)  # maintains device
                # model loading needs ack
                out_q.put(True)
            continue
        if not len(prompts):
            images = []
        else:
            # actual inference
            # print("Inference", prompts, kwargs, model.device)
            if kwargs.get("generator", None) is not None and kwargs["generator"] > 0:
                # NOTE different seed for each worker, but fixed!
                kwargs["generator"] = kwargs["generator"] + worker_seed
                # for repeatable results: tensor generated on cpu for model parallel
                # TODO unify model parallel interface, still using StableDiffusionPipeline
                if mp_ass is not None:
                    kwargs["generator"] = torch.Generator("cpu").manual_seed(
                        kwargs["generator"]
                    )
            else:
                kwargs.pop("generator", None)
            try:
                with torch.autocast("cuda"):
                    images: List[Image.Image] = model(prompt=prompts, **kwargs).images
            except Exception as e:
                print(f"[Model {device_id}] Error during inference:", e)
                # TODO proper error propagation to master process
                images = [
                    Image.fromarray(
                        np.zeros((kwargs["height"], kwargs["width"], 3), dtype=np.uint8)
                    )
                ]
        out_q.put(images)


# class that handles multi-gpu models, mimicking original interface
class StableDiffusionMultiProcessing(object):
    def __init__(
        self, n_procs: int, devices: List[int], model_id_or_path: str = ""
    ) -> None:
        self.devices = devices
        self.n = n_procs
        self._safety_checker = "dummy"
        self._scheduler = "PNDM"
        self._pipeline_type = "text"
        self._pipe_name = model_id_or_path

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

    def _send_cmd_to_all(self, k1, k2, wait_ack=True):
        return self._send_cmd([k1] * self.n, [k2] * self.n, wait_ack=wait_ack)

    def __call__(self, prompt, **kwargs):
        # run inference on different processes, each handles a model on a different GPU (split load evenly)
        # FIXME when n_prompts < n, unused processes get an empty list as input, so we can always wait all processes
        prompt = [list(p) for p in np.array_split(prompt, self.n)]
        # request inference and block for result
        res = self._send_cmd(prompt, [kwargs] * self.n)
        # mimic interface
        return {"images": [img for images in res for img in images]}

    @classmethod
    def from_pretrained(
        cls, n_processes: int, devices: List[int], **kwargs
    ) -> "StableDiffusionMultiProcessing":
        # create communication i/o "channels"
        cls.q = mp.Queue()
        cls.outq = mp.Queue()
        # load nsfw filter CLIP configuration
        # TODO still needed?
        with open("./clip_config.pickle", "rb") as f:
            d = pickle.load(f)
        kwargs["clip_config"] = d

        # create models in their own process and move them to correspoding device
        cls._procs: List[mp.Process] = []
        for i in range(n_processes):
            p = mp.Process(
                target=cuda_inference_process,
                args=(i, devices, cls.q, cls.outq, kwargs),
                daemon=False,
            )
            p.start()
            cls._procs.append(p)

        # wait until you move all models to their respective gpu (consistent with single mode)
        for _ in range(n_processes):
            d = cls.outq.get()
            assert d
        # cls.pipes: List[StableDiffusionPipeline] = models
        return cls(n_processes, devices, kwargs["pretrained_model_name_or_path"])

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
            self._send_cmd_to_all("safety_checker", "clip", wait_ack=False)
        elif self.safety_checker == "clip" and not nsfw_on:
            self._safety_checker == "dummy"
            self._send_cmd_to_all("safety_checker", "dummy", wait_ack=False)

    @property
    def scheduler(self):
        return self._scheduler

    @scheduler.setter
    def scheduler(self, value):
        # avoid re-setting if already set
        if self.scheduler == value or value not in schedulers:
            return
        self._scheduler = value
        self._send_cmd_to_all("scheduler", value, wait_ack=False)

    def enable_attention_slicing(self, value):
        self._send_cmd_to_all("low_vram", value, wait_ack=False)

    # def disable_attention_slicing(self):
    # self._send_cmd_to_all("low_vram", None, wait_ack=False)

    def change_pipeline_type(self, new_type: str):
        assert new_type in ["text", "img2img", "inpaint"]
        if new_type == self._pipeline_type:
            return
        self._pipeline_type = new_type
        self._send_cmd_to_all("pipeline_type", new_type, wait_ack=False)

    def reload_model(self, model_or_path: str):
        # reset all other options to default so they can be restored on next call
        self._send_cmd_to_all("reload_model", model_or_path, wait_ack=True)
        self._safety_checker = "dummy"
        self._scheduler = "PNDM"
        self._pipeline_type = "text"

    def set_nsfw(self, nsfw: bool):
        # this will avoid unnecessary inter-process calls and only set if needed
        if nsfw:
            self.safety_checker = None
        else:
            self.safety_checker = dummy_checker


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
        self._scheduler = self.scheduler
        # self._safety_checker = self.safety_checker

    def to(self, part_to_device: Dict[int, torch.device]):
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
            if type(module) is nn.ModuleList:
                mlist = nn.ModuleList(
                    [ToGPUWrapper(mod, part_to_device[0]) for mod in module]
                )
                setattr(self.unet, layer, mlist)
            else:
                setattr(self.unet, layer, ToGPUWrapper(module, part_to_device[0]))

        # move decoder
        for layer in ["up_blocks", "conv_norm_out", "conv_act", "conv_out"]:
            module = getattr(self.unet, layer)
            if type(module) is nn.ModuleList:
                mlist = nn.ModuleList(
                    [ToGPUWrapper(mod, part_to_device[1]) for mod in module]
                )
                setattr(self.unet, layer, mlist)
            else:
                setattr(self.unet, layer, ToGPUWrapper(module, part_to_device[1]))

        # need to wrap scheduler.step to move sampled noise to unet gpu
        self._wrap_scheduler_step()
        return self

    @property
    def device(self) -> torch.device:
        # NOTE this overrides super so we can handle all tensors devices manually, all `to(self.device)`
        # done in the forward pass become a no-op
        return None

    def _wrap_scheduler_step(self):
        prev_foo = self._scheduler.step

        def wrapper(x, i, sample: torch.Tensor, *args, **kwargs):
            sample = sample.to(self.unet.up_blocks.device)
            return prev_foo(x, i, sample, *args, **kwargs)

        self._scheduler.step = wrapper

    # override this interface for setting
    # @property
    # def safety_checker(self):
    #     return self._safety_checker

    # @safety_checker.setter
    # def safety_checker(self, value):
    #     # switch nsfw on, otherwise don't bother re-setting on processes
    #     if self.safety_checker is None and value is not None:
    #         self._safety_checker == value
    #     elif self.safety_checker is not None and value is None:
    #         self._safety_checker == None

    @property
    def scheduler(self):
        return self._scheduler

    @scheduler.setter
    def scheduler(
        self, value: Union[DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler]
    ):
        # if self.scheduler.__class__.__name__ == value.__class__.__name__:
        # return
        if not hasattr(self, "_scheduler"):
            # used during init phase
            self._scheduler = value
        else:
            self._scheduler = value
            self._wrap_scheduler_step()
