import pickle
from typing import Tuple, List
import torch
import torch.nn as nn
from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline
import numpy as np
from PIL import Image
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers import CLIPConfig
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from schedulers import schedulers

def dummy_checker(images, **kwargs):
    # removes nsfw filter
    return images, False

def get_gpu_setting(env_var: str) -> Tuple[bool, List[int]]:
    if not torch.cuda.is_available():
        print("GPU not detected! Make sure you have a GPU to reduce inference time!")
        return False, []
    return True, [0]
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


def cuda_inference_process(
    device_id: int, in_q: mp.Queue, out_q: mp.Queue, model_kwargs
) -> StableDiffusionPipeline:
    # code executed by the torch.multiprocessing process, handling inference on device `device_id`
    try:
        print(f"Creating and moving model to cuda:{device_id} ({torch.cuda.get_device_name(device_id)})..")
        model: StableDiffusionPipeline = StableDiffusionPipeline.from_pretrained(
            **model_kwargs
        ).to(f"cuda:{device_id}")
        # disable nsfw filter by default
        model.safety_checker = dummy_checker
        # create nsfw clip filter so we can re-set it if needed
        safety_checker = StableDiffusionSafetyChecker(CLIPConfig(**model_kwargs.pop("clip_config")))
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
                model.safety_checker = safety_checker if kwargs == "clip" else dummy_checker
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
        prompts = [list(p) for p in np.array_split(prompts, self.n)]
        # request inference and block for result
        res = self._send_cmd(prompts, [kwargs] * self.n)
        print("RESULT", res)
        # mimic interface
        return {"sample": [img for images in res for img in images]}

    @classmethod
    def from_pretrained(cls, devices: List[int], **kwargs):
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
            p = mp.Process(target=cuda_inference_process, args=(d, cls.q, cls.outq, kwargs), daemon=False)
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
            self._send_cmd(["safety_checker"]*self.n, ["clip"]*self.n, wait_ack=False)
        elif self.safety_checker == "clip" and not nsfw_on:
            self._safety_checker == "dummy"
            self._send_cmd(["safety_checker"]*self.n, ["dummy"]*self.n, wait_ack=False)
            
    @property
    def scheduler(self):
        return self._scheduler

    @scheduler.setter
    def scheduler(self, value):
        # avoid re-setting if already set
        if self.scheduler == value or value not in schedulers:
            return
        self._scheduler = value        
        self._send_cmd(["scheduler"]*self.n, [value]*self.n, wait_ack=False)
