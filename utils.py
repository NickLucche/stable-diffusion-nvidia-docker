from typing import Tuple, List
import torch
import torch.nn as nn
from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline
import numpy as np
from PIL import Image
import torch.distributed as dist
import torch.multiprocessing as mp


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
    rank: int, device_id: int, in_q: mp.Queue, out_q: mp.Queue, model_kwargs
) -> StableDiffusionPipeline:
    # code executed by the torch.multiprocessing process, handling inference on device `device_id`
    device_id = device_id[rank]
    try:
        print(f"Creating and moving model to {device_id}..")
        model: StableDiffusionPipeline = StableDiffusionPipeline.from_pretrained(
            **model_kwargs
        ).to(f"cuda:{device_id}")
        out_q.put_nowait(True)
    except:
        out_q.put_nowait(False)
        return
    while True:
        prompts, kwargs = in_q.get()
        if type(prompts) is not list:
            # special command
            if prompts == "quit":
                break
            else:
                # set attribute cmd
                setattr(model, prompts, kwargs)
        if not len(prompts):
            images = []
        else:
            print("Inference", prompts, kwargs, model.device)
            images: List[Image.Image] = model(prompts, **kwargs)["sample"]
        out_q.put_nowait(images)


class StableDiffusionMultiProcessing(object):
    def __init__(self, devices: List[int]) -> None:
        self.devices = devices
        self.n = len(devices)

    def _send_cmd(self, k1, k2):
        # send a cmd to all processes (put item in queue)
        for i in range(self.n):
            self.q.put_nowait((k1[i], k2[i]))
        # and wait for its completion
        res = []
        for _ in range(self.n):
            res.append(self.outq.get())
        return res

    def __call__(self, prompts, **kwargs):
        # run inference on different processes, each handles a model on a different GPU (split load evenly)
        print("prompts!", prompts)
        prompts = [list(p) for p in np.array_split(prompts, self.n)]
        # request inference and block for result
        res = self._send_cmd(prompts, [kwargs] * self.n)
        # mimic interface
        return {"sample": np.concatenate(res)}

    @classmethod
    def from_pretrained(cls, devices: List[int], **kwargs):
        # create communication i/o "channels"
        cls.q = mp.Queue()
        cls.outq = mp.Queue()
        # create models in their own process and move them to correspoding device
        cls.ctx: mp.ProcessContext = mp.spawn(
            cuda_inference_process,
            args=(devices, cls.q, cls.outq, kwargs,),
            nprocs=len(devices),
            join=False,
            daemon=False,
        )

        # wait until you move all models to their respective gpu (consistent with single mode)
        for _ in range(len(devices)):
            d = cls.outq.get()
            assert d
        # cls.pipes: List[StableDiffusionPipeline] = models
        return cls(devices)

    def __del__(self):
        # exit and join condition
        for _ in range(self.n):
            self.q.put_nowait(("quit", ""))
        self.ctx.join()

    def __len__(self):
        return self.n

    # mimic interface
    # def safety_checker(self):
        # return self.pipes[0].safety_checker
