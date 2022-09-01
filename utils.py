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


def create_cuda_model(rank:int, device_id: int, in_q: mp.Queue, out_q: mp.Queue, kwargs) -> StableDiffusionPipeline:
    device_id = device_id[rank]
    try:
        model: StableDiffusionPipeline = StableDiffusionPipeline.from_pretrained(**kwargs).to(f"cuda:{device_id}")
        out_q.put_nowait(True)
    except:
        out_q.put_nowait(False)
    while True:
        prompt = in_q.get()
        print(model)
        a = np.random.randn(512, 512, 3)
        out_q.put_nowait(np.array([a.clip(-1, 1)]) )


# def model_forward(model: StableDiffusionPipeline, prompts: List[str], q: mp.Queue, kwargs):
#     if not len(prompts):
#         return []
#     print('model', prompts, kwargs)
#     images: List[Image.Image] = model(prompts, **kwargs)
#     q.put_nowait([Image.fromarray(np.random.randn(512, 512, 3))])
#     # print("OUT",images.keys())
    # return [im.copy() for im in images["sample"]]


class StableDiffusionContainer(object):
    def __init__(self, pipelines: List[StableDiffusionPipeline]) -> None:
        self.pipes: List[StableDiffusionPipeline] = pipelines
        print("pipes", self.pipes)

    # def __setattr__(self, name: str, val) -> None:
    #     for pipe in self.pipes:
    #         setattr(pipe, name, val)

    def __call__(self, prompts, **kwargs):
        # run inference on different processes, each handles a model on a different GPU (split load evenly)
        # TODO fix len(self) with some n param
        print("prompts!", prompts)
        prompts = [list(p) for p in np.array_split(prompts, 1)]
        for p in prompts:
            self.q.put_nowait(p)
        res = []
        while len(res) < len(prompts):
            print("OUTQ GET", len(res))
            res.append(self.outq.get())
        # res = self.pool.starmap(
        #     model_forward, [(p, prompts[i], kwargs) for i, p in enumerate(self.pipes)]
        # )
        # mimic interface
        return {"sample": np.concatenate(res)}

    @classmethod
    def from_pretrained(cls, devices: List[int], **kwargs):
        print("devices", len(devices))
        # create models and move them to correspoding device
        cls.q = mp.Queue()
        cls.outq = mp.Queue()
        # models = cls.pool.starmap(
        #     create_cuda_model, [(d_id, kwargs) for d_id in devices]
        # )
        cls.ctx:mp.ProcessContext = mp.spawn(create_cuda_model,
            args=(devices, cls.q, cls.outq, kwargs,),
            nprocs=len(devices),
            join=False, daemon=False)

        # wait until you move all models to their respective gpu (consistent with single mode)
        for _ in range(len(devices)):
            d = cls.outq.get()
            assert d
        # cls.pipes: List[StableDiffusionPipeline] = models
        return cls([])

    # def __del__(self):
        # TODO exit and join condition
        # self.pool.terminate()

    def __len__(self):
        return len(self.pipes)

    # mimic interface
    def safety_checker(self):
        return self.pipes[0].safety_checker
