# just a util file to gather the supported noise schedulers
from diffusers.schedulers import *
# setup noise schedulers
schedulers_names = [
    "EulerDiscrete",
    "DDIM",
    "PNDM",
    "K-LMS linear",
    "K-LMS scaled",
]
schedulers_cls = [
    EulerDiscreteScheduler,
    DDIMScheduler,
    PNDMScheduler,
    LMSDiscreteScheduler,
    LMSDiscreteScheduler,
]
# NOTE scheduler params are now loaded from pre-trained model
# schedulers_args = [
#     dict(),
#     {
#         "beta_end": 0.012,
#         "beta_schedule": "scaled_linear",
#         "beta_start": 0.00085,
#         "num_train_timesteps": 1000,
#         "skip_prk_steps": True,
#     },
#     dict(),
#     dict(beta_schedule="scaled_linear"),
# ]
# scheduler_name -> scheduler_class
schedulers = dict(zip(schedulers_names, schedulers_cls))