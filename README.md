A friend of mine working in art/design wanted to try out [Stable Diffusion](https://stability.ai/blog/stable-diffusion-public-release) on his own GPU-equipped PC, but he doesn't know much about coding, so I thought that baking a quick docker build was an easy way for me to help him out with his experiments. This repo holds the files that go into that build.

I also took the liberty of throwing in a simple web UI (made with gradio) to wrap the model. Peraphs we can evolve it a bit to offer a few more functionalities (see TODO).

# Requirements
 - OS: Ubuntu (tested on 20.04) or Windows
 - Nvidia GPU with at least 6GB vRAM (gtx 700 onward, please refer [here](https://docs.nvidia.com/deeplearning/cudnn/support-matrix/index.html)). Mind that the bigger the image size (or the number of images) you want to dream, the more memory you're gonna need. For reference, dreaming a 256x256 image should take up ~5gb, while a 512x512 around 7gb. 
 - Free Disk space > 2.8gb
 - Docker and Nvidia-docker.

# Installation

First of all, make sure to have docker installed in your machine 
The easiest way to try 
Note that in order to get your token, you must fist register (for free) to huggingface website and only then head to https://huggingface.co/settings/tokens.

My advice is that you start the container with

`docker run --name stable-diffusion --gpus all -it -e TOKEN=<YOUR_TOKEN> -p 7860:7860 stable-diffusion` 

the *first time* you run it, as it will download the model weights (can take a few minutes to do so).
Then you can simply do `docker stop stable-diffusion` to stop the container and `docker start stable-diffusion` to bring it back up.

Once the init phase is finished, you should be able to head to http://localhost:7860/ in your favorite browser (a message will pop-up in your terminal) and see something like this:

IMAGE

By default, the half-precision/fp16 model is loaded. This is the recommended approach if you're planning to run the model on a GPU with < 10GB of memory. To disable FP16 and run inference using single-precision set the environment variable FP16=0 as a docker run option, like so:

`docker run -e FP16=0 ...`  


# Samples


# TODO
 - [ ] allow other input modalities (images)
 - [ ] move model to specifiec GPU number (env variable)
 - [ ] test on older cudnn