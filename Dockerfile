FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
# TODO this version must be fixed as soon as it is released!
RUN apt update && apt install -y git && pip install --upgrade git+https://github.com/huggingface/diffusers.git
ENV GRADIO_SERVER_PORT=7860
ENV GRADIO_SERVER_NAME=0.0.0.0
EXPOSE 7860
ENTRYPOINT ["python3", "server.py"]