FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime
WORKDIR /app
COPY *.py requirements.txt *.pickle /app
RUN apt update && apt install -y git
RUN pip install -r requirements.txt
ENV GRADIO_SERVER_PORT=7860
ENV GRADIO_SERVER_NAME=0.0.0.0
EXPOSE 7860
ENTRYPOINT ["python3", "server.py"]