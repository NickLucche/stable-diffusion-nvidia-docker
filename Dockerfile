FROM ghcr.io/pytorch/pytorch:2.0.0-runtime
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
RUN pip install --upgrade diffusers[torch]==0.14.0
ENV GRADIO_SERVER_PORT=7860
ENV GRADIO_SERVER_NAME=0.0.0.0
EXPOSE 7860
ENTRYPOINT ["python3", "server.py"]