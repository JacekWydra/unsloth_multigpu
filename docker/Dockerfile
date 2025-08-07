FROM pytorch/pytorch:2.7.0-cuda12.6-cudnn9-runtime 

WORKDIR /unsloth_multigpu

COPY requirements.txt requirements.txt

RUN apt update && apt install -y build-essential
RUN pip install --extra-index-url -r requirements.txt --no-cache-dir

COPY . .
