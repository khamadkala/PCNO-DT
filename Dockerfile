FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y python3.11 python3-pip libglib2.0-0 && apt-get clean
WORKDIR /workspace
COPY requirements.txt .
RUN python3.11 -m pip install --no-cache-dir -r requirements.txt
COPY . .
RUN python3.11 -m pip install --no-deps .
ENTRYPOINT ["pcno-train"]

