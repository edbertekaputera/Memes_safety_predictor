# This is a sample Dockerfile
# You may adapt this to meet your environment needs

FROM --platform=linux/x86_64 nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Install prerequisites
RUN apt-get update && \
	apt-get upgrade -y && \
    apt-get install -y curl && \
    apt-get install -y wget

# Install necessary libraries
RUN apt-get install -y python3 python3-pip tesseract-ocr tesseract-ocr-tam tesseract-ocr-chi-sim libgtk-3-dev \
	build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev \
    libavcodec-dev libavformat-dev libswscale-dev libv4l-dev \
    libxvidcore-dev libx264-dev libjpeg-dev libpng-dev libtiff-dev \
    gfortran openexr libatlas-base-dev \
    libtbbmalloc2 libtbb-dev libopenexr-dev \
    libgstreamer-plugins-base1.0-dev libgstreamer1.0-dev libgl1-mesa-dev libglib2.0-0 \
	unzip git git-lfs

# Set the working directory in the container
WORKDIR /app

# Copy your Python script into the container at /app
COPY . /app/

ENV HF_HOME=/tmp

# Install & download neccessary model weights required
RUN ./install_weights.sh
RUN mv /app/model.ckpt /app/resources/pretrained_weights/model.ckpt

# Install python requirements
RUN pip install -r requirements.txt

# # Cleanup unnecessary files
RUN apt-get clean && \
    apt-get purge -y curl \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

ENTRYPOINT ["python3", "main.py"]