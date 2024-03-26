 # This is a sample Dockerfile
# You may adapt this to meet your environment needs

FROM --platform=linux/x86_64 ubuntu:22.04

# Install prerequisites
RUN apt-get update && \
	apt-get upgrade -y && \
    apt-get install -y curl && \
    apt-get install -y wget

# # Install Nvidia Driver
# RUN apt-get install -y nvidia-driver-535

# # Add the NVIDIA repository key
# RUN curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | apt-key add -

# # Add the NVIDIA repository
# RUN distribution=$(. /etc/os-release;echo $ID$VERSION_ID) && \
#     curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/nvidia-container-toolkit.list | \
#     tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# # Update package lists and install the NVIDIA Container Toolkit
# RUN apt-get install -y nvidia-container-toolkit

# # Install CUDNN
# RUN wget -O /tmp/cudnn-local-repo-ubuntu2204-8.9.7.29_1.0-1_amd64.deb https://storage.googleapis.com/hateful_meme_bucket/cudnn-local-repo-ubuntu2204-8.9.7.29_1.0-1_amd64.deb && \
#     apt install /tmp/cudnn-local-repo-ubuntu2204-8.9.7.29_1.0-1_amd64.deb

# RUN cp /var/cudnn-local-repo-ubuntu2204-8.9.7.29/cudnn-*.gpg /usr/share/keyrings/

# RUN apt install -y libcudnn8 ibcudnn8-dev libcudnn8-samples

# Install necessary libraries
RUN apt-get install -y python3 python3-pip tesseract-ocr libgtk-3-dev \
	build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev \
    libavcodec-dev libavformat-dev libswscale-dev libv4l-dev \
    libxvidcore-dev libx264-dev libjpeg-dev libpng-dev libtiff-dev \
    gfortran openexr libatlas-base-dev \
    libtbbmalloc2 libtbb-dev libopenexr-dev \
    libgstreamer-plugins-base1.0-dev libgstreamer1.0-dev libgl1-mesa-dev libglib2.0-0 \
	unzip git git-lfs

# # Cleanup unnecessary files
# RUN apt-get clean && \
# 	rm -f /tmp/cudnn-local-repo-ubuntu2204-8.9.7.29_1.0-1_amd64.deb \
#     && rm -rf /var/cudnn-local-repo-ubuntu2204-8.9.7.29 \
#     && rm -f /etc/apt/sources.list.d/nvidia-container-toolkit.list \
#     && apt-get purge -y curl \
#     && apt-get autoremove -y \
#     && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy your Python script into the container at /app
COPY . /app/

# Install python requirements
RUN pip install -r requirements.txt

# Install & download neccessary model weights required
RUN ./install_weights.sh

ENTRYPOINT ["python3", "main.py"]