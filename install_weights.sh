#!/bin/bash

# Download resources
wget https://github.com/miccunifi/ISSUES/releases/download/latest/resources.zip
unzip ./resources.zip
rm resources.zip

# remove unnecessary folders
rm -r ./resources/datasets
rmdir ./resources/pretrained_models
rm -r ./resources/pretrained_weights/harmeme

export PRETRAINED_WEIGHTS_DIR="./resources/pretrained_weights"

# Install CLIP model weights
wget -P "$PRETRAINED_WEIGHTS_DIR/clip/" https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt

# Install ISSUES pretrained weights
# wget -P "$PRETRAINED_WEIGHTS_DIR" https://github.com/miccunifi/ISSUES/releases/download/latest/hmc_text-inv-comb_best.ckpt

# Install FastText pretrained weights
wget -P "$PRETRAINED_WEIGHTS_DIR/fasttext/" https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin

# Installing hugging face weights
git lfs install
git clone https://huggingface.co/facebook/m2m100_1.2B "$PRETRAINED_WEIGHTS_DIR/facebook/m2m100_1.2B"
rm -rf "$PRETRAINED_WEIGHTS_DIR/facebook/m2m100_1.2B/.git"
rm -rf "$PRETRAINED_WEIGHTS_DIR/facebook/m2m100_1.2B/rust_model.ot"