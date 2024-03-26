# Memes_safety_predictor

To build the container

```bash
docker build -t <CONTAINER_NAME> .
```

To run the container,
```bash
cat test/stdin.csv |
docker run --init \
        --attach "stdin" \
        --attach "stdout" \
        --attach "stderr" \
        --cpus 8 \
        --memory 6g \
        --memory-swap 0 \
        --ulimit nproc=1024 \
        --ulimit nofile=1024 \
        --read-only \
        --mount type=bind,source="$(pwd)"/test/images,target=/images,readonly \
        --mount type=tmpfs,destination=/tmp,tmpfs-size=5368709120,tmpfs-mode=1777 \
        --interactive \
        <CONTAINER_NAME> \
test/stdout.csv \
test/stderr.csv
```

Note that this is the same as running it locally with,
```bash
cat test/stdin_local.csv | \
    python3 main.py \
test/stdout.csv \
test/stderr.csv
```

But, you would need the weights to be installed locally on the directory.
To do this, simply run the `install_weights.sh` or run the code as followed,
```bash
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
wget -P "$PRETRAINED_WEIGHTS_DIR" https://github.com/miccunifi/ISSUES/releases/download/latest/hmc_text-inv-comb_best.ckpt

# Install FastText pretrained weights
wget -P "$PRETRAINED_WEIGHTS_DIR/fasttext/" https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin

# Installing hugging face weights
git lfs install
git clone https://huggingface.co/facebook/m2m100_418M "$PRETRAINED_WEIGHTS_DIR/facebook/m2m100_418M"
```