# Memes_safety_predictor

To build the container

```bash
docker build -t <CONTAINER_NAME> .
```

To run the container
```bash
cat test/stdin.csv |
docker run --init \
        --attach "stdin" \
        --attach "stdout" \
        --attach "stderr" \
        --cpus 4 \
        --memory 4g \
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