#!/bin/bash

verbs=""
for path in /dev/infiniband/uverbs*; do
    verbs="$verbs --device=$path"
done

root_dir=$(realpath $(dirname $0)/..)


set -x
exec docker run --rm -it --name=dev-pplx-garden \
    -v $root_dir:/app \
    --init \
    --shm-size=32g \
    --ulimit=memlock=-1 \
    --ulimit=stack=67108864 \
    --gpus=all \
    $verbs \
    --device=/dev/gdrdrv \
    --cap-add=IPC_LOCK \
    --cap-add=SYS_ADMIN \
    --cap-add=SYS_PTRACE \
    --security-opt=seccomp=unconfined \
    --network host \
    pplx-garden-dev
