#!/usr/bin/env bash

#export WEBDAV_HOSTNAME=""
#export WEBDAV_USER=""
#export WEBDAV_PASS=""

PIP=$(which pip3)

${PIP} install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

major_version=$(python3 -c "import torch
major_version, _ = torch.cuda.get_device_capability()
print(major_version)")

# Compare and call a different command if major_version > 8
if (( major_version >= 8 )); then
    # Use this for new GPUs like Ampere, Hopper GPUs (RTX 30xx, RTX 40xx, A100, H100, L40)
    ${PIP} install --no-deps packaging ninja einops flash-attn xformers trl peft accelerate bitsandbytes
else
    # Use this for older GPUs (V100, Tesla T4, RTX 20xx)
    ${PIP} install --no-deps xformers trl peft accelerate bitsandbytes
fi

${PIP} install datasets huggingface_hub pycryptodomex webdavclient3 tqdm zstandard wandb


wandb login

python3 train.py

