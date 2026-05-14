#!/bin/bash
# Source this script when entering the cluster job

export HF_HOME=/scratch/rozkosz/.cache/huggingface
export HF_TOKEN="$HF_TOKEN"
export PATH=$HOME/.local/bin:$PATH
export USER=rozkosz
export LOGNAME=rozkosz
export HOME=/home/rozkosz

mkdir -p $HF_HOME

pip install transformers accelerate huggingface_hub --quiet
