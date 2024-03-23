#!/usr/bin/env bash

set -ex

export LLAMA_CPP_LIB="/home/llama.cpp/libllama.so"
export CUDA_HOME=/usr/local/cuda/

# export LLAMACPP_VERBOSE="true"

#echo "Rebuilding LLAMA CPP"
#cd /home/llama.cpp/
#export LLAMA_CUBLAS=on
#make clean && make libllama.so
#pip uninstall llama-cpp-python
#CMAKE_ARGS="-DLLAMA_CLBLAST=on" FORCE_CMAKE=1 pip install llama-cpp-python --no-cache-dir

cd /home/novel-ai/dataset/

pip install -r requirements.txt
python3 generator_llm_2.py
