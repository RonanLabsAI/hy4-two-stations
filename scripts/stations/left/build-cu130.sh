set -e
# Rebuild 2026-11-20 16:3x PT: mount at the host path so RUNPATH resolves on the host; NCCL off to match the
# 13.2 build (all rungs use -sm layer, NCCL unused). First build (container path /w, NCCL on) failed G0 link check.
cd ~/l6-4h
[ -d llama.cpp-cu130 ] || git clone -q ~/hy4/llama.cpp llama.cpp-cu130
cd llama.cpp-cu130 && git checkout -q f8d4cc414 && git reset -q --hard f8d4cc414 && git clean -qfdx
git apply /models/Hy4-GGUF/0001-stq1_0-quant-and-cuda.patch
sha256sum /models/Hy4-GGUF/0001-stq1_0-quant-and-cuda.patch > ../patch.sha256
IMG=lmsysorg/sglang@sha256:352f1373e721c37d0849cf10ac9ec3c5b81dce76e148b9f271c1b59085eaba1d
docker image inspect $IMG >/dev/null 2>&1 || IMG=352f1373e721
P=$HOME/l6-4h
docker run --rm --name l64h-build-cu130 --user $(id -u):$(id -g) -e HOME=/tmp -v $P:$P -w $P/llama.cpp-cu130 $IMG bash -lc '
  nvcc --version | tail -2
  cmake -B build -DGGML_CUDA=ON -DGGML_CUDA_NCCL=OFF -DGGML_RPC=ON -DCMAKE_CUDA_ARCHITECTURES="103a;120a" -DGGML_NATIVE=OFF -DLLAMA_CURL=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_BUILD_HTML=OFF -DLLAMA_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release > ../cmake-cu130.log 2>&1
  cmake --build build -j 64 > ../make-cu130.log 2>&1
  ls build/bin | grep -E "^(llama-server|llama-perplexity|llama-completion|rpc-server|test-backend-ops)$"'
echo BUILD_CU130_OK
