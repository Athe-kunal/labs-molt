#!/bin/bash
# Incremental enroot rebuild of DeepEP inside the existing molt cu13 image (the cluster
# has enroot but NO docker). Bumps deep_ep to 42144303 (== 17cfb817 + DeepEP #638,
# HybridEP token-capacity padding; mirrors dockerfile/Dockerfile + Automodel d8d5ee36/#2678).
# Produces a NEW .sqsh and repoints the molt-cu13.sqsh symlink only on success — the current
# image is never mutated in place. Verified building deep_ep-1.2.1+4214430. See memory
# reference_deepep_container_build for the gotchas this encodes.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DEEPEP_COMMIT="${DEEPEP_COMMIT:-42144303752422ade37f24bca9e2dde12df70e09}"
SRC="${SRC:-images/lightning_rl-cu13-vllm023-sm100-deepep2614.sqsh}"
NEW="${NEW:-images/lightning_rl-cu13-vllm023-sm100-deepep${DEEPEP_COMMIT:0:8}.sqsh}"
LINK="${LINK:-images/molt-cu13.sqsh}"
NAME="${NAME:-lrl-deepep-bump}"
EROOT="${EROOT:-$HOME/.enroot_build}"

# gotcha 1: /raid (enroot default) is tiny/full -> point data/temp/cache at /lustre.
mkdir -p "$EROOT/data" "$EROOT/tmp" "$EROOT/cache"
export ENROOT_DATA_PATH="$EROOT/data" ENROOT_TEMP_PATH="$EROOT/tmp" ENROOT_CACHE_PATH="$EROOT/cache"
export ENROOT_MAX_PROCESSORS=8          # gotcha 2: nproc=188 -> EINTR on /lustre
export NVIDIA_VISIBLE_DEVICES=void      # gotcha 3: skip the GPU hook (CPU/nvcc build only)

test -f "$SRC"
echo "[build] $(date) creating sandbox '$NAME' from $SRC"
enroot remove -f "$NAME" 2>/dev/null || true
enroot create -n "$NAME" "$SRC"

echo "[build] $(date) rebuilding DeepEP @ $DEEPEP_COMMIT in-sandbox (via stdin; gotcha 4)"
DEEPEP_COMMIT="$DEEPEP_COMMIT" enroot start --root --rw -e DEEPEP_COMMIT "$NAME" bash -s <<'SANDBOX'
set -euo pipefail
# Ubuntu 24.04 system python is PEP-668 externally-managed (matches the Dockerfile's
# ENV PIP_BREAK_SYSTEM_PACKAGES=1); without this, pip refuses the system-site installs.
export PIP_BREAK_SYSTEM_PACKAGES=1 DEBIAN_FRONTEND=noninteractive
export RDMA_CORE_HOME=/opt/rdma-core/build
ARCH_LIB=$(dpkg-architecture -qDEB_HOST_MULTIARCH 2>/dev/null || echo x86_64-linux-gnu)
apt-get update && apt-get install -y --allow-change-held-packages rdma-core libibverbs-dev
test -f /usr/lib/${ARCH_LIB}/libmlx5.so || ln -sf /usr/lib/${ARCH_LIB}/libmlx5.so.1 /usr/lib/${ARCH_LIB}/libmlx5.so
mkdir -p ${RDMA_CORE_HOME}
ln -sfn /usr/include ${RDMA_CORE_HOME}/include
ln -sfn /usr/lib/${ARCH_LIB} ${RDMA_CORE_HOME}/lib
# deepep.patch (== dockerfile/deepep.patch) inlined: cccl include dir + pynvml->nvidia-ml-py.
cat > /opt/deepep.patch <<'PATCH'
diff --git a/setup.py b/setup.py
index 63ce332..4e13462 100644
--- a/setup.py
+++ b/setup.py
@@ -37,7 +37,7 @@ if __name__ == '__main__':
                  '-Wno-sign-compare', '-Wno-reorder', '-Wno-attributes']
     nvcc_flags = ['-O3', '-Xcompiler', '-O3']
     sources = ['csrc/deep_ep.cpp', 'csrc/kernels/runtime.cu', 'csrc/kernels/layout.cu', 'csrc/kernels/intranode.cu']
-    include_dirs = ['csrc/']
+    include_dirs = ['csrc/', '/usr/local/cuda/include/cccl/']
     library_dirs = []
     nvcc_dlink = []
     extra_link_args = []
@@ -278,7 +278,7 @@ if __name__ == '__main__':
         packages=setuptools.find_packages(
             include=['deep_ep']
         ),
         install_requires=[
-            'pynvml',
+            'nvidia-ml-py>=12.0.0',
         ],
         ext_modules=[
             get_extension_deep_ep_cpp(),
PATCH
rm -rf /opt/DeepEP
git clone https://github.com/deepseek-ai/DeepEP.git /opt/DeepEP
cd /opt/DeepEP
git fetch origin "$DEEPEP_COMMIT" && git checkout FETCH_HEAD
patch -p1 < /opt/deepep.patch
python -m pip install --no-cache-dir nvidia-nvshmem-cu13==3.6.5
NVSHMEM_LIB_PATH=$(pip show nvidia-nvshmem-cu13 | grep "Location:" | cut -d' ' -f2)/nvidia/nvshmem/lib
ln -sf ${NVSHMEM_LIB_PATH}/libnvshmem_host.so.3 ${NVSHMEM_LIB_PATH}/libnvshmem_host.so
apt-get install -y --no-install-recommends libnvidia-ml-dev
env RDMA_CORE_HOME=${RDMA_CORE_HOME} HYBRID_EP_MULTINODE=1 \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64/:${LD_LIBRARY_PATH:-} \
    TORCH_CUDA_ARCH_LIST="9.0 10.0" MAX_JOBS=32 \
    python -m pip install --no-cache-dir --no-build-isolation -v --force-reinstall --no-deps .
python -c "import importlib.metadata as m; print('[build] deep_ep dist:', m.version('deep_ep'))" \
  || echo "[build] NOTE: verify deep_ep CUDA import at runtime (needs a GPU)"
apt-get purge -y libnvidia-ml-dev && apt-get autoremove -y
rm -rf /var/lib/apt/lists/* /opt/deepep.patch && cd / && rm -rf /opt/DeepEP
echo "[build] in-sandbox DeepEP rebuild complete"
SANDBOX

echo "[build] $(date) exporting -> $NEW"
rm -f "$NEW"
enroot export -o "$NEW" "$NAME"
enroot remove -f "$NAME"
ln -sf "$(basename "$NEW")" "$LINK"
echo "[build] $(date) DONE. $(ls -la "$NEW"); symlink: $(ls -la "$LINK")"
