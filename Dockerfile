# 使用NVIDIA CUDA基础镜像
FROM nvidia/cuda:12.1.0-devel-ubuntu20.04

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive
ENV MUJOCO_GL=egl
ENV PYTHONPATH=/workspace/DISCOVERSE

# 添加deadsnakes PPA来安装Python 3.9
RUN apt-get update && apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa

# 安装系统依赖和Python 3.9
RUN apt-get update && apt-get install -y \
    git \
    curl \
    libgl1-mesa-dev \
    libglib2.0-0 \
    ffmpeg \
    libsm6 \
    libxext6 \
    python3.9 \
    python3.9-dev \
    python3.9-distutils \
    python3.9-venv \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# 确保使用Python 3.9
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1

# 安装pip
RUN curl -sS https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3 get-pip.py --force-reinstall && \
    rm get-pip.py

# 验证pip安装
RUN python3 -m pip --version

# 创建工作目录
WORKDIR /workspace

# 安装基础Python包
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel

# 安装PyTorch相关依赖
RUN python3 -m pip install --no-cache-dir \
    torch==2.2.1+cu121 \
    torchvision==0.17.1+cu121 \
    -f https://download.pytorch.org/whl/torch_stable.html

# 安装基础依赖
RUN python3 -m pip install --no-cache-dir \
    numpy \
    scipy \
    mediapy \
    opencv-python \
    mujoco

# 安装3DGS相关依赖
RUN python3 -m pip install --no-cache-dir \
    plyfile \
    PyGlm \
    torch>=2.0.0 \
    torchvision>=0.14.0

# 设置CUDA架构（支持RTX 30/40系列）
ENV TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6 8.9 9.0+PTX"

# 复制子模块代码
COPY submodules/diff-gaussian-rasterization /workspace/submodules/diff-gaussian-rasterization

# 安装diff-gaussian-rasterization
WORKDIR /workspace/submodules/diff-gaussian-rasterization
RUN pip install .

# 回到工作目录
WORKDIR /workspace

# 创建版本检查脚本
COPY <<'EOF' /usr/local/bin/check-versions
#!/bin/bash
echo "=== 系统环境检查 ==="
echo "Python版本："
python3 --version
echo -e "\nCUDA版本："
nvcc --version
echo -e "\nGPU信息："
nvidia-smi
echo -e "\n=== Python包版本检查 ==="
python3 -c "import torch; print(f'\nPyTorch信息：\n- PyTorch版本：{torch.__version__}\n- CUDA是否可用：{torch.cuda.is_available()}\n- CUDA版本：{torch.version.cuda}\n- 当前设备：{torch.cuda.get_device_name(0) if torch.cuda.is_available() else "无GPU"}')"
python3 -c "import torchvision; print(f'Torchvision版本：{torchvision.__version__}')"
python3 -c "import numpy; print(f'Numpy版本：{numpy.__version__}')"
python3 -c "import cv2; print(f'OpenCV版本：{cv2.__version__}')"
python3 -c "import mujoco; print(f'Mujoco版本：{mujoco.__version__}')"
EOF

RUN chmod +x /usr/local/bin/check-versions 

# 添加健康检查
HEALTHCHECK CMD python3 -c "import discoverse, torch, numpy, cv2, mujoco; print('All dependencies installed successfully')" 