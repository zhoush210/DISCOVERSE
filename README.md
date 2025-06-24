# DISCOVERSE: Efficient Robot Simulation in Complex High-Fidelity Environments

<div align="center">

[![Paper](https://img.shields.io/badge/Paper-arXiv-red.svg)](https://air-discoverse.github.io/)
[![Website](https://img.shields.io/badge/Website-DISCOVERSE-blue.svg)](https://air-discoverse.github.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-Available-blue.svg)](doc/docker.md)

https://github.com/user-attachments/assets/78893813-d3fd-48a1-8bb4-5b0d87bf900f

*A unified, modular, open-source 3DGS-based simulation framework for Real2Sim2Real robot learning*

</div>

[中文文档](README_zh.md)

## 🌟 Key Features

DISCOVERSE represents a breakthrough in robotic simulation technology, offering unprecedented realism and efficiency for robot learning applications:

### 🎯 **High-Fidelity Real2Sim Generation**
- **Hierarchical scene reconstruction** for both background environments and interactive objects
- **Advanced laser-scanning integration** with LiDAR sensors for precise geometry capture
- **AI-powered 3D generation** using state-of-the-art generative models
- **Physically-based relighting** for photorealistic appearance matching
- **Mesh-Gaussian transfer** technology for seamless asset integration

### ⚡ **Exceptional Performance & Efficiency**
- **650 FPS rendering** for 5 cameras with RGB-D output (3× faster than ORBIT/Isaac Lab)
- **Massively parallel simulation** with GPU acceleration
- **Real-time 3D Gaussian Splatting** rendering engine
- **MuJoCo physics integration** for accurate contact dynamics
- **Optimized CUDA kernels** for maximum throughput

### 🔧 **Universal Compatibility & Flexibility**
- **Multi-format asset support**: 3DGS (.ply), Mesh (.obj/.stl), MJCF (.xml)
- **Diverse robot platforms**: Robotic arms, mobile manipulators, quadcopters, humanoids
- **Multiple sensor modalities**: RGB, Depth, LiDAR, IMU, tactile sensors
- **ROS2 integration** with seamless real-world deployment
- **Comprehensive randomization** including generative-based domain adaptation

### 🎓 **End-to-End Learning Pipeline**
- **Automated data collection** with 100× efficiency improvement over real-world
- **Multiple learning algorithms**: ACT, Diffusion Policy, RDT, and more
- **Zero-shot Sim2Real transfer** with state-of-the-art performance
- **Imitation learning workflows** from demonstration to deployment

## 🐳 Quick Start with Docker

The fastest way to get started with DISCOVERSE:

```bash
# Download pre-built Docker image
# Baidu Netdisk: https://pan.baidu.com/s/1mLC3Hz-m78Y6qFhurwb8VQ?pwd=xmp9

# Or build from source (recommended)
git clone https://github.com/TATP-233/DISCOVERSE.git --recursive
cd DISCOVERSE
docker build -t discoverse:latest .

# Run with GPU support
docker run -it --rm --gpus all \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v $(pwd):/workspace \
    discoverse:latest
```

For detailed Docker setup, see our [Docker deployment guide](doc/docker.md).

## 📦 Installation

### Prerequisites
- **Python 3.8+**
- **CUDA 11.8+** (for 3DGS rendering)
- **NVIDIA GPU** with 8GB+ VRAM (recommended)

### Basic Installation
```bash
# Clone repository with submodules
git clone https://github.com/TATP-233/DISCOVERSE.git --recursive
cd DISCOVERSE

# Install Python dependencies
pip install -r requirements.txt
pip install -e .
```

### Download Assets
Download model files from:
- [Baidu Netdisk](https://pan.baidu.com/s/1y4NdHDU7alCEmjC1ebtR8Q?pwd=bkca) 
- [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/d/0b92cdaeb58e414d85cc/)

Extract to the `models/` directory:
```
models/
├── meshes/          # Mesh geometries
├── textures/        # Material textures  
├── 3dgs/           # Gaussian Splatting models
│   ├── airbot_play/
│   ├── mmk2/
│   ├── objects/
│   └── scenes/
├── mjcf/           # MuJoCo scene descriptions
└── urdf/           # Robot descriptions
```

## 📷 Photorealistic Rendering Setup

For high-fidelity 3DGS rendering capabilities:

### 1. CUDA Installation
Install CUDA 11.8+ from [NVIDIA's official site](https://developer.nvidia.com/cuda-toolkit-archive).

### 2. 3DGS Dependencies
```bash
# Install Gaussian Splatting requirements
pip install -r requirements_gs.txt

# Build differential Gaussian rasterization
cd submodules/diff-gaussian-rasterization/
git checkout 8829d14

# Apply required patches
sed -i 's/(p_view.z <= 0.2f)/(p_view.z <= 0.01f)/' cuda_rasterizer/auxiliary.h
sed -i '361s/D += depths\[collected_id\[j\]\] \* alpha \* T;/if (depths[collected_id[j]] < 50.0f)\n        D += depths[collected_id[j]] * alpha * T;/' cuda_rasterizer/forward.cu

# Install
cd ../..
pip install submodules/diff-gaussian-rasterization
```

### 3. Model Visualization
View 3DGS models online using [SuperSplat](https://playcanvas.com/supersplat/editor) - simply drag and drop `.ply` files.

## 🔨 Real2Sim Pipeline

<img src="./assets/real2sim.jpg" alt="Real2Sim Pipeline"/>

DISCOVERSE features a comprehensive Real2Sim pipeline for creating digital twins of real environments. For detailed instructions, visit our [Real2Sim repository](https://github.com/GuangyuWang99/DISCOVERSE-Real2Sim).

## 💡 Usage Examples

### Basic Robot Simulation
```bash
# Launch Airbot Play robotic arm
python3 discoverse/envs/airbot_play_base.py

# Run manipulation tasks
python3 discoverse/examples/tasks_airbot_play/block_place.py
python3 discoverse/examples/tasks_airbot_play/coffeecup_place.py
python3 discoverse/examples/tasks_airbot_play/cuplid_cover.py
python3 discoverse/examples/tasks_airbot_play/drawer_open.py
```

https://github.com/user-attachments/assets/6d80119a-31e1-4ddf-9af5-ee28e949ea81

### Advanced Applications

#### Active SLAM
```bash
python3 discoverse/examples/active_slam/dummy_robot.py
```
<img src="./assets/active_slam.jpg" alt="Active SLAM" style="zoom: 33%;" />

#### Multi-Agent Coordination
```bash
python3 discoverse/examples/skyrover_on_rm2car/skyrover_and_rm2car.py
```
<img src="./assets/skyrover.png" alt="Multi-agent collaboration" style="zoom: 50%;" />

### Interactive Controls
- **'h'** - Show help menu
- **'F5'** - Reload MJCF scene
- **'r'** - Reset simulation state
- **'['/'']'** - Switch camera views
- **'Esc'** - Toggle free camera mode
- **'p'** - Print robot state information
- **'g'** - Toggle Gaussian rendering
- **'d'** - Toggle depth visualization

## 🎓 Learning & Training

### Imitation Learning Quickstart

DISCOVERSE provides complete workflows for data collection, training, and inference:

1. **Data Collection**: [Guide](./doc/imitation_learning/data.md)
2. **Model Training**: [Guide](./doc/imitation_learning/training.md) 
3. **Policy Inference**: [Guide](./doc/imitation_learning/inference.md)

### Supported Algorithms
- **ACT** (Action Chunking with Transformers)
- **Diffusion Policy** 
- **RDT** (Robotics Diffusion Transformer)
- **Custom algorithms** via extensible framework

### Domain Randomization
<div align="center">

https://github.com/user-attachments/assets/848db380-557c-469d-b274-2c9addf0b6bb

*Advanced image randomization powered by generative models*
</div>

DISCOVERSE incorporates state-of-the-art randomization techniques including:
- **Generative image synthesis** for diverse visual conditions
- **Physics parameter randomization** for robust policies
- **Lighting and material variations** for photorealistic adaptation

See our [randomization guide](doc/Randomain.md) for implementation details.

## 🏆 Performance Benchmarks

DISCOVERSE demonstrates superior Sim2Real transfer performance:

| Method | Close Laptop | Push Mouse | Pick Kiwi | **Average** |
|--------|-------------|------------|-----------|-------------|
| MuJoCo | 2% | 48% | 8% | 19.3% |
| SAPIEN | 0% | 24% | 0% | 8.0% |
| SplatSim | 56% | 68% | 26% | 50.0% |
| **DISCOVERSE** | **66%** | **74%** | **48%** | **62.7%** |
| **DISCOVERSE + Aug** | **86%** | **90%** | **76%** | **84.0%** |

*Zero-shot Sim2Real success rates using ACT policy*

## ⏩ Recent Updates

- **2025.01.13**: 🎉 DISCOVERSE open source release
- **2025.01.16**: 🐳 Docker support added
- **2025.01.14**: 🏁 [S2R2025 Competition](https://sim2real.net/track/track?nav=S2R2025) launched
- **2025.02.17**: 📈 Diffusion Policy baseline integration
- **2025.02.19**: 📡 Point cloud sensor support added

## 🤝 Community & Support

### Getting Help
- 📖 **Documentation**: Comprehensive guides in `/doc` directory
- 💬 **Issues**: Report bugs and request features via GitHub Issues
- 🔄 **Discussions**: Join community discussions for Q&A and collaboration

### Contributing
We welcome contributions! Please see our contributing guidelines and join our growing community of robotics researchers and developers.

<div align="center">
<img src="./assets/wechat.jpeg" alt="WeChat Community" style="zoom:50%;" />

*Join our WeChat community for updates and discussions*
</div>

## ❔ Troubleshooting

<details>
<summary><b>Common Installation Issues</b></summary>

**CUDA/PyTorch Version Mismatch**

`diff-gaussian-rasterization` fails to install due to mismatched pytorch and cuda versions: Please install the specified version of pytorch

```bash
# Install matching PyTorch version for your CUDA
pip install torch==2.2.1 torchvision==0.17.1 --index-url https://download.pytorch.org/whl/cu118
```

**Missing GLM Headers**

If you encounter error:`DISCOVERSE/submodules/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.cu:23:10: fatal error: glm/glm.hpp: no such file`
```bash
conda install -c conda-forge glm
export CPATH=$CONDA_PREFIX/include:$CPATH
pip install submodules/diff-gaussian-rasterization
```

**Server Deployment**

If you want to use it on a server, please specify the environment variable:
```bash
export MUJOCO_GL=egl  # For headless servers
```

**Graphics Driver Issues**

If you encounter errors:
```bash
GLFWError: (65542) b'GLX: No GLXFBConfigs returned'
GLFWError: (65545) b'GLX: Failed to find a suitable GLXFBConfig'
```
Check EGL vendor:
```bash
eglinfo | grep "EGL vendor"
```
If output includes:
libEGL warning: egl: failed to create dri2 screen
It indicates a conflict between Intel and NVIDIA drivers.
Check graphic driver prime:
```bash
prime-select query
```
If output is `on-demand`, switch to `nvidia` mode, then reboot or relogin!
```bash
sudo prime-select nvidia
```
Set the following environment variables to fix:
``` bash
export __NV_PRIME_RENDER_OFFLOAD=1
export __GLX_VENDOR_LIBRARY_NAME=nvidia
```
</details>

## ⚖️ License

DISCOVERSE is released under the [MIT License](LICENSE). See the license file for details.

## 📜 Citation

If you find DISCOVERSE helpful in your research, please consider citing our work:

```bibtex
@misc{discoverse2024,
      title={DISCOVERSE: Efficient Robot Simulation in Complex High-Fidelity Environments},
      author={Yufei Jia and Guangyu Wang and Yuhang Dong and Junzhe Wu and Yupei Zeng and Haizhou Ge and Kairui Ding and Zike Yan and Weibin Gu and Chuxuan Li and Ziming Wang and Yunjie Cheng and Wei Sui and Ruqi Huang and Guyue Zhou},
      url={https://air-discoverse.github.io/},
      year={2024}
}
```

---

<div align="center">

**DISCOVERSE** - *Bridging the gap between simulation and reality for next-generation robotics*

[🌐 Website](https://air-discoverse.github.io/) | [📄 Paper](https://air-discoverse.github.io/) | [🐳 Docker](doc/docker.md) | [📚 Documentation](doc/) | [🏆 Competition](https://sim2real.net/track/track?nav=S2R2025)

</div>