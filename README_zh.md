# DISCOVERSE: Efficient Robot Simulation in Complex High-Fidelity Environments

<div align="center">

[![论文](https://img.shields.io/badge/Paper-arXiv-red.svg)](https://air-discoverse.github.io/)
[![网站](https://img.shields.io/badge/Website-DISCOVERSE-blue.svg)](https://air-discoverse.github.io/)
[![许可证](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-Available-blue.svg)](doc/docker.md)

https://github.com/user-attachments/assets/78893813-d3fd-48a1-8bb4-5b0d87bf900f

*基于3DGS的统一、模块化、开源Real2Sim2Real机器人学习仿真框架*

</div>


<!-- echo "# Robot-SIm" >> README.md
git init
git add README.md
git commit -m "first commit"
git branch -M main
git remote add origin https://github.com/HEADQIANG/Robot-SIm.git
git push -u origin main -->

## 🌟 核心特性

DISCOVERSE代表了机器人仿真技术的突破，为机器人学习应用提供了前所未有的真实感和效率：

### 🎯 **高保真Real2Sim生成**
- **分层场景重建**：支持背景环境和交互物体的分层重建
- **先进激光扫描集成**：集成LiDAR传感器进行精确几何捕获
- **AI驱动3D生成**：使用最先进的生成模型
- **基于物理的重新光照**：实现逼真的外观匹配
- **网格-高斯转换技术**：实现无缝资产集成

### ⚡ **卓越性能与效率**
- **650 FPS渲染**：5个相机RGB-D输出（比ORBIT/Isaac Lab快3倍）
- **大规模并行仿真**：GPU加速
- **实时3D高斯散射**：渲染引擎
- **MuJoCo物理集成**：精确接触动力学
- **优化CUDA内核**：最大吞吐量

### 🔧 **通用兼容性与灵活性**
- **多格式资产支持**：3DGS (.ply), 网格 (.obj/.stl), MJCF (.xml)
- **多样化机器人平台**：机械臂、移动操作臂、四旋翼、人形机器人
- **多种传感器模态**：RGB、深度、LiDAR、IMU、触觉传感器
- **ROS2集成**：无缝真实世界部署
- **全面随机化**：包括基于生成的域适应

### 🎓 **端到端学习管道**
- **自动化数据收集**：比真实世界效率提升100倍
- **多种学习算法**：ACT、Diffusion Policy、RDT等
- **零样本Sim2Real迁移**：最先进性能
- **模仿学习工作流**：从演示到部署

## 🐳 Docker快速开始

开始使用DISCOVERSE的最快方式：

```bash
# 下载预构建Docker镜像
# 百度网盘：https://pan.baidu.com/s/1mLC3Hz-m78Y6qFhurwb8VQ?pwd=xmp9

# 或从源码构建（推荐）
git clone https://github.com/TATP-233/DISCOVERSE.git --recursive
cd DISCOVERSE
docker build -t discoverse:latest .

# 使用GPU支持运行
docker run -it --rm --gpus all \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v $(pwd):/workspace \
    discoverse:latest
```

详细的Docker设置请参见我们的[Docker部署指南](doc/docker.md)。

## 📦 安装

### 先决条件
- **Python 3.8+**
- **CUDA 11.8+**（用于3DGS渲染）
- **NVIDIA GPU**，推荐8GB+显存

### 基础安装
```bash
# 克隆仓库和子模块
git clone https://github.com/TATP-233/DISCOVERSE.git --recursive
cd DISCOVERSE

# 安装Python依赖
pip install -r requirements.txt
pip install -e .
```

### 下载资产文件
从以下地址下载模型文件：
- [百度网盘](https://pan.baidu.com/s/1y4NdHDU7alCEmjC1ebtR8Q?pwd=bkca) 
- [清华云盘](https://cloud.tsinghua.edu.cn/d/0b92cdaeb58e414d85cc/)

解压到`models/`目录：
```
models/
├── meshes/          # 网格几何
├── textures/        # 材质纹理  
├── 3dgs/           # 高斯散射模型
│   ├── airbot_play/
│   ├── mmk2/
│   ├── objects/
│   └── scenes/
├── mjcf/           # MuJoCo场景描述
└── urdf/           # 机器人描述
```

## 📷 逼真渲染设置

用于高保真3DGS渲染功能：

### 1. CUDA安装
从[NVIDIA官网](https://developer.nvidia.com/cuda-toolkit-archive)安装CUDA 11.8+。

### 2. 3DGS依赖
```bash
# 安装gaussian splatting依赖
pip install -r requirements_gs.txt

# 构建diff-gaussian-rasterization
cd submodules/diff-gaussian-rasterization/
git checkout 8829d14

# 应用必要补丁
sed -i 's/(p_view.z <= 0.2f)/(p_view.z <= 0.01f)/' cuda_rasterizer/auxiliary.h
sed -i '361s/D += depths\[collected_id\[j\]\] \* alpha \* T;/if (depths[collected_id[j]] < 50.0f)\n        D += depths[collected_id[j]] * alpha * T;/' cuda_rasterizer/forward.cu

# 安装
cd ../..
pip install submodules/diff-gaussian-rasterization
```

### 3. 模型可视化
使用[SuperSplat](https://playcanvas.com/supersplat/editor)在线查看3DGS模型 - 只需拖放`.ply`文件。

## 🔨 Real2Sim管道

<img src="./assets/real2sim.jpg" alt="Real2Sim管道"/>

DISCOVERSE具有全面的Real2Sim管道，用于创建真实环境的数字孪生。详细说明请访问我们的[Real2Sim仓库](https://github.com/GuangyuWang99/DISCOVERSE-Real2Sim)。

## 💡 使用示例

### 基础机器人仿真
```bash
# 启动Airbot Play机械臂
python3 discoverse/envs/airbot_play_base.py

# 运行操作任务
python3 discoverse/examples/tasks_airbot_play/block_place.py
python3 discoverse/examples/tasks_airbot_play/coffeecup_place.py
python3 discoverse/examples/tasks_airbot_play/cuplid_cover.py
python3 discoverse/examples/tasks_airbot_play/drawer_open.py
```

https://github.com/user-attachments/assets/6d80119a-31e1-4ddf-9af5-ee28e949ea81

### 高级应用

#### 主动SLAM
```bash
python3 discoverse/examples/active_slam/dummy_robot.py
```
<img src="./assets/active_slam.jpg" alt="主动SLAM" style="zoom: 33%;" />

#### 多智能体协作
```bash
python3 discoverse/examples/skyrover_on_rm2car/skyrover_and_rm2car.py
```
<img src="./assets/skyrover.png" alt="多智能体协作" style="zoom: 50%;" />

### 交互式控制
- **'h'** - 显示帮助菜单
- **'F5'** - 重新加载MJCF场景
- **'r'** - 重置仿真状态
- **'['/'']'** - 切换相机视角
- **'Esc'** - 切换自由相机模式
- **'p'** - 打印机器人状态信息
- **'g'** - 切换高斯渲染
- **'d'** - 切换深度可视化

## 🎓 学习与训练

### 模仿学习快速开始

DISCOVERSE提供数据收集、训练和推理的完整工作流：

1. **数据收集**：[指南](./doc/imitation_learning/data.md)
2. **模型训练**：[指南](./doc/imitation_learning/training.md) 
3. **策略推理**：[指南](./doc/imitation_learning/inference.md)

### 支持的算法
- **ACT**（Action Chunking with Transformers）
- **Diffusion Policy** 
- **RDT**（Robotics Diffusion Transformer）
- **自定义算法**通过可扩展框架

### 域随机化
<div align="center">

https://github.com/user-attachments/assets/848db380-557c-469d-b274-2c9addf0b6bb

*由生成模型驱动的高级图像随机化*
</div>

DISCOVERSE集成了最先进的随机化技术，包括：
- **生成式图像合成**用于多样化视觉条件
- **物理参数随机化**用于鲁棒策略
- **光照和材质变化**用于逼真适应

详细实现请参见我们的[随机化指南](doc/Randomain.md)。

## 🏆 性能基准

DISCOVERSE展示了卓越的Sim2Real迁移性能：

| 方法 | 关闭笔记本 | 推动鼠标 | 拿起猕猴桃 | **平均** |
|--------|-------------|------------|-----------|-------------|
| MuJoCo | 2% | 48% | 8% | 19.3% |
| SAPIEN | 0% | 24% | 0% | 8.0% |
| SplatSim | 56% | 68% | 26% | 50.0% |
| **DISCOVERSE** | **66%** | **74%** | **48%** | **62.7%** |
| **DISCOVERSE + Aug** | **86%** | **90%** | **76%** | **84.0%** |

*使用ACT策略的零样本Sim2Real成功率*

## ⏩ 最近更新

- **2025.01.13**：🎉 DISCOVERSE开源发布
- **2025.01.16**：🐳 添加Docker支持
- **2025.01.14**：🏁 [S2R2025竞赛](https://sim2real.net/track/track?nav=S2R2025)启动
- **2025.02.17**：📈 集成Diffusion Policy基线
- **2025.02.19**：📡 添加点云传感器支持

## 🤝 社区与支持

### 获取帮助
- 📖 **文档**：`/doc`目录中的全面指南
- 💬 **问题**：通过GitHub Issues报告错误和请求功能
- 🔄 **讨论**：加入社区讨论进行问答和协作

### 贡献
我们欢迎贡献！请查看我们的贡献指南，加入我们不断壮大的机器人研究者和开发者社区。

<div align="center">
<img src="./assets/wechat.jpeg" alt="微信社区" style="zoom:50%;" />

*加入我们的微信社区获取更新和讨论*
</div>

## ❔ 故障排除

<details>
<summary><b>常见安装问题</b></summary>

**CUDA/PyTorch版本不匹配**

`diff-gaussian-rasterization` fails to install due to mismatched pytorch and cuda versions: Please install the specified version of pytorch

```bash
# 为您的CUDA安装匹配的PyTorch版本
pip install torch==2.2.1 torchvision==0.17.1 --index-url https://download.pytorch.org/whl/cu118
```

**缺少GLM头文件**
如果遇到了:`DISCOVERSE/submodules/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.cu:23:10: fatal error: glm/glm.hpp: no such file`
```bash
conda install -c conda-forge glm
export CPATH=$CONDA_PREFIX/include:$CPATH
```

**服务器部署**
在服务器上部署时，指定环境变量：
```bash
export MUJOCO_GL=egl  # 用于无头服务器
```

**图形驱动问题**

如果遇到报错:
```bash
GLFWError: (65542) b'GLX: No GLXFBConfigs returned'
GLFWError: (65545) b'GLX: Failed to find a suitable GLXFBConfig'
```
检查 EGL vendor:
```bash
eglinfo | grep "EGL vendor"
```
如果输入包含以下内容:
libEGL warning: egl: failed to create dri2 screen
It indicates a conflict between Intel and NVIDIA drivers.
检查 优先图形驱动:
```bash
prime-select query
```
如果输出是 `on-demand`, 切换到 `nvidia` 模式, 重新启动电脑!
```bash
sudo prime-select nvidia
```
设置环境变量来修正问题:
``` bash
export __NV_PRIME_RENDER_OFFLOAD=1
export __GLX_VENDOR_LIBRARY_NAME=nvidia
```

</details>

## ⚖️ 许可证

DISCOVERSE在[MIT许可证](LICENSE)下发布。详细信息请参见许可证文件。

## 📜 引用

如果您发现DISCOVERSE对您的研究有帮助，请考虑引用我们的工作：

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

**DISCOVERSE** - *为下一代机器人技术弥合仿真与现实的差距*

[🌐 网站](https://air-discoverse.github.io/) | [📄 论文](https://air-discoverse.github.io/) | [🐳 Docker](doc/docker.md) | [📚 文档](doc/) | [🏆 竞赛](https://sim2real.net/track/track?nav=S2R2025)

</div> 