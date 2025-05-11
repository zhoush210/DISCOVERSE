# Domain Randomization (域随机化) 用于 Sim-to-Real 迁移

## 1. Domain Randomization 简介

### 1.1 什么是 Domain Randomization？
Domain Randomization (DR) 是一种主要应用于机器人学和强化学习领域的技术，旨在改善在仿真环境 (sim) 中学习到的策略迁移到真实世界 (real) 的效果。其核心思想是在大量视觉和物理属性显著不同的仿真环境中训练模型。通过在训练过程中让模型接触多种多样的变化，使其能够适应仿真与现实之间的差异，从而弥合"sim-to-real gap"（仿真到现实的鸿沟）。

DR 中的变化可以包括：
- **视觉域 (Visual Domain)**: 光照条件、纹理、颜色、相机位置、干扰物等。
- **动力学域 (Dynamics Domain)**: 物理参数，如质量、摩擦力、电机扭矩、传感器噪声等。

### 1.2 为什么要使用 Domain Randomization？
- **弥合 Sim-to-Real 鸿沟**: 仿真是对真实世界的不完美近似。DR 通过不让模型过拟合于单一特定的仿真环境，帮助模型更好地泛化。
- **减少人工投入**: 创建高度逼真的仿真既耗时又昂贵。DR 使得即便使用照片真实感较低但变化更丰富的仿真也能进行有效学习。
- **数据增强**: DR 是一种强大的数据增强形式，特别是对于需要大量多样化训练数据的深度学习模型。
- **提高鲁棒性**: 使用 DR 训练的策略通常对真实世界中的意外变化或噪声更具鲁棒性。

## 2. `discoverse/randomain` 工具概述

`discoverse/randomain` 工具包位于 `discoverse/randomain` 目录下，提供了一个将域随机化应用于视频序列的流程，这些视频序列通常从机器人仿真环境中捕获。它专注于**视觉域随机化**，利用强大的生成模型逐帧改变场景的外观。

**主要功能包括：**
- **数据采样**: 从仿真中捕获同步的 RGB、深度和分割掩码信息。
- **生成式场景修改**: 使用文本提示和深度信息，结合最先进的生成模型（通过 ComfyUI），以变化的视觉外观重新渲染视频帧。
- **用于时间一致性的光流**:采用光流技术生成中间帧，确保更平滑的过渡并减少计算负荷。

其目标是将"纯净"的仿真视频转化为多个视觉上截然不同的版本，这些版本随后可用于训练更鲁棒的感知模型或策略。所有生成的数据和中间样本通常存储在 `data/randomain` 目录下。

## 3. 核心技术

`discoverse/randomain` 流水线集成了几种先进技术：

### 3.1 基于 ComfyUI 的生成模型

[ComfyUI](https://github.com/comfyanonymous/ComfyUI) 是一个功能强大且模块化的基于节点的图形用户界面，专用于 Stable Diffusion。本工具包利用 ComfyUI作为后端来驱动生成模型。

- **Stable Diffusion XL (SDXL) Turbo**:
    - **原理**: SDXL Turbo 是 SDXL 的蒸馏版本，专为从文本提示快速生成高质量图像而设计。它通过减少通常所需的扩散步骤数量来实现这种速度。`sd_xl_turbo_1.0_fp16.safetensors` 模型是一个16位浮点版本，在性能和精度之间取得了平衡。
    - **作用**: 作为主要的文本到图像生成引擎，根据输入提示创建新颖的视觉外观。

- **ControlNet (用于深度信息)**:
    - **原理**: ControlNet 是一种神经网络结构，可为像 SDXL 这样的预训练扩散模型添加额外条件。`controlnet_depth_sdxl_1.0` 模型专门用于根据深度图调节图像生成。这意味着它可以生成与给定3D场景结构（由深度图定义）一致的图像，同时根据文本提示改变纹理、风格和对象。
    - **作用**: 对于保持原始场景的几何完整性至关重要。通过使用仿真中的深度图，ControlNet 确保随机化后的场景元素遵循原始对象的放置、形状和整体3D布局。

- **VAE (Variational Autoencoder - 变分自编码器)**:
    - **原理**: VAE 用于潜在扩散模型（如 Stable Diffusion）中，将图像编码为低维潜在表示，并将其解码回像素空间。`sdxl_vae.safetensors` 模型针对 SDXL 进行了优化。
    - **作用**: 处理图像数据的压缩和解压缩，使扩散过程能够在计算成本更低的、更易于管理的潜在空间中进行。

### 3.2 光流 (Optical Flow)

光流是指视觉场景中物体、表面和边缘因观察者（例如相机）与场景之间的相对运动而产生的表观运动模式。

- **在 `randomain` 中的用途**: 当生成随机化帧序列时，为每一帧都运行完整的生成模型计算成本可能很高。如果指定了大于1的 `flow_interval`（光流间隔），则生成模型仅在关键帧上运行。这些关键帧之间的帧随后使用光流进行插值。这样可以在尝试保持时间平滑性的同时加快整个过程。

- **支持的方法**:
    - **Farneback 方法 (`rgb`)**:
        - **原理**: 一种经典的稠密光流（即为每个像素计算光流矢量）算法，基于 Gunnar Farneback 的多项式展开。它分析两个连续帧之间的图像强度变化。
        - **优点**: 相对轻量级，不需要GPU或预训练模型。
        - **缺点**: 在大位移、遮挡或复杂纹理情况下，其准确性可能低于深度学习方法。
    - **RAFT (Recurrent All-Pairs Field Transforms) (`raft`)**:
        - **原理**: 一种最先进的光流深度学习模型。它使用循环架构，通过查询从所有像素对构建的关联体积中的特征来迭代更新光流场。
        - **优点**: 通常更准确、更鲁棒，尤其适用于复杂场景和较大运动。
        - **缺点**: 需要预训练模型（权重通过 Google Drive 链接提供），并且计算密集，通常受益于GPU加速。

## 4. 设置与安装

### 4.1 生成模型设置 (ComfyUI)

1.  **克隆 ComfyUI**:
    ```bash
    # 导航到您存放子模块或外部工具的首选目录
    # 例如，如果您的项目根目录是 DISCOVERSE:
    # mkdir -p submodules
    # cd submodules
    git clone https://github.com/comfyanonymous/ComfyUI
    ```

2.  **安装 ComfyUI 依赖**:
    ```bash
    cd ComfyUI
    pip install -r requirements.txt
    cd ../.. # 返回项目根目录或相关目录
    ```

### 4.2 模型部署

`randomain` 工具期望特定的生成模型放置在特定的目录结构中。假设您的 ComfyUI 位于 `submodules/ComfyUI`，则模型应放置在 `submodules/ComfyUI/models` 内。

- **Checkpoints (主模型)**:
    - 模型: `sd_xl_turbo_1.0_fp16.safetensors`
    - 下载地址: [Hugging Face - stabilityai/sdxl-turbo](https://huggingface.co/stabilityai/sdxl-turbo/blob/main/sd_xl_turbo_1.0_fp16.safetensors)
    - 部署路径: `submodules/ComfyUI/models/checkpoints/sd_xl_turbo_1.0_fp16.safetensors`

- **ControlNet (深度模型)**:
    - 模型: `controlnet_depth_sdxl_1.0.safetensors` (或将 `diffusion_pytorch_model.safetensors` 重命名)
    - 下载地址: [Hugging Face - diffusers/controlnet-depth-sdxl-1.0](https://huggingface.co/diffusers/controlnet-depth-sdxl-1.0/blob/main/diffusion_pytorch_model.safetensors) (如果脚本期望确切的文件名，您可能需要将此文件重命名为 `controlnet_depth_sdxl_1.0.safetensors`，或者在 ComfyUI 工作流中调整路径)。原始文档示例显示为 `controlnet_depth_sdxl_1.0.safetensors`。
    - 部署路径: `submodules/ComfyUI/models/controlnet/controlnet_depth_sdxl_1.0.safetensors`

- **VAE (变分自编码器)**:
    - 模型: `sdxl_vae.safetensors` (或将 `diffusion_pytorch_model.safetensors` 重命名)
    - 下载地址: [Hugging Face - stabilityai/sdxl-vae](https://huggingface.co/stabilityai/sdxl-vae/blob/main/diffusion_pytorch_model.safetensors) (与 ControlNet 类似，您可能需要将此文件重命名为 `sdxl_vae.safetensors`)。
    - 部署路径: `submodules/ComfyUI/models/vae/sdxl_vae.safetensors`

**`submodules/ComfyUI` 内部的目录结构应如下所示：**
```
ComfyUI/
├── models/
│   ├── checkpoints/
│   │   └── sd_xl_turbo_1.0_fp16.safetensors
│   ├── controlnet/
│   │   └── controlnet_depth_sdxl_1.0.safetensors
│   └── vae/
│       └── sdxl_vae.safetensors
├── ... (其他 ComfyUI 文件和文件夹)
```
原始文档提到在 `randomain` 的根目录下有一个 `models` 文件夹，其中包含 `extra_model_paths.yaml`。这似乎是 ComfyUI 用于指定备用模型位置的功能。如果您正在使用此 `extra_model_paths.yaml` (例如 `discoverse/randomain/models/extra_model_paths.yaml`)，请确保它正确指向您存储模型的任何位置（如果不在默认的 ComfyUI 路径中）。原始文档中的示例：
```
randomain/ # 这似乎指的是 discoverse/randomain
├── models/
│   ├── checkpoints/
│   │   └── sd_xl_turbo_1.0_fp16.safetensors
│   ├── controlnet/
│   │   └── controlnet_depth_sdxl_1.0.safetensors
│   ├── extra_model_paths.yaml  # ComfyUI 会查找此文件
│   └── vae/
│       └── sdxl_vae.safetensors
```
如果使用此结构，`extra_model_paths.yaml` 可能包含如下路径：
```yaml
# extra_model_paths.yaml 示例内容
# 这会告诉 ComfyUI 在相对于此 yaml 文件位置的这些特定子目录中查找，或使用绝对路径。
checkpoints: ./checkpoints
controlnet: ./controlnet
vae: ./vae
# 或者，如果模型在其他地方：
# checkpoints: /path/to/my/global/checkpoints
```
关键是 ComfyUI 必须能够找到这些模型。

### 4.3 环境配置 (路径链接)

为了使 `discoverse/randomain` 脚本能够正确地与 ComfyUI 及其模型交互，您需要设置环境变量：

1.  **`PYTHONPATH`**: 添加 ComfyUI 目录的路径，以便 Python 可以导入其模块。
    ```bash
    export PYTHONPATH=/path/to/your/ComfyUI:$PYTHONPATH
    # 例如，如果 ComfyUI 位于 submodules/ComfyUI:
    # export PYTHONPATH=$(pwd)/submodules/ComfyUI:$PYTHONPATH 
    # (确保从您的项目根目录运行此命令或使用绝对路径)
    ```

2.  **`COMFYUI_CONFIG_PATH`**: 如果您使用 `extra_model_paths.yaml` 来告诉 ComfyUI 您的模型在哪里（特别是当它们不在默认的 `ComfyUI/models` 子目录中时），您需要将 ComfyUI 指向此配置文件。
    ```bash
    export COMFYUI_CONFIG_PATH=/path/to/your/randomain/models/extra_model_paths.yaml
    # 例如，如果它位于 discoverse/randomain/models/extra_model_paths.yaml:
    # export COMFYUI_CONFIG_PATH=$(pwd)/discoverse/randomain/models/extra_model_paths.yaml
    ```
    如果您将所有模型直接放入标准的 `ComfyUI/models/...` 子目录中，您可能不严格需要 `extra_model_paths.yaml` 或 `COMFYUI_CONFIG_PATH`，因为 ComfyUI 会检查默认位置。但是，原始文档暗示了它的使用。

**建议**: 将这些 `export` 命令添加到您的 shell 配置文件（例如 `~/.bashrc`, `~/.zshrc`）或项目的特定环境激活脚本中，以避免每次手动设置。

### 4.4 光流模型设置 (RAFT)

-   如果您选择使用 `RAFT` 方法进行光流处理 (`flow_method='raft'`):
    -   下载预训练的 RAFT 权重。原始文档指向一个 Google Drive 文件夹：[RAFT Models](https://drive.google.com/drive/folders/1sWDsfuZ3Up38EUQt7-JDTT1HcGHuJgvT)。您可能需要 `raft-things.pth`（用于通用光流）或类似模型。
    -   将下载的 `.pth` 文件放置到 `discoverse/randomain/models/flow/` 目录下。例如：
        ```
        discoverse/randomain/
        ├── models/
        │   ├── flow/
        │   │   └── raft-things.pth # 或您下载的具体模型名称
        ```
-   如果使用 `Farneback` 方法 (`flow_method='rgb'`)，则无需下载额外的模型，因为它是一个 OpenCV 算法。
-   要集成其他光流方法，您需要在 `discoverse/randomain` 代码库中实现与 `FlowCompute/raft` 类似的接口。

## 5. 使用流程

生成随机化数据的过程包括三个主要阶段：

### 5.1 阶段 1: 样本采集

此阶段涉及运行您现有的仿真（例如，机械臂执行任务）并逐帧捕获必要的视觉数据。`discoverse.randomain.utils.SampleforDR` 类专为此设计。

1.  **实例化 `SampleforDR`**:
    ```python
    from discoverse.randomain.utils import SampleforDR

    # 示例配置 (cfg 来自您项目的配置系统)
    # objs = ['block_green', 'bowl_pink'] # 关键可操作对象列表
    # robot_parts = ['panda_link0', 'panda_link1', ..., 'panda_hand'] # 用于分割的机器人链接名称列表
    # cam_ids = cfg.obs_rgb_cam_id # 要记录的相机ID列表或单个ID
    # save_dir = "data/randomain/trajectory_000" # 此轨迹样本的基础目录
    # fps = 30 
    # max_vis_dis = 1.0 # 深度归一化的最大可视化距离 (米)

    samples = SampleforDR(
        objs=objs,
        robot_parts=robot_parts,
        cam_ids=cam_ids,
        save_dir=save_dir,
        fps=cfg.render_set.get('fps', fps), # 从配置中获取 FPS 或使用默认值
        max_vis_dis=max_vis_dis
    )
    ```
    -   **`objs`**: 字符串列表，每个字符串是场景中关键动态对象的唯一名称（例如，可操作的块、工具、目标）。这些对象将保存其单独的分割掩码。
    -   **`robot_parts`**: 字符串列表，表示机器人链接/部件的名称。这些将被组合成一个单一的"机器人"掩码。
    -   **`cam_ids`**: 用于记录的相机标识符。
    -   **`save_dir`**: 用于存储特定轨迹运行的原始采样数据（视频）的目录。
    -   **`fps`**: 输出视频的每秒帧数。应与您的仿真渲染速率匹配。
    -   **`max_vis_dis`**: 深度相机归一化的最大距离（米）。超出此距离的深度值将被截断。这会影响保存的深度视频中深度值的缩放方式。默认为1.0米。

2.  **在线采样 (在仿真循环期间)**:
    在您的仿真循环内部，在每个步骤之后或以所需的帧速率调用 `sampling` 方法。
    ```python
    # 假设 'sim_node' 是一个对象或接口，提供
    # 对模拟器当前状态的访问，包括渲染图像、
    # 深度图和分割掩码。
    # 这需要根据您的特定模拟器 API 进行调整。
    # 例如，sim_node 可能有如下方法：
    # sim_node.get_rgb_image(cam_id)
    # sim_node.get_depth_image(cam_id, max_vis_dis)
    # sim_node.get_segmentation_mask(cam_id) -> 返回对象、机器人、背景的掩码

    samples.sampling(sim_node) # 重复调用此方法
    ```
    `SampleforDR.sampling(sim_node)` 方法预计会：
    -   对于 `cam_ids` 中的每个相机：
        -   获取 RGB 图像。
        -   获取深度图像（使用 `max_vis_dis`进行归一化）。
        -   获取区分以下内容的分割掩码：
            -   `objs` 中的每个对象。
            -   `robot_parts` 中的所有部件（组合成一个机器人掩码）。
            -   背景（所有不是 `obj` 或 `robot_part` 的部分）。
    -   在内部存储这些帧。

3.  **保存采集的数据**:
    仿真轨迹完成后：
    ```python
    samples.save()
    ```
    这会将采集的帧作为一组 `.mp4` 视频文件写入指定的 `save_dir`。预期的输出文件是：
    -   `rgb_<cam_id>.mp4`: 指定相机的 RGB 视频。
    -   `depth_<cam_id>.mp4`: 归一化后的深度视频。
    -   `mask_<obj_name>_<cam_id>.mp4`: `objs` 中每个对象的二进制掩码视频。
    -   `mask_robot_<cam_id>.mp4`: 组合机器人部件的二进制掩码视频。
    -   `mask_background_<cam_id>.mp4`: 背景的二进制掩码视频。
    *（原始文档提到 `cam.mp4`, `depth.mp4`, `obj1.mp4` 等。命名约定可能略有不同或可配置，但内容是关键。）*

### 5.2 阶段 2: 提示词生成

使用生成模型进行有效的域随机化需要良好的文本提示。`augment.py` 脚本（推测位于 `discoverse/randomain` 中）有助于创建这些提示。

-   **目的**: 生成一组多样化的文本描述，以指导 ComfyUI 图像生成过程。这些提示描述了期望的场景、对象和整体风格。

-   **操作模式**:

    1.  **`mode = 'input'` (从预定义描述批量生成)**:
        您为前景对象、机器人、背景、一般场景描述和否定提示提供基本描述。
        ```python
        # augment.py 中 'block_place' 任务的示例
        mode = 'input'
        fore_objs = { # 字典：{掩码中的对象名: "文本描述"}
            "block_green": "一个绿色方块",
            "bowl_pink": "一个粉色碗",
            # 机器人掩码通常单独处理或作为 fore_objs 的一部分
            "robot": "一个黑色机械臂" 
        }
        background = '一张桌子' # 一般背景描述
        scene = '在一个房间里，一个机械臂正在执行将方块夹入碗中的任务' # 整体动作/上下文
        negative = "场景中没有多余的物体，模糊，低质量" # 需要避免的内容
        num_prompts = 50 # 从这些输入生成的多样化提示数量
        ```
        -   然后，脚本可能会使用同义词替换、模板填充，甚至基于 LLM 的释义（如果集成）等技术来生成 `num_prompts` 个变体。例如，"一个绿色方块"可能变成"一个鲜艳的石灰绿色立方体"或"一块小而青翠的砖块"。背景"一张桌子"可能变成"一张木制书桌"、"一个金属车间工作台"等。`scene` 提示提供上下文。

    2.  **`mode = 'example'` (从示例提示增强)**:
        您提供一个包含种子提示的文件路径（例如 `example.jsonl`）。
        ```python
        # augment.py 中的示例
        mode = 'example'
        input_path = 'path/to/your/example_prompts.jsonl' # 每行一个 JSON 对象，每个对象是一个示例提示的文件
        num_prompts = 50
        ```
        -   `example.jsonl` 将包含结构化提示，可能带有占位符或特定样式。然后脚本增强这些示例以创建 `num_prompts` 个变体。
        -   如果您有一组高质量的提示并希望系统地扩展它们，则此模式很有用。

-   **输出**: `augment.py` 脚本通常会输出一个文件（例如 `.txt` 或 `.jsonl` 文件），其中包含生成的提示列表，供 `generate.py` 使用。

### 5.3 阶段 3: 随机化场景生成

这是核心步骤，使用采样数据和生成的提示来创建最终的随机化视频序列。这由 `discoverse/randomain/generate.py` 处理。

```bash
cd discoverse/randomain
python generate.py [--arg_name arg_value ...]
```

**`generate.py` 的关键操作步骤 (概念性)**:

1.  **加载数据**: 读取特定轨迹 (`work_dir`) 和相机 (`cam_id`) 的采样视频（RGB、深度、掩码）。
2.  **加载提示**: 加载由 `augment.py` 生成的文本提示。
3.  **遍历帧/关键帧**:
    -   对于输入视频中的每个**关键帧** (由 `flow_interval` 确定):
        -   选择一个提示（例如，轮询或从加载的提示中随机选择）。
        -   获取相应的深度帧。
        -   可能会使用对象掩码将不同的提示应用于不同区域或对场景的某些部分进行修复/扩展绘制 (高级功能)。基本用法暗示了一个由整体深度调节的全局提示。
        -   将深度图和提示发送到 ComfyUI（通过其 API 或命令行界面，如果 ComfyUI 作为服务器运行）。
        -   ComfyUI 使用 SDXL Turbo + ControlNet (Depth) 生成一个新的 RGB 图像，该图像匹配提示的描述，同时遵循深度帧的几何形状。
        -   存储生成的随机化帧。
    -   如果 `flow_interval > 1`:
        -   对于两个生成的关键帧*之间*的帧：
            -   使用选定的 `flow_method`（`rgb` 表示 Farneback，`raft` 表示 RAFT）计算两个关键帧（或与关键帧对应的原始 RGB 帧）之间的光流。
            -   使用计算出的光流对先前生成的关键帧（或两个关键帧的组合）进行变形，以合成中间帧。
            -   存储插值帧。
4.  **保存输出**: 将所有生成和插值的帧组合成一个新的随机化输出视频文件。

## 6. `generate.py` 的关键参数

以下是 `generate.py` 的关键参数。有关完整列表和默认值，请参阅脚本本身。

-   **`--task_name`** (例如 `block_place`):
    -   **描述**: 任务的名称，通常用于组织输出文件或选择特定于任务的提示。
-   **`--work_dir`** (例如 `000`, `001`):
    -   **描述**: 指定 `data/randomain/`（或类似的基本路径）内的子目录，其中包含单个轨迹的采样数据。这通常对应于 `SampleforDR` 期间使用的 `save_dir` 名称。
-   **`--cam_id`** (例如 `front_camera`):
    -   **描述**: 应处理其采样数据的相机的标识符。这应与采样期间使用的 `cam_ids` 之一匹配。
-   **`--fore_objs`** (例如 `['block_green', 'bowl_pink', 'robot']`):
    -   **描述**: 前景对象名称列表（如果机器人要被视为提示的前景元素，则包括机器人）。如果生成逻辑具有这种粒度，这有助于潜在地关联提示的特定部分或掩码。它应与 `SampleforDR` 和 `augment.py` 中使用的对象名称一致。
-   **`--wide`, `--height`** (例如 `1280`, `768`):
    -   **描述**: 输入处理（如果进行大小调整）和输出生成的图像/视频的宽度和高度。建议使用 SDXL 良好支持的尺寸。
-   **`--num_steps`** (例如 `4`):
    -   **描述**: SDXL Turbo 的扩散步骤数。Turbo 模型专为极少的步骤（例如 1-8 步）而设计。
-   **`--flow_interval`** (例如 `1`, `5`, `10`):
    -   **描述**: 确定完整生成模型的运行频率。
        -   `1`: 使用 ComfyUI 生成每一帧（最高质量/一致性，最慢）。
        -   `N > 1`: 每 `N` 帧使用 ComfyUI 生成一帧。`N-1` 个中间帧使用指定的 `flow_method` 生成。
-   **`--flow_method`** (例如 `'rgb'`, `'raft'`):
    -   **描述**: 指定当 `flow_interval > 1` 时使用的光流算法。
        -   `'rgb'`: Farneback 方法。
        -   `'raft'`: RAFT 方法（需要部署 RAFT 模型）。

## 7. 输出

`generate.py` 脚本的主要输出将是：
-   **随机化视频**: 通常位于 `work_dir` 的子目录内（例如 `data/randomain/000/randomized_front_camera.mp4`）。这些视频包含视觉上已更改的场景。
-   **可能的中间文件**: 根据实现方式，它可能还会保存单个生成的帧或光流场。

目标是将这些随机化视频用作增强的训练数据，用于下游任务，例如训练机器人策略或感知系统。 