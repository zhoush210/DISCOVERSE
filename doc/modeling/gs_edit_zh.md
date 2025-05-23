# 3D 高斯溅射 (3D Gaussian Splatting) 模型编辑指南

## 1. 引言

### 1.1 什么是 3D 高斯溅射 (3D Gaussian Splatting)？
3D 高斯溅射（3DGS）是一种新兴的场景表示和渲染技术。它使用数百万个各向异性的3D高斯函数来表示场景。每个高斯函数具有位置、形状（通过协方差矩阵或等效的缩放和旋转参数定义）、颜色（通常使用球谐函数表示以实现视角相关效果）和不透明度等属性。3DGS 能够以高保真度实时渲染复杂场景，并且在逆向渲染（从图像重建3D场景）方面表现出色。

### 1.2 3DGS PLY 文件格式
3DGS 模型通常存储为 `.ply` 文件。与传统的网格（Mesh）`.ply` 文件不同，3DGS 的 `.ply` 文件包含描述每个高斯函数的特定属性。

**如何区分一个 `.ply` 文件是否是 3DGS 模型？**
主要看文件的头部信息（header）。一个典型的 3DGS `.ply` 文件头部如下所示：

```ply
ply
format binary_little_endian 1.0
element vertex 1000000  // 表示高斯函数的数量
property float x
property float y
property float z
property float nx        // 通常不使用法线 对应数值均为0
property float ny
property float nz
property float f_dc_0    // 球谐函数的直流分量 (基础颜色 R)
property float f_dc_1    // 球谐函数的直流分量 (基础颜色 G)
property float f_dc_2    // 球谐函数的直流分量 (基础颜色 B)
property float f_rest_0  // 球谐函数的高阶系数 (共 (max_sh_degree+1)^2 - 1 个)
// ... (更多 f_rest_... 属性)
property float opacity   // 不透明度
property float scale_0   // 高斯函数的缩放参数
property float scale_1
property float scale_2
property float rot_0     // 高斯函数的旋转参数 (四元数 w)
property float rot_1     // 高斯函数的旋转参数 (四元数 x)
property float rot_2     // 高斯函数的旋转参数 (四元数 y)
property float rot_3     // 高斯函数的旋转参数 (四元数 z)
end_header
```
**关键判断依据**：检查头部是否包含 `f_dc_` (球谐函数直流分量) 和 `f_rest_` (球谐函数高阶系数)，以及 `scale_` 和 `rot_` (高斯函数的尺度和旋转) 属性。这些是3DGS模型的典型特征。

## 2. `gsply_edit.py` 脚本介绍

本仓库提供了一个用于编辑 3DGS 模型的 Python 脚本：`scripts/gsply_edit.py`。
该脚本允许用户对 3DGS 模型进行定量的平移、旋转和缩放操作。

**核心功能：**
*   **几何变换**：修改每个高斯函数中心点的位置。
*   **姿态调整**：旋转每个高斯函数的方向。
*   **尺度调整**：缩放每个高斯函数的大小。
*   **球谐函数 (SH) 旋转**：由于球谐函数表示的颜色是视角相关的，当高斯函数本身被旋转时，其对应的SH系数也需要进行相应的旋转，以保持外观一致性。脚本会自动处理SH系数的旋转。

## 3. 依赖安装
运行此脚本需要以下 Python 库：
*   `numpy`：用于数值计算。
*   `scipy`：用于科学计算，特别是其空间变换模块（如四元数和旋转矩阵的转换）。
*   `torch`：PyTorch库，部分计算（如SH旋转）可能利用其张量运算。
*   `einops`：用于更灵活、更强大的张量操作。
*   `e3nn`：用于处理与SO(3)群相关的操作，特别是球谐函数的旋转。
*   `tqdm`：用于显示进度条。

你可以使用 pip 安装这些依赖：
```bash
pip install numpy scipy torch einops e3nn tqdm
```

## 4. 核心概念详解

### 4.1 变换操作
脚本支持以下基本变换：
*   **平移 (Translation)**：将模型沿 X, Y, Z 轴移动指定的距离。
*   **旋转 (Rotation)**：将模型绕指定轴旋转指定的角度。旋转通常通过四元数 (Quaternion) 定义。脚本内部会将此旋转应用于每个高斯函数的位置和姿态。
*   **缩放 (Scaling)**：将模型整体放大或缩小指定的倍数。这会影响高斯函数的位置（相对于原点）和它自身的尺度参数。

### 4.2 变换顺序
当同时指定平移、旋转和缩放时，脚本应用的变换顺序如下：
1.  **旋转**：首先应用旋转到高斯函数的位置和姿态。
2.  **平移**：然后应用平移到旋转后的高斯函数位置。
3.  **缩放**：最后，应用缩放。缩放会同时影响高斯函数的位置（使其离原点更远或更近）和高斯函数自身的 `scale_` 属性。

这些变换组合成一个 4x4 的变换矩阵，应用于每个高斯函数位姿。SH系数会根据变换矩阵中的旋转部分进行相应调整。

### 4.3 球谐函数 (Spherical Harmonics - SH)
在 3DGS 中，球谐函数用于表示高斯函数的颜色，特别是其随视角变化的辐射度（view-dependent appearance）。
*   `f_dc_0`, `f_dc_1`, `f_dc_2`：代表 SH 的零阶系数（直流分量），近似于物体的基础漫反射颜色。
*   `f_rest_...`：代表 SH 的高阶系数，这些系数使得颜色能够随着观察方向的改变而变化，从而模拟出更复杂的光照和材质效果，如高光。

当高斯函数本身发生旋转时，其局部坐标系会改变，因此描述其视角相关外观的SH系数也必须相应旋转，以确保从任何方向观察时颜色表现正确。`gsply_edit.py` 脚本会自动处理这个复杂的旋转过程。

### 4.4 高斯参数
每个高斯函数由以下主要参数定义：
*   **位置 (`x`, `y`, `z`)**: 高斯函数的中心点坐标。
*   **旋转 (`rot_0`, `rot_1`, `rot_2`, `rot_3`)**: 定义高斯函数方向的四元数 (通常存储为 `w, x, y, z` 顺序)。脚本输入时也应遵循此约定或脚本内部约定的顺序。
*   **尺度 (`scale_0`, `scale_1`, `scale_2`)**: 定义高斯函数在三个主轴方向上的大小或标准差。这些尺度与旋转共同定义了高斯函数的协方差矩阵。
*   **不透明度 (`opacity`)**: 高斯函数的不透明度。

脚本在进行变换时，会正确更新这些参数。例如，旋转操作会更新高斯函数的位置（如果它不在原点）和 `rot_` 四元数。缩放操作会更新位置和 `scale_` 参数。

## 5. 脚本使用

### 5.1 命令行参数
脚本通过命令行参数接收输入：

```bash
python scripts/gsply_edit.py <input_file> [options]
```

*   **`input_file`** (必选):
    *   类型: `str`
    *   描述: 输入的 3DGS 二进制 `.ply` 文件的路径。

*   **`-o, --output_file`** (可选):
    *   类型: `str`
    *   描述: 输出的变换后的 `.ply` 文件路径。
    *   默认值: 如果不指定，输出文件名将在输入文件名的基础上添加 `_trans.ply` 后缀 (例如, `input.ply` -> `input_trans.ply`)。

*   **`-t, --transform`** (可选):
    *   类型: `float` (3个值)
    *   格式: `x y z`
    *   描述: 定义平移向量。
    *   示例: `--transform 0.1 0.0 -0.2` (沿X轴平移0.1，Y轴不变，Z轴平移-0.2)。
    *   默认值: `[0, 0, 0]` (不平移)。

*   **`-r, --rotation`** (可选):
    *   类型: `float` (4个值)
    *   格式: `x y z w` (四元数)
    *   描述: 定义旋转。**请注意四元数的顺序为 `x y z w`。**
    *   示例: `--rotation 0.0 0.0 0.38268343 0.92387953` (绕Z轴旋转45度)。
    *   默认值: `[0, 0, 0, 1]` (不旋转，单位四元数)。

*   **`-s, --scale`** (可选):
    *   类型: `float`
    *   描述: 定义均匀缩放因子。
    *   示例: `--scale 1.5` (放大1.5倍)，`--scale 0.5` (缩小到0.5倍)。
    *   默认值: `1.0` (不缩放)。

### 5.2 注意事项
*   输入文件必须是二进制小端序 (binary_little_endian) 格式的 PLY 文件。
*   确保提供的四元数 (`--rotation` 参数) 是 `x y z w` 格式。
*   变换操作是围绕场景原点 (0,0,0) 进行的。如果想绕模型中心旋转或缩放，需要先将模型中心平移到原点，执行操作，然后再平移回去（这需要多次运行脚本或手动计算变换）。

## 6. 使用示例

### 6.1 准备工作：计算旋转四元数
如果你需要根据欧拉角计算四元数，可以使用 `scipy`：
```python
from scipy.spatial.transform import Rotation

# 示例：绕Z轴顺时针旋转45度 (等效于欧拉角 z=-45 或 z=315)
# 或者，如果脚本的旋转是逆时针的，则用正角度
# 假设脚本期望的旋转是标准的右手坐标系旋转
# 绕Z轴正向旋转45度 (从+X看向+Y为逆时针)
r = Rotation.from_euler('z', 45, degrees=True)
quat_xyzw = r.as_quat() # scipy 输出 [x, y, z, w]
print(f"绕Z轴旋转45度的四元数 (xyzw): {quat_xyzw}")
# [0.         0.         0.38268343 0.92387953]
```

### 6.2 示例命令

**场景**：假设我们有一个名为 `data/scene.ply` 的3DGS模型。

1.  **仅平移**：将模型沿X轴正向移动0.5个单位，沿Z轴负向移动0.2个单位。
    ```bash
    python scripts/gsply_edit.py data/scene.ply -o data/scene_translated.ply -t 0.5 0.0 -0.2
    ```

2.  **仅旋转**：将模型绕Z轴（向上为正）逆时针旋转45度。
    上面计算得到的四元数为 `[0, 0, 0.38268343, 0.92387953]`。
    ```bash
    python scripts/gsply_edit.py data/scene.ply -o data/scene_rotated.ply -r 0.0 0.0 0.38268343 0.92387953
    ```

3.  **仅缩放**：将模型放大到原来的2倍。
    ```bash
    python scripts/gsply_edit.py data/scene.ply -o data/scene_scaled.ply -s 2.0
    ```

4.  **组合变换**：
    *   沿X轴平移 0.3 米，Y轴平移 0.4 米，Z轴平移 -0.5 米。
    *   接着，（相对于新的原点）绕Z轴顺时针旋转 45 度 (这里我们用之前计算的绕Z轴旋转45度的四元数，如果希望顺时针，需要用 `Rotation.from_euler('z', -45, degrees=True)`)。假设我们使用之前计算的逆时针45度四元数：`0. 0. 0.38268343 0.92387953`。
    *   最后，将尺寸缩放到原先的1.5倍。
    *   模型输入路径是：`data/000000.ply`，模型输出路径是：`data/000000_transpose.ply`。

    ```bash
    python scripts/gsply_edit.py data/000000.ply -o data/000000_transpose.ply -t 0.3 0.4 -0.5 -r 0. 0. 0.38268343 0.92387953 -s 1.5
    ```
    **注意**：脚本内部的变换矩阵构建顺序是先应用旋转，再应用平移（通过矩阵乘法 `transformMatrix @ pose_arr[:]`，其中 `transformMatrix` 由旋转和平移构成）。缩放是最后独立应用的。这意味着平移是在物体已被旋转到新朝向后，沿着世界坐标轴进行的。如果你期望的平移是在物体原始坐标系下，然后旋转和平移整个物体，你需要仔细构造变换参数或分步操作。

## 7. 3DGS 模型在线编辑

除了使用脚本进行定量编辑外，还可以使用在线工具进行可视化编辑：
[3dgs在线编辑 (SuperSplat Editor)](https://superspl.at/editor)
该工具支持的功能包括：
1.  平移
2.  旋转
3.  缩放
4.  多种方式选中点云（高斯函数）
5.  删除选中的点云
6.  切换不同的渲染模式

这对于快速预览、粗略调整或移除不需要的部分非常有用。
