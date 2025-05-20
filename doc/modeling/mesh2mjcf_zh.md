# Mesh2MJCF 工具使用文档

## 1. 简介

### 1.1 什么是 Mesh？
Mesh（网格）是3D模型的一种表示方式，由顶点（vertices）、边（edges）和面（faces）组成。常见的网格文件格式包括：
- OBJ：一种简单的3D模型格式，支持顶点、纹理坐标和法线信息
- STL：常用于3D打印的格式，只包含三角形面片信息

### 1.2 什么是 MJCF？
MJCF（MuJoCo XML Configuration Format）是 MuJoCo 物理引擎使用的配置文件格式。它使用 XML 语法来描述一个完整的物理仿真环境，或者环境中的一部分（例如一个机器人模型）。MJCF 文件定义了：
- **物理世界 (World)**: 包括全局参数如重力、时间步长、求解器设置等。
- **物体 (Bodies)**: 场景中的基本单元，具有层级结构（父子关系），可以嵌套定义复杂的铰接系统。
- **几何形状 (Geometries)**: 附加在物体上的可见或可碰撞形状，如球体、胶囊体、圆柱体、方盒、平面以及网格（Mesh）。
- **关节 (Joints)**: 连接物体并定义它们之间相对运动的自由度，如转动关节（hinge）、滑动关节（slide）、球状关节（ball）或自由关节（free）。
- **物理属性 (Physical Properties)**: 如质量（mass）、惯性张量（inertia）、摩擦力（friction）、阻尼（damping）等。
- **资源 (Assets)**: 如网格文件（.obj, .stl, .msh）、纹理图片（.png）、高度场数据等。
- **传感器 (Sensors)**: 用于测量仿真过程中的各种物理量，如触摸、力、加速度、关节角度等。
- **执行器 (Actuators)**: 用于对模型施加力或控制，如马达。
- **其他**: 还包括灯光、摄像机、皮肤（skin，用于软体）、肌腱（tendon）等。

**MJCF 的核心特点和主要标签：**

*   **层级结构**: MJCF 文件以 `<mujoco>` 作为根元素。模型的主体结构通常定义在 `<worldbody>` 标签内。
    *   **`<mujoco model="model_name">`**: 根元素，可以包含模型名称。
    *   **`<compiler>`**: 定义编译选项，如网格和纹理目录 (`meshdir`, `texturedir`)，角度单位 (`angle="degree"` 或 `"radian"`)，坐标约定等。脚本中设置了 `meshdir="../meshes"` 和 `texturedir="../textures/"`，意味着 MJCF 文件会从其所在位置的上一级目录中的 `meshes` 和 `textures` 文件夹寻找资源。
    *   **`<option>`**: 定义全局物理参数，如重力 (`gravity`)、时间步长 (`timestep`)、积分器类型等。脚本中预览时设置了 `gravity="0 0 -9.81"`。
    *   **`<asset>`**: 资源定义区。
        *   **`<mesh name="unique_mesh_name" file="path/to/mesh.obj" scale="1 1 1"/>`**: 定义一个网格资源。`name` 必须唯一，`file` 指向实际的网格文件。`scale` 可以缩放网格。
        *   **`<texture name="unique_texture_name" type="2d" file="path/to/texture.png"/>`**: 定义一个2D纹理。
        *   **`<material name="unique_material_name" texture="texture_name" rgba="0.8 0.2 0.2 1"/>`**: 定义材质，可以关联纹理或直接指定颜色和反射属性等。
    *   **`<default>`**: 可以为不同类别的元素（如 geom, joint）定义默认属性。脚本中使用它来为视觉几何体设置默认的碰撞属性。
        *   **`<default class="obj_visual">`**: 定义一个名为 `obj_visual` 的默认类。
        *   **`<geom group="2" type="mesh" contype="0" conaffinity="0"/>`**: 在 `obj_visual` 类下的几何体，默认属于碰撞组2，类型为网格，并且 `contype="0" conaffinity="0"` 使其成为纯视觉物体，不参与物理碰撞。
    *   **`<worldbody>`**: 包含场景中的所有物体和灯光等。
        *   **`<body>`**: 定义一个物理实体。可以嵌套，形成父子关系。具有 `name`, `pos` (位置), `quat` (姿态，四元数) 或 `euler` (欧拉角) 等属性。
            *   **`<joint type="free"/>`**: 在 `<body>` 内部定义，赋予该物体6个自由度（3平移，3旋转）相对于其父物体（或世界，如果是顶级物体）。脚本中，如果不是 `fix_joint`，则会添加此自由关节。
            *   **`<inertial pos="0 0 0" mass="0.1" diaginertia="0.01 0.01 0.01"/>`**: 定义物体的惯性属性。`pos` 是质心相对于物体坐标系的位置，`mass` 是质量，`diaginertia` 是对角惯性张量（假设惯性主轴与物体坐标轴对齐）。
            *   **`<geom type="sphere" size="0.1" rgba="1 0 0 1" mass="0"/>`**: 定义一个几何形状附加到此物体。`type` 可以是 `plane`, `sphere`, `capsule`, `box`, `cylinder`, `mesh`。`size` 根据类型有不同含义。`rgba` 定义颜色。`mass` 如果在这里指定，MuJoCo会基于几何形状和密度自动计算惯性（如果 `inertial` 标签未提供或不完整）。通常建议在 `inertial` 标签中明确指定质量和惯量。如果 `geom` 用于碰撞，则需要设置 `contype` 和 `conaffinity`。
            *   **`<geom mesh="unique_mesh_name" material="material_name" class="obj_visual"/>`**: 使用在 `<asset>` 中定义的网格和材质。通过 `class` 可以继承默认属性。
        *   **`<light pos="0 0 3" dir="0 0 -1"/>`**: 定义光源。
    *   **`<include file="another_model.xml"/>`**: 用于将一个 MJCF 文件的内容包含到另一个文件中，便于模块化设计。脚本大量使用此功能，将资源依赖 (`_dependencies.xml`) 和物体定义 (`{asset_name}.xml`) 分开，并在预览时包含它们。

MJCF 的设计旨在平衡表达能力和仿真效率。通过这种结构化的 XML 文件，用户可以精确地构建和控制复杂的物理场景。本脚本 (`mesh2mjcf.py`) 的核心任务就是将输入的网格文件转换为符合 MJCF规范的 `.xml` 文件，使其可以在 MuJoCo 中加载和仿真。

## 2. 物理概念

在进行物理仿真时，准确地定义物体的物理属性至关重要。这些属性直接影响物体在仿真世界中的行为。`mesh2mjcf.py` 脚本允许用户指定其中两个最基本的属性：质量和惯性张量。

### 2.1 质量（Mass）
- **定义**: 质量是衡量物体所含物质多少的物理量，也是物体惯性大小的度量。在国际单位制中，单位为千克 (kg)。
- **在仿真中的作用**:
    - **动力学**: 根据牛顿第二定律 (F=ma)，质量决定了物体在受到外力时加速度的大小。质量越大，获得相同加速度所需的外力越大。
    - **重力**: 物体受到的重力与其质量成正比 (Fg = mg)。
    - **碰撞响应**: 在碰撞事件中，质量较大的物体对质量较小的物体影响更大。
- **脚本中的参数**: `--mass` (浮点数，默认值: `0.001` kg)。
- **设置考量**:
    - **真实性**: 理想情况下，质量应与真实物体的质量一致。
    - **密度**: 如果物体的真实质量未知，可以根据其体积和大致密度估算。例如，水的密度是 1000 kg/m³。
    - **仿真稳定性**: 质量过小或过大的物体，或者场景中物体质量差异悬殊，都可能导致仿真不稳定。MuJoCo 对质量比例有一定的容忍度，但极端情况仍需注意。如果一个非常轻的物体与一个非常重的物体交互，可能需要更小的时间步长或调整求解器参数。
    - 该脚本默认质量 `0.001` kg (1克) 非常小，适用于小型桌面物体。对于大型物体，你需要显著增大此值。

### 2.2 惯性张量（Inertia Tensor）
- **定义**: 惯性张量描述了刚体在旋转运动中抵抗角加速度的特性，类似于线性运动中质量抵抗线性加速度的作用。它是一个对称的 3x3 矩阵，其元素取决于物体的质量分布和几何形状。
    \[
    I = \\begin{bmatrix}
    I_{xx} & I_{xy} & I_{xz} \\\
    I_{yx} & I_{yy} & I_{yz} \\\
    I_{zx} & I_{zy} & I_{zz}
    \\end{bmatrix}
    \]
    其中，对角线元素 \(I_{xx}, I_{yy}, I_{zz}\) 分别是绕物体自身坐标系 x, y, z 轴旋转的转动惯量。非对角线元素 \(I_{xy}, I_{xz}, I_{yz}\) 称为惯性积，表示质量分布的不对称性。
- **主轴和对角惯性**: 任何刚体都存在三个相互垂直的特殊轴，称为惯性主轴。如果物体的坐标系与这些主轴对齐，则惯性张量中的所有惯性积（非对角线元素）都为零，惯性张量成为对角矩阵：
    \[
    I = \\begin{bmatrix}
    I_{1} & 0 & 0 \\\
    0 & I_{2} & 0 \\\
    0 & 0 & I_{3}
    \\end{bmatrix}
    \]
    这里的 \(I_1, I_2, I_3\) 是绕三个主轴的转动惯量，称为主转动惯量。
- **在仿真中的作用**:
    - **旋转动力学**: 惯性张量决定了物体在受到力矩作用时的角加速度。不同的转动惯量意味着绕不同轴旋转的难易程度不同。
    - **稳定性**: 不准确的惯性张量会导致物体在仿真中表现出不自然的旋转行为，例如，一个细长的杆子可能很容易绕其长轴旋转，但很难绕其短轴翻滚。
- **脚本中的参数**: `--diaginertia` (3个浮点数，默认值: `[0.00002, 0.00002, 0.00002]`)。
    - 这个参数对应的是**对角化后的惯性张量的主转动惯量**，即 \(I_1, I_2, I_3\)。脚本假设输入的网格的坐标轴已经与其惯性主轴对齐，或者用户提供的就是沿物体局部坐标系x,y,z轴的转动惯量值。
- **设置考量**:
    - **自动计算**: 许多 CAD 软件可以根据3D模型的几何形状和指定的密度自动计算质量和惯性张量。MuJoCo 自身也可以根据 `<geom>` 的形状和密度（如果定义了 `density` 属性）来估算惯性，但这通常用于简单几何体。对于复杂网格，最好由用户提供。
    - **估算**: 对于简单形状，有标准的转动惯量公式（例如，实心球体、圆柱体、长方体）。对于复杂形状，精确估算很困难。
    - **对称性**: 如果物体具有对称性（例如，一个球体或一个立方体），其惯性主轴通常与对称轴对齐，且对应的主转动惯量可能相等。脚本的默认值 `[2e-5, 2e-5, 2e-5]` 意味着它假设物体是一个在三个轴上惯性特性相似的小物体。
    - **单位**: MuJoCo 中的惯性单位通常是 kg·m²。
    - 不正确的惯性值，特别是相对于质量而言过小或过大的惯性值，都可能导致仿真不稳定或物体行为怪异。

理解并正确设置这些物理参数是实现逼真和稳定物理仿真的关键步骤。

## 3. 凸分解（Convex Decomposition）

### 3.1 什么是凸分解？
- **凸形状 (Convex Shape)**: 一个几何形状是凸的，如果形状内任意两点之间的连线段完全包含在形状内部。直观地说，凸形状没有"凹陷"或"洞穴"。例如，球体、立方体、胶囊体都是凸的。
- **非凸形状 (Non-Convex Shape / Concave Shape)**: 如果一个形状不是凸的，那么它就是非凸的。例如，一个星形、一个字母"L"的形状、一个有孔的甜甜圈（环面）都是非凸的。

- **凸分解 (Convex Decomposition)**: 是将一个复杂的非凸几何形状精确地或近似地表示为一组（通常是多个）凸形状的并集的过程。理想情况下，这些凸形状的并集应该与原始非凸形状的体积和表面尽可能接近。
    - 例如，一个字母"L"的形状可以被分解为两个长方体。
    - 一个更复杂的物体，如一把椅子，可能需要被分解成多个凸包（convex hulls）来近似其形状。

- **为什么需要凸分解？**
    - **碰撞检测效率**: 许多高效的碰撞检测算法（如GJK、SAT）是为凸形状设计的。检测两个凸形状之间的碰撞远比检测两个任意非凸形状之间的碰撞要快得多和简单得多。
    - **碰撞检测鲁棒性**: 对于非凸形状，碰撞检测算法可能更复杂，更容易出错，或者产生不稳定的接触点。

### 3.2 在 MuJoCo 中的作用
- **MuJoCo 的碰撞系统**: MuJoCo 的内置碰撞检测引擎主要依赖于成对的凸几何体之间的检测。它可以处理多种基本凸形状（球、胶囊、盒、圆柱、平面、椭球）以及凸网格（通过 `<mesh>` 标签加载的凸多面体）。
- **处理非凸网格**: 当你直接加载一个复杂的非凸网格（如一把椅子、一个复杂的机械零件）作为 MuJoCo 中的单个 `<geom>` 时，MuJoCo 默认会计算该网格的**凸包 (convex hull)** 并将其用于碰撞。这意味着，如果你的椅子有凹进去的座位部分，其凸包会"填满"这个凹陷，导致碰撞行为与真实形状不符（例如，一个小球无法滚入座位，而是会停在凸包的表面上）。
- **通过凸分解提高精度**: 为了获得对非凸物体更精确的碰撞响应，我们可以预先将其分解为多个凸部件。然后，在 MJCF 中，原始的非凸物体由多个 `<geom>` 元素表示，每个 `<geom>` 对应一个凸部件网格。
    - **优点**:
        - **更精确的接触**: 物体可以更真实地相互作用，例如物体可以进入另一个物体的凹陷部分。
        - **更稳定的仿真**: 清晰定义的接触点和法线有助于求解器计算更稳定的力。
    - **脚本中的实现 (`-cd` / `--convex_decomposition`)**: 当使用此选项时，`mesh2mjcf.py` 脚本会调用 `coacd` 库 (CoACD: Contact-Aware Convex Decomposition) 来执行凸分解。`coacd` 会尝试生成一组尽可能少但又能较好地近似原始网格的凸部件。
        - 脚本会将每个生成的凸部件保存为一个单独的 `.obj` 文件。
        - 在生成的 `{asset_name}_dependencies.xml` 文件中，会为每个凸部件创建一个 `<mesh>` 条目。
        - 在生成的 `{asset_name}.xml` 文件中，会为每个凸部件创建一个 `<geom type="mesh" ... />` 条目，这些 `geom` 将共同构成物体的碰撞模型。此时，原始的完整网格通常只用作视觉模型（如果提供了纹理或单独的视觉 `<geom>`）。
- **权衡与考量**:
    - **精度 vs. 性能**: 分解出的凸部件越多，对原始形状的近似就越精确，但同时也会增加仿真中需要处理的碰撞对数量，从而可能降低仿真速度。
    - **分解质量**: 凸分解算法的质量也很重要。好的分解应该能够用较少的部件捕捉到形状的关键特征。
    - **适用场景**: 对于需要精细操作（如机器人抓取复杂物体）、物体紧密堆叠或物体需要在复杂环境中导航的场景，使用凸分解通常是必要的。
    - **视觉 vs. 碰撞模型**: 常见做法是使用原始的、可能非凸的、高细节的网格作为**视觉模型** (visual geometry)，而使用其凸分解版本或者简化的凸包作为**碰撞模型** (collision geometry)。这样可以在保持视觉保真度的同时，获得高效且相对准确的物理交互。脚本在启用凸分解且有纹理时，会将原始网格用于视觉（通过 `class="obj_visual"`使其不参与碰撞），而分解出的凸部件则用于碰撞。

通过凸分解，我们可以让 MuJoCo 更准确地模拟复杂形状物体之间的物理交互。

## 4. 安装依赖

为了完整使用 `mesh2mjcf.py` 脚本的所有功能，你需要安装一些 Python 库。部分功能（如凸分解和预览）依赖特定的库。

### 4.1 必需依赖 (用于凸分解)
如果你计划使用凸分解功能 (`-cd` 或 `--convex_decomposition` 选项)，以下两个库是必需的：

-   **`trimesh`**: 一个用于加载和处理各种3D网格文件（包括 .obj, .stl 等）的 Python 库。它提供了读取网格顶点、面片信息，以及导出网格的功能。在此脚本中，`trimesh` 用于：
    -   加载用户提供的输入网格文件。
    -   将 `coacd` 分解出的凸包部件（顶点和面）转换回 `trimesh` 对象，以便可以将其导出为 `.obj` 文件。

-   **`coacd`**: (Contact-Aware Convex Decomposition) 一个实现凸分解算法的库。它能够将一个非凸网格分解为一组近似的凸包部件，同时试图优化接触点，使得分解结果在物理仿真中表现更好。
    -   脚本使用 `coacd.Mesh` 来准备网格数据，并调用 `coacd.run_coacd` 来执行分解。

**安装命令**:
```bash
pip install trimesh coacd
```
如果未安装这些库而尝试使用 `-cd` 选项，脚本会打印错误信息并退出。

### 4.2 可选依赖 (用于预览)
如果你希望在转换完成后立即使用 MuJoCo 内置查看器预览生成的模型 (`--verbose` 选项)，你需要安装 `mujoco` Python 包。

-   **`mujoco`**: MuJoCo 官方 Python 绑定。它不仅提供了与 MuJoCo 仿真核心交互的 API，还包含了一个简单的 MJCF 查看器模块 `mujoco.viewer`。
    -   脚本通过 `python -m mujoco.viewer --mjcf=/path/to/your/model.xml` 命令来启动这个查看器。

**安装命令**:
```bash
pip install mujoco
```
如果未安装 `mujoco` 而尝试使用 `--verbose` 选项，脚本本身可能不会直接报错（因为它只是构造一个命令行字符串并用 `os.system` 执行），但 `os.system` 调用会失败，或者 MuJoCo 查看器无法启动，并可能在终端显示相应的错误信息。

**其他隐式依赖**: 
- `argparse`: 用于解析命令行参数，是 Python 标准库的一部分，通常无需额外安装。
- `numpy`: `trimesh` 和 `coacd` (以及 `mujoco`) 通常会依赖 `numpy` 进行数值运算。通过 `pip` 安装它们时，`numpy` 通常会自动作为依赖被安装。

## 5. 使用方法

`mesh2mjcf.py` 脚本通过命令行界面 (CLI) 进行操作。你需要打开终端或命令行提示符，导航到脚本所在的目录（或者确保脚本在你的系统路径中），然后执行它。

### 5.1 基本命令格式

```bash
python scripts/mesh2mjcf.py <input_file_path> [options]
```
- `python`: 你的 Python 解释器 (可能是 `python3`，取决于你的环境)。
- `scripts/mesh2mjcf.py`: 指向脚本文件的路径。如果你的当前目录就是 `scripts` 的父目录，那么这个路径是正确的。
- `<input_file_path>`: **必需参数**。指定你要转换的网格文件的完整路径或相对路径。支持 `.obj` 和 `.stl` 格式。
- `[options]`: 一系列可选参数，用于控制转换过程的各个方面，如颜色、物理属性、是否进行凸分解等。

### 5.2 参数说明

以下是脚本支持的详细参数列表：

-   **`input_file`** (位置参数, 必需)
    -   类型: `str`
    -   描述: 输入的网格文件路径。请确保文件存在并且是有效的 `.obj` 或 `.stl` 文件。脚本会根据文件扩展名来确定文件类型。
    -   示例: `data/my_model.obj`, `/path/to/your/mesh.stl`

-   **`--rgba R G B A`** (可选)
    -   类型: 4个 `float` 值
    -   范围: 每个值通常在 `0.0` 到 `1.0` 之间。
    -   默认值: `[0.5, 0.5, 0.5, 1.0]` (灰色，不透明)
    -   描述: 定义网格的 RGBA (红, 绿, 蓝, 透明度) 颜色。 
        -   如果提供了 `--texture` 选项，此处的 Alpha (透明度) 值会被脚本内部逻辑强制设为 `0.0` 以确保纹理完全显示，但 R, G, B 值仍可能影响无纹理的凸分解部件的颜色（如果进行了凸分解）。
    -   示例: `--rgba 0.8 0.2 0.2 1.0` (红色，不透明)

-   **`--texture TEXTURE_PATH`** (可选)
    -   类型: `str`
    -   默认值: `None`
    -   描述: 指定纹理贴图文件的路径。目前主要支持 `.png` 格式（脚本示例中是这样，具体支持取决于 MuJoCo 和底层渲染器）。
        -   如果提供，脚本会将纹理文件复制到资源目录，并在 MJCF 中创建相应的 `<texture>` 和 `<material>` 定义。
        -   纹理坐标 (UV coordinates) 应该已经存在于你的 `.obj` 文件中。`.stl` 文件通常不包含纹理坐标。
        -   如上所述，提供纹理时，`--rgba` 的 Alpha 值会被设为 `0.0`。
    -   示例: `--texture /path/to/my_texture.png`

-   **`--mass MASS`** (可选)
    -   类型: `float`
    -   默认值: `0.001` (单位: kg)
    -   描述: 指定物体的质量。详见"物理概念"部分的质量说明。
    -   示例: `--mass 0.5` (0.5 kg)

-   **`--diaginertia D D D`** (可选)
    -   类型: 3个 `float` 值
    -   默认值: `[2e-5, 2e-5, 2e-5]` (单位: kg·m²)
    -   描述: 指定物体对角化后的惯性张量（即沿物体局部 x, y, z 轴的主转动惯量）。详见"物理概念"部分的惯性张量说明。
    -   示例: `--diaginertia 0.01 0.01 0.005`

-   **`--fix_joint`** (可选标志)
    -   类型: 无参数 (布尔标志)
    -   默认行为: 不固定，即物体有一个自由关节 (`<joint type="free"/>`)，可以在空间中自由移动和旋转。
    -   描述: 如果使用此标志，物体将不会被赋予自由关节。这意味着它在 MJCF 模型中将是静态的，或者其运动将完全由父物体的运动决定（如果它被包含在另一个 `<body>` 中）。这对于创建固定在世界上的物体（如地面、墙壁）或复杂机器人中没有独立自由度的连杆部分很有用。
    -   示例: `--fix_joint`

-   **`-cd`, `--convex_decomposition`** (可选标志)
    -   类型: 无参数 (布尔标志)
    -   默认行为: 不进行凸分解。
    -   描述: 如果使用此标志，脚本将尝试使用 `coacd` 库将输入的网格分解为多个凸包部件。这对于非凸物体获得更精确的碰撞响应非常重要。详见"凸分解"部分的说明。
        -   **注意**: 此选项需要 `trimesh` 和 `coacd` 库已安装。
        -   凸分解可能需要一些时间，具体取决于网格的复杂性。
    -   示例: `-cd` 或 `--convex_decomposition`

-   **`--verbose`** (可选标志)
    -   类型: 无参数 (布尔标志)
    -   默认行为: 不启动预览。
    -   描述: 如果使用此标志，脚本在成功生成 MJCF 文件后，会尝试使用 `mujoco.viewer` 启动一个简单的预览窗口来显示生成的模型。这对于快速检查转换结果非常有用。
        -   **注意**: 此选项需要 `mujoco` Python 包已安装，并且你的系统能够运行 OpenGL 应用。
        -   预览时会生成一个临时的 `_tmp_preview.xml` 文件，其中包含一个简单的场景（地面、灯光和导入的模型）。
    -   示例: `--verbose`

理解这些参数将帮助你根据具体需求定制转换过程，生成适合你仿真任务的 MJCF 模型。

## 6. 使用示例

### 6.1 基本转换
```bash
python scripts/mesh2mjcf.py /path/to/your/model.obj
```

### 6.2 指定颜色
```bash
python scripts/mesh2mjcf.py /path/to/your/model.stl --rgba 0.8 0.2 0.2 1.0
```

### 6.3 使用纹理
```bash
python scripts/mesh2mjcf.py /path/to/your/model.obj --texture /path/to/your/texture.png
```

### 6.4 固定物体
```bash
python scripts/mesh2mjcf.py /path/to/your/model.obj --fix_joint
```

### 6.5 进行凸分解
```bash
python scripts/mesh2mjcf.py /path/to/your/model.obj -cd
```

### 6.6 预览模型
```bash
python scripts/mesh2mjcf.py /path/to/your/model.obj --verbose
```

### 6.7 组合使用
```bash
python scripts/mesh2mjcf.py /path/to/your/model.obj --texture /path/to/texture.png -cd --verbose
```

### 6.8 指定物理属性
```bash
python scripts/mesh2mjcf.py /path/to/your/model.obj --mass 0.5 --diaginertia 0.01 0.01 0.005
```

## 7. 输出文件

脚本会在 DISCOVERSE_ASSETS_DIR 目录下生成以下文件：
- 网格文件：`meshes/obj/{asset_name}/`
- 资源依赖文件：`mjcf/object/{asset_name}_dependencies.xml`
- 物体定义文件：`mjcf/object/{asset_name}.xml`

如果使用了纹理，纹理文件会被复制到：`textures/obj/{asset_name}/`

## 8. 脚本 `mesh2mjcf.py` 详解

本节将详细解释 `scripts/mesh2mjcf.py` 脚本的主要工作流程和代码逻辑。

### 8.1 导入模块与全局变量
- **`os`, `shutil`**: 用于文件和目录操作，如路径处理、文件复制、目录创建和删除。
- **`argparse`**: 用于解析命令行参数，使得脚本可以灵活地接收用户输入。
- **`discoverse.DISCOVERSE_ASSETS_DIR`**: 从 `discoverse` 包导入的全局变量，指定了所有生成的资源（网格、MJCF文件、纹理）应该存放的基础目录。

### 8.2 命令行参数解析 ( `if __name__ == "__main__":` 内 )
脚本使用 `argparse.ArgumentParser` 来定义和解析命令行参数：
- **`input_file`**: (必需) 输入的网格文件路径 (.obj 或 .stl)。
- **`--rgba`**: (可选) 网格的 RGBA 颜色，4个浮点数，默认为 `[0.5, 0.5, 0.5, 1]`。
- **`--texture`**: (可选) 纹理文件路径 (.png)。如果提供，RGBA 的透明度将被忽略 (脚本会将其设为0)。
- **`--mass`**: (可选) 网格质量，默认为 `0.001` kg。
- **`--diaginertia`**: (可选) 网格对角惯性张量，3个浮点数，默认为 `[2e-5, 2e-5, 2e-5]`。
- **`--fix_joint`**: (可选，标志) 是否将物体固定在世界中 (不使用 `free` 关节)。
- **`-cd` / `--convex_decomposition`**: (可选，标志) 是否进行凸分解。需要 `coacd` 和 `trimesh` 库。
- **`--verbose`**: (可选，标志) 是否在转换后使用 MuJoCo 查看器预览。

### 8.3 初始化与设置
- **凸分解依赖检查**: 如果 `convex_decomposition` 为 `True`，脚本会尝试导入 `coacd` 和 `trimesh`。如果导入失败，则打印错误信息并退出。
- **路径与名称处理**: 
    - `asset_name` 从输入文件名中提取 (例如 `/path/to/model.obj` -> `model`)。
    - 检查输入文件类型是否为 `.obj` 或 `.stl`。
- **输出目录准备**: 
    - `output_dir`: `${DISCOVERSE_ASSETS_DIR}/meshes/obj/{asset_name}`，用于存放处理后的网格文件 (原始网格或凸分解后的部件)。
    - `mjcf_obj_dir`: `${DISCOVERSE_ASSETS_DIR}/mjcf/object`，用于存放生成的 MJCF 文件。
    - 如果这些目录已存在，会先删除旧的 `output_dir` 再创建新的，确保输出的纯净性。
    - 原始输入网格文件会被复制到 `output_dir`。

### 8.4 纹理处理
- 如果用户通过 `--texture` 提供了纹理文件且文件存在：
    - `texture_name` 被设置为纹理文件的基本名称。
    - `texture_dir`: 纹理在 MJCF 中引用的相对路径，格式为 `obj/{asset_name}/{texture_name}`。
    - `texture_target_path`: 纹理文件将被复制到的目标完整路径，位于 `${DISCOVERSE_ASSETS_DIR}/textures/obj/{asset_name}/{texture_name}`。
    - 脚本会创建目标纹理目录 (如果不存在) 并复制纹理文件。
    - **重要**: `rgba[3]` (透明度) 被设置为 `0.0`。这是因为在 MuJoCo 中，如果一个材质同时定义了 RGBA 和纹理，纹理会优先显示，但 RGBA 的透明度仍然可能影响整体外观。将其设为0可以避免不必要的混合效果，确保纹理完全显示。
- 如果没有提供纹理，`texture_name` 为 `None`。

### 8.5 生成 MJCF 内容 - 资源文件 (`_dependencies.xml`)
这个文件主要包含 `<asset>` 定义。
- 初始化 `asset_config` 字符串，以 `<mujocoinclude><asset>` 开头。
- **纹理和材质定义** (如果 `texture_name` 不是 `None`):
    - `<texture type="2d" name="{asset_name}_texture" file="{texture_dir}"/>`: 定义一个2D纹理，引用之前处理好的纹理路径。
    - `<material name="{asset_name}_texture" texture="{asset_name}_texture"/>`: 定义一个材质，并将其与上面定义的纹理关联。
- **主网格定义**:
    - `<mesh name="{asset_name}" file="obj/{asset_name}/{os.path.basename(input_file)}"/>`: 定义主物体的网格。`file` 属性指向位于 `meshes/obj/{asset_name}/` 目录下的原始网格文件副本。

### 8.6 生成 MJCF 内容 - 物体文件 (`{asset_name}.xml`)
这个文件定义了物体的 `<body>` 结构。
- 初始化 `geom_config` 字符串，以 `<mujocoinclude>` 开头。
- **关节定义**:
    - 如果 `fix_joint` 为 `False` (默认)，则添加 `<joint type="free"/>`，使物体可以在空间中自由移动和旋转。
- **惯性定义**:
    - `<inertial pos="0 0 0" mass="{mass}" diaginertia="{diaginertia[0]} {diaginertia[1]} {diaginertia[2]}" />`: 使用用户提供的或默认的质量和对角惯性张量值。
- **视觉几何体定义 (主物体)**:
    - 如果使用了纹理 (`texture_name is not None`):
        - `<geom material="{asset_name}_texture" mesh="{asset_name}" class="obj_visual"/>`: 创建一个几何体，使用上面定义的材质 (带纹理) 和主网格。`class="obj_visual"` 使其默认为不参与碰撞的纯视觉物体 (具体行为由预览MJCF中的default class定义)。

### 8.7 凸分解处理 (如果 `convex_decomposition` 为 `True`)
- 打印提示信息。
- 使用 `trimesh.load(input_file, force="mesh")` 加载网格。
- `mesh_coacd = coacd.Mesh(mesh.vertices, mesh.faces)`: 将 trimesh 对象转换为 coacd 所需的格式。
- `parts = coacd.run_coacd(mesh_coacd)`: 执行凸分解，返回一个包含多个凸包部件的列表。每个 `part` 包含顶点和面。
- **遍历每个凸包部件**:
    - `part_filename = f"part_{i}.obj"`: 为每个部件生成文件名。
    - `output_part_file = os.path.join(output_dir, part_filename)`: 部件OBJ文件的完整输出路径。
    - `part_mesh = trimesh.Trimesh(vertices=part[0], faces=part[1])`: 将部件的顶点和面转换回 trimesh 对象。
    - `part_mesh.export(output_part_file)`: 将该凸包部件导出为一个新的 `.obj` 文件，保存在 `output_dir`。
    - **更新资源文件 (`asset_config`)**: 
        - `asset_config += '    <mesh name="{}_part_{}" file="obj/{}/{}"/>\n'.format(asset_name, i, asset_name, part_filename)`: 为每个凸包部件在 `_dependencies.xml` 中添加一个新的 `<mesh>` 定义，引用其对应的 `part_i.obj` 文件。
    - **更新物体文件 (`geom_config`)**:
        - `geom_config += '  <geom type="mesh" rgba="{} {} {} {}" mesh="{}_part_{}"/>\n'.format(rgba[0], rgba[1], rgba[2], rgba[3], asset_name, i)`: 为每个凸包部件在 `{asset_name}.xml` 中添加一个 `<geom>`。这些几何体将用于碰撞。它们使用用户提供的 RGBA 颜色。如果主物体使用了纹理，这些碰撞部件依然使用 RGBA 颜色，因为它们是独立的几何体。脚本当前设置 `rgba[3]=0` 当有纹理时，这可能会导致凸分解部件在有纹理时不可见，除非预览器或主MJCF另有设定，或者这里的 `rgba` 是指原始命令行参数的副本。
        *注意：原始脚本在这里有个 `if texture_name is None: pass` 的逻辑，似乎未完全处理有纹理时的凸分解部件的视觉问题，凸分解的部件总是使用 RGBA。主视觉网格使用纹理，碰撞部件使用 RGBA。*
- 打印分解结果信息。

### 8.8 非凸分解处理 (如果 `convex_decomposition` 为 `False`)
- 如果没有进行凸分解，并且未使用纹理 (`texture_name is None`):
    - `geom_config += '  <geom type="mesh" rgba="{} {} {} {}" mesh="{}"/>\n'.format(rgba[0], rgba[1], rgba[2], rgba[3], asset_name)`: 添加一个使用原始网格和 RGBA 颜色的 `<geom>`。这个 `<geom>` 既作为视觉也作为碰撞体（除非在主MJCF中被 `class="obj_visual"` 覆盖其碰撞属性，且没有其他专门的碰撞geom）。

### 8.9 完成并写入 MJCF 文件
- **资源文件**: `asset_config += '  </asset>\n</mujocoinclude>\n'` 添加闭合标签。
- `asset_file_path = os.path.join(mjcf_obj_dir, f"{asset_name}_dependencies.xml")`: 确定资源文件的最终路径。
- 将 `asset_config` 内容写入该文件。
- **物体文件**: `geom_config += '</mujocoinclude>\n'` 添加闭合标签。
- `geom_file_path = os.path.join(mjcf_obj_dir, f"{asset_name}.xml")`: 确定物体定义文件的最终路径。
- 将 `geom_config` 内容写入该文件。
- 打印成功信息和文件路径。

### 8.10 可视化预览 (如果 `verbose` 为 `True`)
- 打印启动查看器信息。
- **查找 Python 解释器**: 使用 `shutil.which('python3')` 或 `shutil.which('python')` 查找可用的 Python 解释器路径。
- **创建临时预览 MJCF 文件 (`_tmp_preview.xml`)**: 
    - 该文件位于 `${DISCOVERSE_ASSETS_DIR}/mjcf/_tmp_preview.xml`。
    - 文件内容是一个完整的 MJCF 场景定义，包括：
        - `<mujoco model="temp_preview_env">`
        - `<option gravity="0 0 -9.81"/>`
        - `<compiler meshdir="../meshes" texturedir="../textures/"/>`: **重要**，指定了相对路径，让 MuJoCo 能找到 `_dependencies.xml` 中引用的网格和纹理文件。`../meshes` 指向 `${DISCOVERSE_ASSETS_DIR}/meshes`，`../textures/` 指向 `${DISCOVERSE_ASSETS_DIR}/textures`。
        - `<include file="object/{asset_name}_dependencies.xml"/>`: 包含生成的资源文件。
        - `<default class="obj_visual"> <geom group="2" type="mesh" contype="0" conaffinity="0"/> </default>`: 定义了一个默认类 `obj_visual`。属于此类的几何体将不参与碰撞 (`contype="0" conaffinity="0"`)并属于视觉组2。这是为了确保如果主物体有纹理并使用了 `class="obj_visual"`，它只是视觉上的；如果进行了凸分解，则那些凸包部件（没有指定class）会作为碰撞体。
        - `<worldbody>`: 包含一个地面 (`<geom name="floor" .../>`) 和一个灯光 (`<light .../>`)。
        - `<body name="{asset_name}" pos="0 0 0.5"> <include file="object/{asset_name}.xml"/> </body>`: 创建一个物体，将其放置在 `(0,0,0.5)` 的位置，并包含生成的 `{asset_name}.xml` 文件内容（即关节、惯性和几何体定义）。
- **执行 MuJoCo 查看器**: 
    - `cmd_line = f"{py_dir} -m mujoco.viewer --mjcf={tmp_world_mjcf}"`: 构建命令行。
    - `os.system(cmd_line)`: 执行命令启动查看器。
- **清理**: `os.remove(tmp_world_mjcf)` 删除临时预览文件。

### 8.11 总结
该脚本通过组合字符串来动态生成 XML 内容，然后将其写入 `.xml` 文件。它区分了资源定义和物体结构定义，并通过 MuJoCo 的 `<include>`机制将它们模块化。凸分解功能依赖外部库，并为每个凸包部件生成独立的网格文件和 MJCF 中的 `<mesh>`及`<geom>`条目。预览功能通过创建一个临时的、包含完整场景的 MJCF 文件来实现。
