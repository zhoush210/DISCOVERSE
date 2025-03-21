3dgs模型的后缀为.ply，部分mesh模型的后缀也是.ply，区分一个.ply模型是否是3dgs模型，需要看文件的头部信息，通常一个3dgs模型的头部信息为：
```
ply
format binary_little_endian 1.0
element vertex 1000
property float x
property float y
property float z
property float nx
property float ny
property float nz
property float f_dc_0
property float f_dc_1
property float f_dc_2
property float f_rest_0
......
property float opacity
property float scale_0
property float scale_1
property float scale_2
property float rot_0
property float rot_1
property float rot_2
property float rot_3
end_header
```
判断其是否包含 `f_dc_` 字段，若包含，则为3dgs模型。

3dgs模型在线编辑：[3dgs在线编辑](https://superspl.at/editor)，支持的功能包括：
1. 平移
2. 旋转
3. 缩放
4. 选中点云（不同选择方式）
5. 删除选中点云
6. 不同渲染模式

本仓库提供了3dgs模型的编辑脚本，可以定量平移、旋转、缩放模型。
脚本位于：`scripts/gaussainSplattingTranspose.py`

如果想让模型沿着x轴平移 0.3 米，y轴平移0.4米，z轴平移-0.5米，沿着z轴顺时针旋转 45 度，再将尺寸缩放到原先的1.5倍，模型输入路径是：`data/000000.ply`，模型输出路径是：`data/000000_transpose.ply`，则命令如下（操作顺序是先旋转再平移，最后缩放）：

```python
# 沿着z轴旋转45度，对应的四元数为(xyzw顺序)：
>>> from scipy.spatial.transform import Rotation
>>> r = Rotation.from_euler('z', 45, degrees=True)
>>> print(r.as_quat())
[0., 0., 0.38268343, 0.92387953]
```

```bash
python scripts/gaussainSplattingTranspose.py --input_path data/000000.ply --output_path data/000000_transpose.ply --translation 0.3 0.4 -0.5 --rotation 0. 0. 0.38268343 0.92387953 --scale 1.5
```
