## 命令

```bash
python3 policies/train.py <policy> [args]
```

解释：
- `policy`: 位置参数，指定策略的类型，目前支持的选项：act
- `[args]`: 不同的策略有不同的命令行参数，请参考下面对应策略的说明

## act

### 依赖安装

```bash
pip install -r policies/act/requirements/train_eval.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 训练配置
参考的训练配置文件位于`policies/act/configurations/task_configs/example_task.py`中，其中主要参数解释如下：
- `camera_names`: 训练数据中相机的序号
- `state_dim`: 训练数据中观测向量的维度
- `action_dim`: 训练数据中动作向量的维度
- `batch_size_train`: 训练时的batch_size大小
- `batch_size_validate`: 验证时的batch_size大小
- `chunk_size`: 单次预测的动作数量
- `num_epochs`: 训练的总步数
- `learning_rate`: 学习率

训练特定任务时，需要复制一份配置文件并重命名为任务名，后续将通过任务名索引相关配置文件。


### 数据集位置
仿真采集的数据默认位于discoverse仓库根目录的data文件夹中，而训练时默认从policies/act/data/hdf5中寻找数据。因此，建议使用软连接的方式将前者链接到后者，命令如下（注意修改命令中的路径，并且需要绝对路径）：

```bash
ln -sf /absolute/path/to/discoverse/data /absolute/path/to/discoverse/policies/act/data
```

### 训练命令

```bash
python3 policies/train.py act -tn <task_name>
```

其中`-tn`参数指定任务名，程序会根据任务名分别在`task_configs`和`act/data/hdf5`目录下寻找同名的配置文件和数据集。

### 训练结果

训练结果保存在`policies/act/my_ckpt`目录下。