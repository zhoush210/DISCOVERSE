# MMK2命令
1. 后台运行
```
nohup <python train.py> &
```
2. 生成数据
```
nohup python3 discoverse/examples/tasks_mmk2/cabinet_door_open.py --data_idx 0 --data_set_size 10 &
```
3. 转化数据
```
python3 policies/act/data_process/raw_to_hdf5.py -md mujoco -dir ./data -tn <mmk2_task> -vn cam_0 cam_1 cam_2
```
4. 复制数据
```
mkdir -p policies/act/data/hdf5
```
```
cp -r data/hdf5/* policies/act/data/hdf5 
```
5. 训练
```
nohup python3 policies/train.py act -tn <mmk2_task> &
```
6. 推理
```
python3 policies/infer.py act -tn <mmk2_task> -mts <max_timesteps> -ts <checkpoint> -rn discoverse/examples/tasks_mmk2/cabinet_door_open.py
```