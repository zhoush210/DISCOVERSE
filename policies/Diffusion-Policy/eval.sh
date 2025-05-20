DEBUG=False

task_name=${1}
checkpoint=${2}
gpu_id=${3:-0}  # 如果未提供第3个参数，则默认值为0

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=${gpu_id}

python3 eval.py "$task_name" "$checkpoint"