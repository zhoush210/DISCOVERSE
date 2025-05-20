# 
task_name=${1}
seed=${2:-0}
gpu_id=${3:-0}

DEBUG=False
save_ckpt=True

alg_name=robot_dp
# task choices: See TASK.md
config_name=${alg_name}
addition_info=train
exp_name=${task_name}-robot_dp-${addition_info}
run_dir="data/outputs/${exp_name}_seed${seed}"

echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

wandb_mode=offline

export HYDRA_FULL_ERROR=1 
export CUDA_VISIBLE_DEVICES=${gpu_id}

python train.py --config-name=${config_name}.yaml \
                            task.name=${task_name} \
                            task.dataset.zarr_path="data/${task_name}.zarr" \
                            training.debug=$DEBUG \
                            training.seed=${seed} \
                            training.device="cuda:0" \
                            exp_name=${exp_name} \
                            logging.mode=${wandb_mode} \
                            # checkpoint.save_ckpt=${save_ckpt}
                            # hydra.run.dir=${run_dir} \