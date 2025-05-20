import torch  
import os
import hydra
from pathlib import Path

import yaml
from datetime import datetime
import dill
from argparse import ArgumentParser
from diffusion_policy.workspace.robotworkspace import RobotWorkspace
from diffusion_policy.env_runner.dp_runner import DPRunner

from diffusion_policy.env import Env
import datetime

def get_policy(checkpoint, output_dir, device):    
    # load checkpoint
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace: RobotWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    
    # get policy from workspace
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    
    device = torch.device(device)
    policy.to(device)
    policy.eval()

    return policy

class DP:
    def __init__(self, task_name, checkpoint: str):
        self.policy = get_policy(f'checkpoints/{task_name}/{checkpoint}.ckpt', None, 'cuda:0')
        self.runner = DPRunner(output_dir=None)

    def update_obs(self, observation):
        self.runner.update_obs(observation)
    
    def get_action(self, observation=None):
        action = self.runner.get_action(self.policy, observation)
        return action

    def get_last_obs(self):
        return self.runner.obs[-1]

def eval(args, env, dp, save_dir):
    success_sum = 0
    for i in range(args["eval_num"]): 
        obs, t = env.reset()

        while t < args["max_steps"]:
            print("step:",t,"/",args["max_steps"])
            dp.update_obs(obs)
            actions = dp.get_action()
            obs, success, num = env.step(actions)
            t += num
            if success:
                success_sum += 1
                break
        print("success rate:",success_sum,"/",i+1,"=",success_sum/(i+1))

        # save video
        import mediapy
        for id in env.cfg.obs_rgb_cam_id:
            mediapy.write_video(os.path.join(save_dir, f"{i}_cam_{id}.mp4"), [videos[id] for videos in env.video_list], fps=env.cfg.render_set["fps"])
  
def main(usr_args):
    task_name = usr_args.task_name
    checkpoint = usr_args.checkpoint

    with open(f'./task_config/{task_name}.yml', 'r', encoding='utf-8') as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    env = Env(args)

    dp = DP(task_name, checkpoint)

    current_time = datetime.datetime.now().strftime("%y%m%d%H%M")
    save_dir = Path(f'eval_result/{task_name}/{current_time}')
    save_dir.mkdir(parents=True, exist_ok=True)

    eval(args, env, dp, save_dir)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('task_name', type=str, default='block_hammer_beat')
    parser.add_argument('checkpoint', type=str, default="note_1000")
    usr_args = parser.parse_args()
    
    main(usr_args)
