import os
import argparse
import numpy as np
import time
from datetime import datetime
from discoverse import DISCOVERSE_ROOT_DIR
from env import Env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from sbx import PPO
from tqdm import tqdm

def make_env(render=True, seed=0):
    """创建环境的工厂函数
    
    Args:
        render (bool): 是否渲染环境
        seed (int): 随机种子
        
    Returns:
        callable: 创建环境的函数
    """

    def _init():
        try:
            env = Env(render=render)
            return env
        except Exception as e:
            print(f"环境创建失败: {str(e)}")
            raise e

    return _init


def train(render=True, seed=42, total_timesteps=1000000, batch_size=64, n_steps=2048, 
learning_rate=3e-4, log_dir=None, model_path=None, log_interval=10):
    """训练PPO模型
    
    Args:
        render (bool): 是否渲染环境
        seed (int): 随机种子
        total_timesteps (int): 总训练步数
        batch_size (int): 批次大小
        n_steps (int): 每次更新所收集的轨迹长度
        learning_rate (float): 学习率
        log_dir (str): 日志目录，如果为None则使用默认目录
        model_path (str): 预训练模型路径，用于继续训练
        log_interval (int): 日志记录间隔
    """

    # 设置日志目录
    if log_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = os.path.join(DISCOVERSE_ROOT_DIR, f"data/PPO_State/logs_{timestamp}")
    
    os.makedirs(log_dir, exist_ok=True)
    print(f"日志目录: {log_dir}")

    try:
        print("开始创建环境...")
        # 创建训练环境
        env = make_env(render=render, seed=seed)()
        # 使用DummyVecEnv包装单个环境
        env = DummyVecEnv([lambda: env])
        print("环境创建完成")

        # 自定义进度条回调
        class TqdmCallback(BaseCallback):
            def __init__(self, total_timesteps, verbose=0):
                super(TqdmCallback, self).__init__(verbose)
                self.pbar = None
                self.total_timesteps = total_timesteps
                self.start_time = None
                
            def _on_training_start(self):
                self.start_time = time.time()
                self.pbar = tqdm(total=self.total_timesteps, desc="训练进度")

            def _on_step(self):
                self.pbar.update(1)
                # 更新进度条描述，显示已用时间
                elapsed_time = time.time() - self.start_time
                hours, remainder = divmod(elapsed_time, 3600)
                minutes, seconds = divmod(remainder, 60)
                self.pbar.set_description(
                    f"训练进度 [{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}]"
                )
                return True
                
            def _on_training_end(self):
                self.pbar.close()
                self.pbar = None

        # 创建PPO模型或加载预训练模型
        if model_path is not None and os.path.exists(model_path):
            print(f"加载预训练模型: {model_path}")
            model = PPO.load(model_path, env=env)
            # 更新学习率
            model.learning_rate = learning_rate
            print("预训练模型加载完成")
        else:
            print("创建新的PPO模型")
            model = PPO(
                "MlpPolicy",
                env,
                n_steps=n_steps,  # 每次更新所收集的轨迹长度
                batch_size=batch_size,  # 批次大小
                n_epochs=10,  # 每次更新迭代次数
                gamma=0.99,  # 折扣因子
                learning_rate=learning_rate,  # 学习率
                clip_range=0.2,  # PPO策略裁剪范围
                ent_coef=0.01,  # 熵正则化系数
                tensorboard_log=log_dir,
                verbose=1  # 输出详细程度
            )
        
        print("PPO模型创建完成，开始收集经验...")

        # 训练模型
        print(f"开始训练模型，总时间步数: {total_timesteps}")
        model.learn(
            total_timesteps=total_timesteps,
            callback=TqdmCallback(total_timesteps=total_timesteps),
            log_interval=log_interval,  # 每log_interval次更新后记录日志
        )

        # 保存最终模型
        save_path = os.path.join(log_dir, "final_model")
        model.save(save_path)
        print(f"模型已保存到: {save_path}")

    except Exception as e:
        print(f"训练过程发生错误: {str(e)}")
        raise e
    finally:
        if 'env' in locals():
            env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPO强化学习训练脚本")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--render", action="store_true", default=False, help="在训练过程中显示渲染画面 (默认: True)")
    parser.add_argument("--total_timesteps", type=int, default=1000000, help="总训练步数")
    parser.add_argument("--batch_size", type=int, default=2, help="批次大小")
    parser.add_argument("--n_steps", type=int, default=2048, help="每次更新所收集的轨迹长度")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="学习率")
    parser.add_argument("--log_dir", type=str, default=None, help="日志目录")
    parser.add_argument("--model_path", type=str, default=None, help="预训练模型路径，用于继续训练")
    parser.add_argument("--log_interval", type=int, default=10, help="日志记录间隔")
    
    args = parser.parse_args()

    train(
        render=args.render,
        seed=args.seed,
        total_timesteps=args.total_timesteps,
        batch_size=args.batch_size,
        n_steps=args.n_steps,
        learning_rate=args.learning_rate,
        log_dir=args.log_dir,
        model_path=args.model_path,
        log_interval=args.log_interval
    )
