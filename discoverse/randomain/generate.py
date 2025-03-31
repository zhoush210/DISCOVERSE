import cv2
import shutil
import random
import argparse
from concurrent.futures import ThreadPoolExecutor
from params_proto.hyper import Sweep
import numpy as np

from FlowCompute.flowcompute import FlowCompute
from discoverse.randomain.utils import pick, read_frame
from discoverse.randomain.workflow import ImageGen

import os
import sys
from discoverse import DISCOVERSE_ROOT_DIR

def parse_args():
    parser = argparse.ArgumentParser(description="Generate images")

    parser.add_argument('--seed', type=int, default=26, help="Random seed for picking the prompt")
    parser.add_argument('--device', type=str, default='cuda', help="cpu or gpu")
    parser.add_argument('--gpu_id', type=int, default=0, help="Specific GPU device ID to use")

    # basic params
    parser.add_argument('--task_name', type=str, default='block_place', help="Task name")
    parser.add_argument('--work_dir', type=str, default=None, help="Working directory. You can specify a range like '000-100'. If None, process all dirs in segment/")
    parser.add_argument('--cam_id', type=int, default=0, help="Camera ID")
    parser.add_argument('--cam_ids', type=int, nargs='+', default=None, help="Multiple camera IDs to process")
    parser.add_argument('--fore_objs', type=str, nargs='+', default=['block_green', 'bowl_pink', 'robot'], help="List of foreground objects")
    
    # controlnet
    parser.add_argument('--width', type=int, default=640, help="Image width")
    parser.add_argument('--height', type=int, default=480, help="Image height")
    parser.add_argument('--num_steps', type=int, default=1, help="Number of steps for image generation")
    parser.add_argument('--denoising_strength', type=float, default=1.0, help="Denoising strength")
    parser.add_argument('--control_strength', type=float, default=0.8, help="Control strength")
    parser.add_argument('--grow_mask_amount', type=int, default=0, help="Grow mask amount")
    parser.add_argument('--fore_grow_mask_amount', type=int, default=0, help="Foreground grow mask amount")
    parser.add_argument('--background_strength', type=float, default=0.2, help="Background strength")
    parser.add_argument('--fore_strength', type=float, default=2.0, help="Foreground strength")
    
    # flow compute
    parser.add_argument('--flow_interval', type=int, default=3, help="Frame interval for flow computation, set 1 for no flow interval")
    parser.add_argument('--flow_method', type=str, default="raft", help="Method for flow computation (rgb or raft)")
    parser.add_argument('--raft_model_path', type=str, default=DISCOVERSE_ROOT_DIR+"/discoverse/randomain/models/flow/raft-small.pth", help="Path to RAFT model checkpoint")
    parser.add_argument('--raft_small', default=True, help="Use small RAFT model")
    parser.add_argument('--raft_mixed_precision', default=False, help="Use mixed precision for RAFT")
    parser.add_argument('--raft_alternate_corr', default=False, help="Use efficient correlation implementation for RAFT")

    return parser.parse_args()

def main(args):
    random.seed(args.seed)
    imagen = ImageGen()

    control_parameters = pick(
        vars(args),
        "width",
        "height",
        "num_steps",
        "denoising_strength",
        "control_strength",
        "grow_mask_amount",
        "fore_grow_mask_amount",
        "background_strength",
        "fore_strength",
    )
    
    input_keys = ['cam', 'depth', 'background'] + args.fore_objs
    work_dir = os.path.join(DISCOVERSE_ROOT_DIR, f"data/{args.task_name}/segment/{args.work_dir}/{args.cam_id}")
    input_paths = {k: f'{work_dir}/{k}.mp4' for k in input_keys}
    
    # 修改输出路径，统一到output/work_dir目录下，并以cam_id命名
    output_dir = os.path.join(DISCOVERSE_ROOT_DIR, f"data/{args.task_name}/output/{args.work_dir}")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{args.cam_id}.mp4")

    prompt_path = os.path.join(DISCOVERSE_ROOT_DIR, f'discoverse/randomain/prompts/{args.task_name}/prompts.jsonl')
    prompts = Sweep.read(prompt_path)
    
    # 选择第一个提示中的前景对象提示作为固定提示
    first_prompt = prompts[0]
    fixed_fore_objs_prompt = {obj: first_prompt.get(obj, "") for obj in args.fore_objs}
    fixed_negative_prompt = first_prompt.get("negative", "")
    
    # input
    caps = {}
    for key, path in input_paths.items():
        caps[key] = cv2.VideoCapture(path)
        if not caps[key].isOpened():
            raise ValueError(f"cannot open video: {path}")

    fps = caps['cam'].get(cv2.CAP_PROP_FPS)
    frame_width = int(caps['cam'].get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(caps['cam'].get(cv2.CAP_PROP_FRAME_HEIGHT))

    # output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    if os.path.exists(output_path):
        os.remove(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    # 读取视频
    print("--------Reading Video--------")
    frames_list = [] 
    finish = False

    with ThreadPoolExecutor() as executor:
        while True:
            frames = {}
            futures = {executor.submit(read_frame, key, cap): key for key, cap in caps.items()}
            for future in futures:
                key, frame = future.result()
                if frame is None:
                    finish = True
                else:
                    frames[key] = frame

            if finish:
                break
            frames_list.append(frames)
    
    # 生成
    Flow = FlowCompute(method=args.flow_method, model=args.raft_model_path,
                       small=args.raft_small, mixed_precision=args.raft_mixed_precision, alternate_corr=args.raft_alternate_corr, 
                       device=args.device)
    print("--------Generating--------")
    
    gen_list = []
    for i, frames in enumerate(frames_list):
        print(f"{i}/{len(frames_list)} frames")
        if i % args.flow_interval == 0:
            # 只随机化背景提示
            random_prompt = prompts[random.randint(0, len(prompts) - 1)]
            background_prompt = random_prompt.get("background", "")
            
            # 构建提示，前景对象使用固定提示，背景使用随机提示
            prompt = {
                "background": background_prompt,
                "negative": "",
            }
            prompt.update(fixed_fore_objs_prompt)
            
            # 生成图像，但仍然需要提供所有的掩码
            gen_image = imagen.generate(
                depth=frames['depth'],
                masks={k:frames[k] for k in (args.fore_objs+['background'])},
                prompt=prompt,
                **control_parameters,
            )
            
            # 创建一个背景掩码的反向掩码，用于提取前景
            # 确保背景掩码是单通道格式
            if len(frames['background'].shape) == 3:
                background_mask = cv2.cvtColor(frames['background'], cv2.COLOR_BGR2GRAY)
            else:
                background_mask = frames['background'].copy()
            
            # 确保背景掩码为8位无符号整数格式
            background_mask = background_mask.astype(np.uint8)
            
            # 创建前景掩码（背景掩码的反向）
            foreground_mask = cv2.bitwise_not(background_mask)
            
            # 从原始视频帧中提取前景
            original_frame = frames['cam']
            
            # 将生成的图像的背景部分与原始前景合并
            # 仅保留生成图像的背景部分
            gen = cv2.bitwise_and(gen_image, gen_image, mask=background_mask)
            # 提取原始帧的前景部分
            foreground = cv2.bitwise_and(original_frame, original_frame, mask=foreground_mask)
            # 合并背景和前景
            gen = cv2.add(gen, foreground)
        else:
            flow = Flow.compute(frames_list[i-1]['cam'], frames['cam'])
            gen = Flow.warp_forward(gen_list[-1], flow)

        gen_list.append(gen)
    
    # 保存
    for gen in gen_list:
        out.write(gen)
    
    # 结束
    for key, cap in caps.items():
        cap.release()
        out.release()
        


# image_foreDR = DRmerge(image, rgb, [block_mask, bowl_mask, robot_mask], [background_mask])
# image_backDR = DRmerge(image, rgb, [background_mask], [block_mask, bowl_mask, robot_mask])

if __name__ == "__main__":
    args = parse_args()
    
    # 设置使用的GPU
    if args.device == 'cuda':
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
        print(f"Using GPU device ID: {args.gpu_id}")
    
    # 确定要处理的工作目录
    print("work_dir:",args.work_dir)
    if args.work_dir is None:
        # 如果未指定work_dir，则处理segment目录下的所有子目录
        segment_path = os.path.join(DISCOVERSE_ROOT_DIR, f"data/{args.task_name}/segment/")
        work_dirs = [d for d in os.listdir(segment_path) if os.path.isdir(os.path.join(segment_path, d))]
    elif "-" in args.work_dir:
        # 如果work_dir包含"-"，则解析为范围
        segment_path = os.path.join(DISCOVERSE_ROOT_DIR, f"data/{args.task_name}/segment/")
        try:
            range_start, range_end = args.work_dir.split("-")
            # 确保范围的开始和结束是三位数的格式
            if len(range_start) < 3:
                range_start = range_start.zfill(3)
            if len(range_end) < 3:
                range_end = range_end.zfill(3)
                
            # 获取segment目录下的所有工作目录
            all_dirs = [d for d in os.listdir(segment_path) if os.path.isdir(os.path.join(segment_path, d))]
            # 筛选出在指定范围内的目录
            work_dirs = [d for d in all_dirs if range_start <= d <= range_end]
            print(f"Processing work directories in range {range_start}-{range_end}, found {len(work_dirs)} directories")
        except Exception as e:
            print(f"Error parsing work_dir range: {args.work_dir}, error: {str(e)}")
            print("Falling back to using work_dir as a single directory")
            work_dirs = [args.work_dir]
    else:
        # 如果指定了work_dir，则只处理该目录
        work_dirs = [args.work_dir]
    
    # 确定要处理的相机ID
    if args.cam_ids is not None:
        cam_ids = args.cam_ids
    else:
        cam_ids = [args.cam_id]
    
    # 双重循环：对每个工作目录的每个相机ID都进行处理
    for work_dir in work_dirs:
        for cam_id in cam_ids:
            print(f"Processing work_dir: {work_dir}, camera ID: {cam_id}")
            
            # 创建新的参数对象
            current_args = argparse.Namespace(**vars(args))
            current_args.work_dir = work_dir
            current_args.cam_id = cam_id
            
            try:
                main(current_args)
            except Exception as e:
                print(f"Error processing work_dir: {work_dir}, camera ID: {cam_id}")
                print(f"Error message: {str(e)}")
                # 继续处理下一个，而不是中断整个程序
                continue  
