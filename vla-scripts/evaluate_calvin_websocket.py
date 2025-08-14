#!/usr/bin/env python3
"""
  # 先启动服务器
  python evaluation_server.py --generalist_path openvla7b --specialist_path specialist_policy.pt

  # 然后使用统一入口的客户端模式
  python evaluate_calvin_websocket.py --mode websocket-client --server_url ws://localhost:8765
  
  # 本地模式，不需要分离
  python evaluate_calvin_websocket.py --mode local --generalist_path openvla7b --specialist_path specialist_policy.pt
"""
"""Code to evaluate Calvin with WebSocket server-client support."""

import os
os.environ["CALVIN_ROOT"] = "/pfs/pfs-uaDOJM/home/xiongxiao/workspace/RoboDual/calvin"
CALVIN_ROOT = os.environ['CALVIN_ROOT']
# 取消设置http_proxy和https_proxy环境变量
if 'http_proxy' in os.environ:
    del os.environ['http_proxy']
if 'https_proxy' in os.environ:
    del os.environ['https_proxy']

import argparse
import json
import logging
import os
from pathlib import Path
import sys
import time
import copy
from moviepy.editor import ImageSequenceClip
from datetime import timedelta

# This is for using the locally installed repo clone when using slurm
from calvin_agent.models.calvin_base_model import CalvinBaseModel

sys.path.insert(0, Path(__file__).absolute().parents[2].as_posix())

from calvin_agent.evaluation.multistep_sequences import get_sequences
from calvin_agent.evaluation.utils import (
    count_success,
    get_env_state_for_initial_condition,
    get_log_dir,
)
import hydra
import numpy as np
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from termcolor import colored
import torch
from tqdm.auto import tqdm

from dual_sys_evaluation import DualSystemCalvinEvaluation

from ema_pytorch import EMA
from transformers.modeling_outputs import CausalLMOutputWithPast

logger = logging.getLogger(__name__)

os.environ["FFMPEG_BINARY"] = "auto-detect"
CALVIN_ROOT = os.environ['CALVIN_ROOT']

from collections import Counter
import json
import numpy as np

# Import functions from original evaluate_calvin.py
from evaluate_calvin import print_and_save, make_env, evaluate_sequence, rollout

def main(args):
    # Set seed #42
    seed_everything(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.mode == 'websocket-client':
        # WebSocket client mode: import and run WebSocket client
        from evaluation_client import evaluate_policy_client
        
        save_path = '../evaluation_results'
        observation_space = {
            'rgb_obs': ['rgb_static', 'rgb_gripper'],
            'depth_obs': ['depth_static', 'depth_gripper'], 
            'state_obs': ['robot_obs'], 
            'actions': ['rel_actions'], 
            'language': ['language']
        }
        
        eval_dir = save_path + f'/eval{torch.cuda.current_device()}/'
        os.makedirs(eval_dir, exist_ok=True)
        env = make_env(os.path.join(CALVIN_ROOT, 'dataset/calvin_debug_dataset'), observation_space, device)
        
        avg_reward = torch.tensor(evaluate_policy_client(
            args.server_url,
            env,
            save_path + 'success_rate.txt', 
            save_path + 'result.txt', 
            eval_dir=eval_dir,
            ep_len=360,
            num_sequences=1000,
            enrich_lang=args.enrich_lang,
            debug=False,
        )).float().mean().to(device)
        
    elif args.mode == 'websocket-server':
        print("WebSocket server mode detected. Please run evaluation_server_ws.py instead.")
        print(f"Example: python evaluation_server_ws.py --generalist_path {args.generalist_path} --specialist_path {args.specialist_path}")
        return
        
    elif args.mode == 'server-client':
        # HTTP REST API client mode (legacy)
        from evaluation_client import evaluate_policy_client
        
        save_path = '../evaluation_results'
        observation_space = {
            'rgb_obs': ['rgb_static', 'rgb_gripper'],
            'depth_obs': ['depth_static', 'depth_gripper'], 
            'state_obs': ['robot_obs'], 
            'actions': ['rel_actions'], 
            'language': ['language']
        }
        
        eval_dir = save_path + f'/eval{torch.cuda.current_device()}/'
        os.makedirs(eval_dir, exist_ok=True)
        env = make_env(os.path.join(CALVIN_ROOT, 'dataset/calvin_debug_dataset'), observation_space, device)
        
        avg_reward = torch.tensor(evaluate_policy_client(
            args.server_url,
            env,
            save_path + 'success_rate.txt', 
            save_path + 'result.txt', 
            eval_dir=eval_dir,
            ep_len=360,
            num_sequences=1000,
            enrich_lang=args.enrich_lang,
            debug=False,
        )).float().mean().to(device)
        
    elif args.mode == 'server':
        print("HTTP REST server mode detected. Please run evaluation_server.py instead.")
        print(f"Example: python evaluation_server.py --generalist_path {args.generalist_path} --specialist_path {args.specialist_path}")
        return
        
    else:
        # Local mode (original implementation)
        # Load generalist policy
        from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
        quantization_config = None
        processor = AutoProcessor.from_pretrained(args.generalist_path, trust_remote_code=True)
        model = AutoModelForVision2Seq.from_pretrained(
                args.generalist_path,
                torch_dtype=torch.bfloat16,
                quantization_config=quantization_config,
                low_cpu_mem_usage=False,
                trust_remote_code=True,
            )
        model.eval()

        # Load specialist policy
        from prismatic.models.policy.diffusion_policy import DiffusionDiTImagePolicy
        from diffusers.schedulers.scheduling_ddim import DDIMScheduler
        from diffusers.schedulers import DPMSolverMultistepScheduler

        scheduler = DDIMScheduler( num_train_timesteps = 100, beta_schedule = 'squaredcos_cap_v2', prediction_type="epsilon" )
        shape_meta = {'action' : {'shape': [7]}}
        diffusion_policy = DiffusionDiTImagePolicy( shape_meta = shape_meta,
                                                    noise_scheduler = scheduler,
                                                    n_action_steps=8, 
                                                    num_inference_steps=10,
                                                    vision_encoder='DINO',
                                                    with_depth=args.with_depth,
                                                    progressive_noise=False,
                                                    with_gripper=args.with_gripper,
                                                    with_tactile=args.with_tactile,
                                                    cond_drop_chance=0.1 if args.with_cfg else 0.,  
                                                    # set cond_drop_chance > 0 to activate CFG
                                                  ).eval()
       

        from openvla.prismatic.vla.action_tokenizer import ActionTokenizer
        action_tokenizer = ActionTokenizer(processor.tokenizer)

        from train_spacialist_calvin import DualSystem
        dual_sys = DualSystem(model, diffusion_policy, action_tokenizer)
        dual_sys.ema_fast_system.load_state_dict(torch.load(args.specialist_path), strict=False)

        dual_sys = dual_sys.to(device)

        save_path = '../evaluation_results'
        observation_space = {
            'rgb_obs': ['rgb_static', 'rgb_gripper', ],  # rgb_tactile
            'depth_obs': ['depth_static', 'depth_gripper'], 
            'state_obs': ['robot_obs'], 
            'actions': ['rel_actions'], 
            'language': ['language']}
        eval_dir = save_path + f'/eval{torch.cuda.current_device()}/'
        os.makedirs(eval_dir, exist_ok=True)
        env = make_env(os.path.join(CALVIN_ROOT, 'dataset/calvin_debug_dataset'), observation_space, device)
        eva = DualSystemCalvinEvaluation(dual_sys, processor, action_tokenizer)
        dual_sys.eval()
        
        # Import the original evaluate_policy function
        from evaluate_calvin import evaluate_policy
        
        avg_reward = torch.tensor(evaluate_policy(
            eva, 
            env,
            save_path+'success_rate.txt', 
            save_path+'result.txt', 
            1,  # num_procs
            0,  # procs_id
            eval_dir = eval_dir,
            ep_len = 360,
            num_sequences = 1000,
            enrich_lang=args.enrich_lang,
            debug = False,
        )).float().mean().to(device)

    print('average success rate ', avg_reward)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="local", 
                        choices=["local", "server", "server-client", "websocket-server", "websocket-client"], 
                        help="Evaluation mode: local (original), server (HTTP REST server), server-client (HTTP REST client), websocket-server (WebSocket server), websocket-client (WebSocket client)")
    parser.add_argument("--server_url", default="ws://localhost:8765", type=str, 
                        help="Server URL for client modes (ws://host:port for WebSocket, http://host:port for REST)")
    parser.add_argument("--generalist_path", default="openvla7b", type=str)
    parser.add_argument("--specialist_path", default="specialist_policy.pt", type=str)
    parser.add_argument("--calvin_path", default="./calvin", type=str)
    parser.add_argument("--log_dir", default="CALVIN_ABC-D", type=str)
    parser.add_argument("--with_depth", default=True, action="store_true")
    parser.add_argument("--with_gripper", default=True, action="store_true")
    parser.add_argument("--with_tactile", default=False, action="store_true")
    parser.add_argument("--with_cfg", default=False, action="store_true")
    parser.add_argument("--enrich_lang", default=False, action="store_true")
    args = parser.parse_args()

    main(args)