#!/usr/bin/env python3

import os
os.environ["CALVIN_ROOT"] = "/pfs/pfs-uaDOJM/home/xiongxiao/workspace/RoboDual/calvin"
CALVIN_ROOT = os.environ['CALVIN_ROOT']
# 取消设置http_proxy和https_proxy环境变量
if 'http_proxy' in os.environ:
    del os.environ['http_proxy']
if 'https_proxy' in os.environ:
    del os.environ['https_proxy']
import argparse
import asyncio
import base64
import io
import json
import logging
import os
import sys
import time
import copy
import websockets
from pathlib import Path
from PIL import Image
import numpy as np

from datetime import timedelta
from moviepy.editor import ImageSequenceClip
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from termcolor import colored
from tqdm.auto import tqdm
import torch
import hydra

sys.path.insert(0, Path(__file__).absolute().parents[2].as_posix())
from calvin_agent.evaluation.multistep_sequences import get_sequences
from calvin_agent.evaluation.utils import (
    count_success,
    get_env_state_for_initial_condition,
    get_log_dir,
)

# Import from the original evaluation script
from evaluate_calvin import print_and_save, make_env

logger = logging.getLogger(__name__)

class RemoteModelClient:
    """WebSocket client that communicates with the evaluation server"""
    
    def __init__(self, server_url):
        self.server_url = server_url
        self.websocket = None
        self.connected = False
        self.client_id = None
        
    async def connect(self):
        """建立WebSocket连接"""
        try:
            self.websocket = await websockets.connect(
                self.server_url,
                ping_interval=30,
                ping_timeout=10,
                max_size=10 * 1024 * 1024,  # 10MB最大消息大小
                compression=None
            )
            
            # 等待连接确认消息
            response = await self.websocket.recv()
            data = json.loads(response)
            
            if data.get("type") == "connection_ack" and data.get("status") == "connected":
                self.connected = True
                self.client_id = data.get("client_id")
                logger.info(f"Connected to evaluation server at {self.server_url}, client_id: {self.client_id}")
            else:
                raise ConnectionError(f"Unexpected connection response: {data}")
                
        except Exception as e:
            logger.error(f"Failed to connect to server: {e}")
            raise
    
    async def disconnect(self):
        """断开WebSocket连接"""
        if self.websocket:
            try:
                await self.websocket.close()
            except Exception as e:
                logger.warning(f"Error closing websocket: {e}")
        self.connected = False
        logger.info("Disconnected from server")
    
    async def reset(self):
        """Reset model state on server"""
        try:
            if not self.connected:
                raise ConnectionError("Not connected to server")
                
            message = {"type": "reset"}
            await self.websocket.send(json.dumps(message))
            
            response = await self.websocket.recv()
            result = json.loads(response)
            
            if result.get("type") == "reset_response" and result.get("status") == "success":
                logger.debug("Model reset successful")
            else:
                logger.error(f"Reset failed: {result}")
                
        except Exception as e:
            logger.error(f"Reset request failed: {e}")
            raise
    
    async def step(self, obs, instruction, step):
        """Get action prediction from server"""
        try:
            if not self.connected:
                raise ConnectionError("Not connected to server")
                
            # 准备观察数据
            obs_data = self._serialize_observation(obs)
            
            # 发送预测请求
            message = {
                "type": "predict",
                "observation": obs_data,
                "instruction": instruction,
                "step": step
            }
            
            await self.websocket.send(json.dumps(message))
            
            # 接收响应
            response = await self.websocket.recv()
            result = json.loads(response)
            
            if result.get("type") == "predict_response":
                if result.get("status") == "success":
                    return np.array(result["action"])
                else:
                    logger.error(f"Prediction failed: {result.get('message', 'Unknown error')}")
                    return None
            else:
                logger.error(f"Unexpected response type: {result.get('type')}")
                return None
                
        except Exception as e:
            logger.error(f"Step request failed: {e}")
            return None
    
    def _serialize_observation(self, obs):
        """Convert observation to JSON-serializable format"""
        obs_data = {}
        
        # Serialize RGB observations
        obs_data['rgb_obs'] = {}
        for key in ['rgb_static', 'rgb_gripper']:
            if key in obs['rgb_obs']:
                img = Image.fromarray(obs['rgb_obs'][key])
                buffer = io.BytesIO()
                img.save(buffer, format='PNG')
                img_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                obs_data['rgb_obs'][key] = img_b64
        
        # Serialize depth observations  
        obs_data['depth_obs'] = {}
        for key in ['depth_static', 'depth_gripper']:
            if key in obs['depth_obs']:
                obs_data['depth_obs'][key] = obs['depth_obs'][key].tolist()
        
        # Serialize robot state
        obs_data['robot_obs'] = obs['robot_obs'].tolist()
        
        return obs_data

    async def ping(self):
        """发送心跳检测"""
        try:
            if not self.connected:
                return False
                
            message = {"type": "ping", "timestamp": time.time()}
            await self.websocket.send(json.dumps(message))
            
            response = await self.websocket.recv()
            result = json.loads(response)
            
            return result.get("type") == "pong"
        except Exception as e:
            logger.error(f"Ping failed: {e}")
            return False

# 同步包装器，使WebSocket客户端可以在同步代码中使用
class SyncRemoteModelClient:
    """同步WebSocket客户端包装器"""
    
    def __init__(self, server_url):
        self.server_url = server_url
        self.loop = None
        self.client = None
        
    def __enter__(self):
        # 创建新的事件循环
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        
        # 创建并连接WebSocket客户端
        self.client = RemoteModelClient(self.server_url)
        self.loop.run_until_complete(self.client.connect())
        
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        # 断开连接并关闭事件循环
        if self.client:
            self.loop.run_until_complete(self.client.disconnect())
        if self.loop:
            self.loop.close()
    
    def reset(self):
        """Reset model state"""
        return self.loop.run_until_complete(self.client.reset())
    
    def step(self, obs, instruction, step):
        """Get action prediction"""
        return self.loop.run_until_complete(self.client.step(obs, instruction, step))

def evaluate_policy_client(server_url, env, eval_sr_path, eval_result_path, eval_dir, ep_len, num_sequences, task_name='test', enrich_lang=False, debug=False):
    """Evaluate policy using remote WebSocket model server"""
    
    # 加载CALVIN 环境中所有子任务的成功条件。
    conf_dir = Path(f"{CALVIN_ROOT}/calvin_models") / "conf"
    task_cfg = OmegaConf.load(conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)
    
    if enrich_lang:
        with open('vla-scripts/enrich_lang_annotations.json', 'r') as f:
            val_annotations = json.load(f)
    else:
        val_annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")
        
    eval_dir = get_log_dir(eval_dir)
    eval_sequences = get_sequences(num_sequences) # 生成用于机器人评估的任务序列，每个序列包含初始状态和要执行的连续任务。(initial_state_1, ["turn_on_led", "open_drawer", "move_slider_right", ...]),

    results = []
    if not debug:
        eval_sequences = tqdm(eval_sequences, desc="Evaluating", leave=True) # 被包装成带进度条的迭代器

    # 使用WebSocket客户端
    with SyncRemoteModelClient(server_url) as model_client:
        sequence_i = 0
        for initial_state, eval_sequence in eval_sequences:
            result = evaluate_sequence_client(env, model_client, task_oracle, initial_state, eval_sequence, val_annotations, debug, eval_dir, sequence_i, ep_len)
            results.append(result)
            if not debug:
                success_list = count_success(results)
                with open(eval_sr_path, 'a') as f:
                    line = f"{sequence_i}/{num_sequences}: "
                    for sr in success_list:
                        line += f"{sr:.3f} | "
                    sequence_i += 1
                    line += "\n"
                    f.write(line)
                eval_sequences.set_description(
                    " ".join([f"{i + 1}/5 : {v * 100:.1f}% |" for i, v in enumerate(success_list)]) + "|"
                )
            else:
                sequence_i += 1
                
    print_and_save(results, eval_sequences, eval_result_path, task_name, None)
    return results

def evaluate_sequence_client(env, model_client, task_checker, initial_state, eval_sequence, val_annotations, debug, eval_dir, sequence_i, ep_len):
    """Evaluate a sequence using remote WebSocket model client"""
    robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
    success_counter = 0
    
    for subtask_i, subtask in enumerate(eval_sequence):
        success = rollout_client(env, model_client, task_checker, subtask, val_annotations, debug, eval_dir, subtask_i, sequence_i, ep_len)
        if success:
            success_counter += 1
        else:
            return success_counter
    return success_counter

def rollout_client(env, model_client, task_oracle, subtask, val_annotations, debug, eval_dir, subtask_i, sequence_i, ep_len):
    """Perform rollout using remote WebSocket model client"""
    obs = env.get_obs()
    lang_annotation = val_annotations[subtask][0]
    model_client.reset()
    start_info = env.get_info()

    for step in range(ep_len):
        action = model_client.step(obs, lang_annotation, step)
        if action is None:
            logger.error(f"Failed to get action at step {step}")
            return False
            
        obs, _, _, current_info = env.step(action)

        # check if current step solves a task
        current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
        if len(current_task_info) > 0:
            return True
    return False

def main(args):
    # Set seed
    seed_everything(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    save_path = './evaluation_results' 
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
        ep_len=args.ep_len,
        num_sequences=args.num_sequences,
        enrich_lang=args.enrich_lang,
        debug=args.debug,
    )).float().mean().to(device)

    print('average success rate ', avg_reward)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--server_url", default="ws://localhost:8765", type=str, help="WebSocket evaluation server URL")
    parser.add_argument("--calvin_path", default="./calvin", type=str)
    parser.add_argument("--log_dir", default="calvin", type=str)
    parser.add_argument("--enrich_lang", default=False, action="store_true")
    parser.add_argument("--num_sequences", default=10, type=int)
    parser.add_argument("--ep_len", default=360, type=int)  # 控制每个子任务最多执行多少步
    parser.add_argument("--debug", default=False, action="store_true")
    args = parser.parse_args()
    main(args)