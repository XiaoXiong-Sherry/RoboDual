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
import sys
import torch
import websockets
from datetime import timedelta
from pathlib import Path
from PIL import Image
import numpy as np
from transformers import AutoModelForVision2Seq, AutoProcessor
from pytorch_lightning import seed_everything
from prismatic.models.policy.diffusion_policy import DiffusionDiTImagePolicy
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from prismatic.vla.action_tokenizer import ActionTokenizer
# from openvla.prismatic.vla.action_tokenizer import ActionTokenizer
from train_spacialist_calvin import DualSystem



sys.path.insert(0, Path(__file__).absolute().parents[2].as_posix())

from dual_sys_evaluation import DualSystemCalvinEvaluation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EvaluationServerWS:
    def __init__(self, args):
        self.args = args
        self.dual_sys = None
        self.processor = None
        self.action_tokenizer = None
        self.eva = None
        self.clients = set()  # 跟踪连接的客户端
        self._setup_model()
    
    def _setup_model(self):
        seed_everything(42)
        
        # Get device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        # Load generalist policy
        self.processor = AutoProcessor.from_pretrained(self.args.generalist_path, trust_remote_code=True)
        model = AutoModelForVision2Seq.from_pretrained(
            self.args.generalist_path,
            torch_dtype=torch.bfloat16,
            quantization_config=None,
            low_cpu_mem_usage=False,
            trust_remote_code=True,
        ).to(device)
        model.eval()
        
        # Load specialist policy  
        scheduler = DDIMScheduler(num_train_timesteps=100, beta_schedule='squaredcos_cap_v2', prediction_type="epsilon")
        shape_meta = {'action': {'shape': [7]}}
        diffusion_policy = DiffusionDiTImagePolicy(
            shape_meta=shape_meta,
            noise_scheduler=scheduler,
            n_action_steps=8,
            num_inference_steps=10,
            vision_encoder='DINO',
            with_depth=self.args.with_depth,
            progressive_noise=False,
            with_gripper=self.args.with_gripper,
            with_tactile=self.args.with_tactile,
            cond_drop_chance=0.1 if self.args.with_cfg else 0.,
        ).to(device).eval()
        
        # Load dual system 
        self.action_tokenizer = ActionTokenizer(self.processor.tokenizer)
        self.dual_sys = DualSystem(model, diffusion_policy, self.action_tokenizer)
        self.dual_sys.ema_fast_system.load_state_dict(torch.load(self.args.specialist_path), strict=False)
        self.dual_sys = self.dual_sys.to(device)
        self.dual_sys.eval()
        
        # Create evaluation wrapper
        self.eva = DualSystemCalvinEvaluation(self.dual_sys, self.processor, self.action_tokenizer)
        
        logger.info("Model setup completed")
    
    def reset_model(self):
        """Reset model state for new sequence"""
        if self.eva:
            self.eva.reset()
        return {"type": "reset_response", "status": "success"}
    
    def predict_action(self, obs_data, instruction, step):
        """Predict action given observation and instruction"""
        try:
            # Parse observation data
            obs = self._parse_observation(obs_data)
            
            # Get action prediction
            action = self.eva.step(obs, instruction, step)
            
            return {
                "type": "predict_response",
                "status": "success", 
                "action": action.tolist(),  # JSON serialization requires Python lists/basic types,
                "step": step
            }
        except Exception as e:
            logger.error(f"Action prediction failed: {str(e)}")
            return {"type": "predict_response", "status": "error", "message": str(e)}
    
    def _parse_observation(self, obs_data):
        """Parse observation data from client
        因为网络传输的数据格式（JSON + base64编码图像）与模型期望的输入格式（numpy数组）不同
        """
        obs = {}
        
        # Parse RGB observations
        obs['rgb_obs'] = {}
        for key in ['rgb_static', 'rgb_gripper']:
            if key in obs_data['rgb_obs']:
                img_data = base64.b64decode(obs_data['rgb_obs'][key])
                img = Image.open(io.BytesIO(img_data))
                obs['rgb_obs'][key] = np.array(img)
        
        # Parse depth observations
        obs['depth_obs'] = {}
        for key in ['depth_static', 'depth_gripper']:
            if key in obs_data['depth_obs']:
                obs['depth_obs'][key] = np.array(obs_data['depth_obs'][key])
        
        # Parse robot state
        obs['robot_obs'] = np.array(obs_data['robot_obs'])
        
        return obs

    async def handle_client(self, websocket):
        """处理单个客户端连接
        当一个客户端成功连接到服务器时，
        websockets.serve 会自动为这个连接创建一个新的任务，并调用此函数来处理该连接上的所有后续通信。
        每个客户端连接都会有一个独立的 handle_client 实例在运行。
        """
        client_id = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
        self.clients.add(websocket)
        logger.info(f"Client {client_id} connected. Total clients: {len(self.clients)}")
        
        try:
            # 发送连接确认
            # 连接成功后，立即向客户端发送一个JSON格式的确认消息。
            # 这让客户端知道它已成功连接，并可以开始发送数据。
            await websocket.send(json.dumps({
                "type": "connection_ack",
                "status": "connected",
                "client_id": client_id
            }))
            
            async for message in websocket:
                try:
                    data = json.loads(message)
                    message_type = data.get("type")
                    
                    if message_type == "ping":
                        # 心跳检测
                        response = {"type": "pong", "timestamp": data.get("timestamp")}
                        
                    elif message_type == "reset":
                        # 重置模型状态; Reset不是重新加载模型权重，而是清理推理过程中积累的上下文状态，确保每个新任务都从"干净"的状态开始
                        response = self.reset_model()
                        
                    elif message_type == "predict":
                        # 动作预测
                        obs_data = data["observation"]
                        instruction = data["instruction"]
                        step = data["step"]
                        response = self.predict_action(obs_data, instruction, step)
                        
                    else:
                        response = {
                            "type": "error",
                            "status": "error",
                            "message": f"Unknown message type: {message_type}"
                        }
                    
                    # 发送响应：将处理结果（无论成功还是失败）打包成JSON字符串并发回给客户端。
                    await websocket.send(json.dumps(response))
                    
                except json.JSONDecodeError as e:
                    error_response = {
                        "type": "error",
                        "status": "error",
                        "message": f"Invalid JSON: {str(e)}"
                    }
                    await websocket.send(json.dumps(error_response))
                    
                except Exception as e:
                    logger.error(f"Error processing message from {client_id}: {str(e)}")
                    error_response = {
                        "type": "error",
                        "status": "error",
                        "message": f"Processing error: {str(e)}"
                    }
                    await websocket.send(json.dumps(error_response))
                    
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client {client_id} disconnected normally")
        except Exception as e:
            logger.error(f"Error with client {client_id}: {str(e)}")
        finally:
            self.clients.discard(websocket)
            logger.info(f"Client {client_id} removed. Total clients: {len(self.clients)}")

    async def start_server(self):
        """启动WebSocket服务器
        它能够同时处理多个客户端连接，并对每个客户端的请求进行异步响应
        """
        logger.info(f"Starting WebSocket evaluation server on {self.args.host}:{self.args.port}")
        
        async with websockets.serve(
            self.handle_client,  # 处理每个客户端连接的函数
            self.args.host,
            self.args.port,
            ping_interval=30,  # 30秒客户端心跳检测
            ping_timeout=10,   # 10秒客户端心跳超时, 超时关闭单个客户端连接
            max_size=10 * 1024 * 1024,  # 10MB最大消息大小
            compression=None   # 禁用压缩以提高性能
        ):
            logger.info("WebSocket server started successfully")
            # 保持服务器运行
            await asyncio.Future()  # server run forever

def main(args):
    # 初始化服务器
    server = EvaluationServerWS(args)
    
    # 运行WebSocket服务器
    try:
        asyncio.run(server.start_server())
    except KeyboardInterrupt:
        logger.info("Server shutdown requested")
    except Exception as e:
        logger.error(f"Server error: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generalist_path", default="pretrained_models/OpenVLA-Generalist", type=str)
    parser.add_argument("--specialist_path", default="pretrained_models/Specialist/Specialist+Depth+Gripper.pt", type=str)
    parser.add_argument("--with_depth", default=True, action="store_true")
    parser.add_argument("--with_gripper", default=True, action="store_true")
    parser.add_argument("--with_tactile", default=False, action="store_true")
    parser.add_argument("--with_cfg", default=False, action="store_true")
    parser.add_argument("--host", default="0.0.0.0", type=str)
    parser.add_argument("--port", default=8765, type=int)  # WebSocket默认端口
    args = parser.parse_args()
    main(args)