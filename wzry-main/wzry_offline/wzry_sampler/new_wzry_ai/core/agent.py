import os
import time
import psutil
import torch
import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass
from threading import Lock
from new_wzry_ai.core.model import HybridNet
from new_wzry_ai.core.memory import PriorityReplayBuffer, Experience
from new_wzry_ai.config.default_config import TrainingConfig
from new_wzry_ai.utils.PrintUtils import print_utils

# 引入 TensorRT 和 pycuda 库
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # 自动初始化 CUDA 驱动

@dataclass
class AgentState:
    epsilon: float
    steps_done: int

class DoubleDQN:
    def __init__(self, action_dims: Tuple[int, int] = (9, 8)):
        self.left_action_dim, self.right_action_dim = action_dims
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device:{self.device}")
        # 创建网络
        self.current_network = HybridNet(self.left_action_dim, self.right_action_dim).to(self.device)  # 创建左手网络输出维度为9
        self.target_network = HybridNet(self.left_action_dim, self.right_action_dim).to(self.device)  # 创建目标左手网络输出维度为9

        # 初始化目标网络
        self.target_network.load_state_dict(
            self.current_network.state_dict())  # 返回的是一个包含模型所有参数和缓冲区（例如 BatchNorm 中的 running_mean 和 running_var 等）状态的字典。

        # 优化器
        self.optimizer = torch.optim.Adam(
            self.current_network.parameters(),  # 返回一个迭代器，包含所有可训练参数的 Tensor
            lr=TrainingConfig.LEARNING_RATE
        )

        # 损失函数
        self.loss_func = torch.nn.SmoothL1Loss(reduction='none')

        # 智能体状态
        self.state = AgentState(
            epsilon=TrainingConfig.EPSILON_START,
            steps_done=0
        )

        # 添加线程锁
        self.train_lock = Lock()
        self.action_lock = Lock()


        # **记录模型结构**
        dummy_input1 = torch.randn(1, 4, 144, 320).to(self.device)  # 假设图像输入
        dummy_input2 = torch.randn(1, 64, 64, 2).to(self.device)  # 假设图像输入
        dummy_state = torch.randn(1, 4, 200).to(self.device)  # 假设状态向量输入
        # 确保模型至少执行一次前向传播
        output = self.current_network(dummy_input1, dummy_input2, dummy_state)

        # ============TensorRT加速================
        # 是否使用 TensorRT 加速推理
        self.use_trt = TrainingConfig.USE_TENSORRT
        self.trt_engine = None
        self.onnx_model_path = TrainingConfig.ONNX_MODEL_PATH
        if self.use_trt:
            # 如果 ONNX 模型不存在，则导出
            if not os.path.exists(self.onnx_model_path):
                self.export_to_onnx()
            # 构建 TensorRT 引擎
            self.trt_engine = self.build_trt_engine(self.onnx_model_path)


    def select_action(self, state: np.ndarray) -> Tuple[int, int]:
        with self.action_lock:
            if np.random.rand() < self.state.epsilon:
                left_action = np.random.randint(self.left_action_dim)
                right_action = np.random.randint(self.right_action_dim)
            else:
                image1=state[0]
                image2=state[1]
                state_vector = state[2]
                if self.use_trt and self.trt_engine is not None:
                    # 注意：reshape 要与 ONNX 模型导出时保持一致
                    left_q_values, right_q_values = self.tensorrt_inference(
                        np.array(image1),
                        np.array(image2),
                        np.array(state_vector)
                    )
                    left_action = int(np.argmax(left_q_values))
                    right_action = int(np.argmax(right_q_values))
                else:
                    # 原始 PyTorch 推理
                    image1_tensor = torch.tensor(np.array(image1), dtype=torch.float32).unsqueeze(0).to(self.device)
                    image2_tensor = torch.tensor(np.array(image2), dtype=torch.float32).unsqueeze(0).to(self.device)
                    state_tensor = torch.tensor(np.array(state_vector), dtype=torch.float32).unsqueeze(0).to(
                        self.device)
                    with torch.no_grad():
                        left_q_tensor, right_q_tensor = self.current_network(image1_tensor, image2_tensor, state_tensor)
                    left_action = int(left_q_tensor.argmax().item())
                    right_action = int(right_q_tensor.argmax().item())
                
            return left_action, right_action

    def export_to_onnx(self) -> None:
        """导出 ONNX 模型"""
        self.current_network.eval()
        dummy_input1 = torch.randn(1, 4, 144, 320).to(self.device)
        dummy_input2 = torch.randn(1, 64, 64, 2).to(self.device)
        dummy_state = torch.randn(1, 4, 200).to(self.device)
        torch.onnx.export(
            self.current_network,
            (dummy_input1, dummy_input2, dummy_state),
            self.onnx_model_path,
            export_params=True,
            opset_version=13,
            do_constant_folding=True,
            input_names=["image1", "image2", "state_vector"],
            output_names=["left_q_values", "right_q_values"]
        )
        print("ONNX 模型导出完成")

    def build_trt_engine(self, onnx_file_path: str, max_batch_size: int = 1) -> Optional[trt.ICudaEngine]:
        """TensorRT 10.9+ 引擎构建 (支持动态批处理)"""
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        EXPLICIT_BATCH = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

        with trt.Builder(TRT_LOGGER) as builder, \
                builder.create_network(EXPLICIT_BATCH) as network, \
                trt.OnnxParser(network, TRT_LOGGER) as parser:

            # 创建配置对象
            config = builder.create_builder_config()
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB

            # 启用 FP16
            if builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)

            # 解析 ONNX 模型
            with open(onnx_file_path, "rb") as model_file:
                if not parser.parse(model_file.read()):
                    print("ERROR: ONNX 解析失败")
                    for error in range(parser.num_errors):
                        print(parser.get_error(error))
                    return None

            # 定义动态输入配置
            profile = builder.create_optimization_profile()
            input_shapes = {
                "image1": (4, 144, 320),  # (C, H, W)
                "image2": (64, 64, 2),  # (H, W, C)
                "state_vector": (4, 200)  # (seq_len, features)
            }

            # 设置每个输入的动态范围
            for name, shape in input_shapes.items():
                profile.set_shape(
                    name,
                    min=(1, *shape),  # 最小批次
                    opt=(max_batch_size, *shape),  # 最优批次
                    max=(max_batch_size, *shape)  # 最大批次
                )
            config.add_optimization_profile(profile)

            # 构建序列化引擎
            serialized_engine = builder.build_serialized_network(network, config)
            if not serialized_engine:
                print("TensorRT 引擎构建失败")
                return None

            # 反序列化引擎
            runtime = trt.Runtime(TRT_LOGGER)
            engine = runtime.deserialize_cuda_engine(serialized_engine)
            print("TensorRT 引擎构建成功")
            return engine

    def tensorrt_inference(self, image1: np.ndarray, image2: np.ndarray, state_vector: np.ndarray) -> Tuple[
        np.ndarray, np.ndarray]:
        if self.trt_engine is None:
            raise RuntimeError("TensorRT 引擎未初始化。")

        context = self.trt_engine.create_execution_context()
        stream = cuda.Stream()

        # 获取所有绑定名称和索引
        binding_names = []
        for i in range(self.trt_engine.num_bindings):
            binding_names.append(self.trt_engine.get_binding_name(i))

        # 根据名称查找索引
        binding_indices = {
            "image1": binding_names.index("image1"),
            "image2": binding_names.index("image2"),
            "state_vector": binding_names.index("state_vector"),
            "left_q_values": binding_names.index("left_q_values"),
            "right_q_values": binding_names.index("right_q_values")
        }

        # 设置动态输入形状
        batch_size = image1.shape[0]
        context.set_binding_shape(binding_indices["image1"], image1.shape)
        context.set_binding_shape(binding_indices["image2"], image2.shape)
        context.set_binding_shape(binding_indices["state_vector"], state_vector.shape)

        # 准备设备内存
        bindings = []

        # 处理输入
        h_inputs = {
            "image1": np.ascontiguousarray(image1.astype(np.float32)),
            "image2": np.ascontiguousarray(image2.astype(np.float32)),
            "state_vector": np.ascontiguousarray(state_vector.astype(np.float32))
        }

        d_inputs = {}
        for name in ["image1", "image2", "state_vector"]:
            idx = binding_indices[name]
            d_input = cuda.mem_alloc(h_inputs[name].nbytes)
            cuda.memcpy_htod_async(d_input, h_inputs[name], stream)
            bindings.append(d_input)
            d_inputs[name] = d_input

        # 处理输出
        h_outputs = {
            "left_q_values": np.empty((batch_size, self.left_action_dim), dtype=np.float32),
            "right_q_values": np.empty((batch_size, self.right_action_dim), dtype=np.float32)
        }

        d_outputs = {}
        for name in ["left_q_values", "right_q_values"]:
            idx = binding_indices[name]
            shape = tuple(context.get_binding_shape(idx))
            dtype = trt.nptype(self.trt_engine.get_binding_dtype(idx))
            h_outputs[name] = np.empty(shape, dtype=dtype)
            d_output = cuda.mem_alloc(h_outputs[name].nbytes)
            bindings.append(d_output)
            d_outputs[name] = d_output

        # 执行推理
        context.execute_async_v2(bindings=[int(b) for b in bindings], stream_handle=stream.handle)
        stream.synchronize()

        # 拷贝输出数据
        for name in ["left_q_values", "right_q_values"]:
            cuda.memcpy_dtoh_async(h_outputs[name], d_outputs[name], stream)

        stream.synchronize()

        # 释放设备内存
        for d in d_inputs.values() + d_outputs.values():
            d.free()

        return h_outputs["left_q_values"], h_outputs["right_q_values"]


    def load_model(self, path: str) -> None:
        try:
            checkpoint = torch.load(
                path,
                map_location=self.device,
                weights_only=True
            )

            # 加载网络状态
            self.current_network.load_state_dict(checkpoint['current_network_state_dict'])
            self.target_network.load_state_dict(checkpoint['target_network_state_dict'])

            # 加载优化器状态
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            # 加载训练状态
            self.state.epsilon = float(checkpoint['epsilon'])
            self.state.steps_done = int(checkpoint['steps_done'])

            print(f"The model and training status are loaded：{path}")
        except Exception as e:
            print(f"An exception occurred while loading the model: {e}")
            self._reset_model()
            
    def _reset_model(self) -> None:
        """重置模型到初始状态"""
        self.current_network = HybridNet(self.left_action_dim,self.right_action_dim).to(self.device)
        self.target_network = HybridNet(self.left_action_dim,self.right_action_dim).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.current_network.parameters(),
            lr=TrainingConfig.LEARNING_RATE
        )
        self.state.epsilon = TrainingConfig.EPSILON_START
        self.state.steps_done = 0
        print("Failed to load, model reinitialized")