# import torch
#
# def check_gpu():
#     if torch.cuda.is_available():
#         device = torch.device("cuda")
#         print(f"GPU is available. Using device: {torch.cuda.get_device_name(device)}")
#     else:
#         print("GPU is not available. Using CPU.")
#
# # 运行检查函数
# #check_gpu()
# count=0
# while True:
#     print(count)
#     count=count+1



#===================================================================
# import pycuda.driver as cuda
# from pycuda.autoinit import context
#
# # 测试 Stream 类是否存在
# print("Stream 类存在:", hasattr(cuda, "Stream"))
#
# # 创建 CUDA 流
# stream = cuda.Stream()
# print("CUDA 流创建成功:", stream)

#=================================================================
# import torch
# print(torch.__version__)          # 应输出 2.6.0+cu124
# print(torch.cuda.is_available())  # 应输出 True
#
# import torchvision
# print(torchvision.__version__)    # 应输出 0.21.0+cu124

#=================================================================

import onnx
print(onnx.__version__)
import tensorrt as trt
print(trt.__version__)  # 输出类似 7.2.3.4 或 8.6.1.6