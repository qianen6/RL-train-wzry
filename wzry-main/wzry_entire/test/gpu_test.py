import torch

def check_gpu():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"GPU is available. Using device: {torch.torch_version}")
    else:
        print("GPU is not available. Using CPU.")

# 运行检查函数
check_gpu()



