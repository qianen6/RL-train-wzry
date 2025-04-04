import torch

def check_gpu():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"GPU is available. Using device: {torch.cuda.get_device_name(device)}")
    else:
        print("GPU is not available. Using CPU.")

# 运行检查函数
#check_gpu()
count=0
while True:
    print(count)
    count=count+1


