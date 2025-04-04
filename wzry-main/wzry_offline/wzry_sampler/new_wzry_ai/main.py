import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import sys
from pathlib import Path
ROOT_DIR = Path(__file__).parent.absolute()
sys.path.insert(0, str(ROOT_DIR))

from training.trainer import WzryTrainer, TrainerConfig
from config.default_config import TrainingConfig

def main():

    print("Honor of Kings AI training main program")
    print("=" * 50)
    
    # 设置训练配置u
    config = TrainerConfig(
        load_model=True,  # 是否加载已有模型继续训练
        model_path=os.path.join(ROOT_DIR, TrainingConfig.MODEL_PATH),  # 模型保存路径
        save_freq=TrainingConfig.SAVE_FREQ,  # 保存频率
    )

    # 创建并启动训练器
    trainer = WzryTrainer(config=config)
    trainer.train()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nThe program has been interrupted by a user")
        sys.exit(0)
    except Exception as e:
        print(f"\nAn error occurred with the program: {e}")
        sys.exit(1)