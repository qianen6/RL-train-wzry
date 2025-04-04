import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import sys
from pathlib import Path
ROOT_DIR = Path(__file__).parent.absolute()
sys.path.insert(0, str(ROOT_DIR))

from training.trainer import WzryTrainer, TrainerConfig
from config.default_config import TrainingConfig


def run_trainer():
    config = TrainerConfig(
        load_model=True,
        model_path=os.path.join(ROOT_DIR, TrainingConfig.MODEL_PATH),
        save_freq=TrainingConfig.SAVE_FREQ,
    )
    trainer = WzryTrainer(config=config)
    trainer.train()

def main():
    print("Honor of Kings AI training main program")
    print("=" * 50)
    # 启动训练器
    run_trainer()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nThe program has been interrupted by a user")
        sys.exit(0)
    except Exception as e:
        print(f"\nAn error occurred with the program: {e}")
        sys.exit(1)