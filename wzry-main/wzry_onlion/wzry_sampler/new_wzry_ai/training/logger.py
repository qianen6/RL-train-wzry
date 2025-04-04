"""
训练日志模块
"""
import logging
import os
from typing import Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class LogConfig:
    log_dir: str = "../logs"
    log_level: int = logging.INFO
    log_format: str = '%(asctime)s - %(levelname)s - %(message)s'

class TrainingLogger: 
    def __init__(self, config: LogConfig = LogConfig()):
        self.config = config
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        os.makedirs(self.config.log_dir, exist_ok=True)
        
        # 创建日志文件名
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(self.config.log_dir, f'training_{timestamp}.log')
        
        # 配置日志记录器
        logger = logging.getLogger('training')
        logger.setLevel(self.config.log_level)
        
        # 添加文件处理器
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(self.config.log_format))
        logger.addHandler(file_handler)
        
        # 添加控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(self.config.log_format))
        logger.addHandler(console_handler)
        
        return logger
    
    def info(self, msg: str):
        """记录信息级别日志"""
        self.logger.info(msg)
        
    def warning(self, msg: str):
        """记录警告级别日志"""
        self.logger.warning(msg)
        
    def error(self, msg: str, exc_info: Optional[Exception] = None):
        """记录错误级别日志"""
        self.logger.error(msg, exc_info=exc_info) 