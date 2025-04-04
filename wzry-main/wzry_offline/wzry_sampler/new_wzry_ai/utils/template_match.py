import time

import cv2
import numpy as np
import os
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from new_wzry_ai.config.default_config import (
    TemplateRegion, 
    Confidence, 
    MatchAlgorithm,
    MONSTER_TEMPLATES
)

@dataclass
class MatchResult:
    success: bool
    x: Optional[int] = None
    y: Optional[int] = None

@dataclass
class TemplateConfig:
    pos: Tuple[int, int, int, int]
    confidence: float
    algorithm: MatchAlgorithm = MatchAlgorithm.CCOEFF_NORMED

class ImageLoader:
    def __init__(self, folder_path: str = 'template'):
        self.folder_path = folder_path
        self.images: Dict[str, np.ndarray] = {}
        self._load_images()
    
    def _load_images(self) -> None:
        if not os.path.exists(self.folder_path):
            raise FileNotFoundError(f"模板文件夹 '{self.folder_path}' 不存在")
            
        for root, _, files in os.walk(self.folder_path):
            for file in files:
                if file.lower().endswith('.png'):
                    image_path = os.path.join(root, file)
                    filename = os.path.splitext(file)[0]
                    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                    if image is None:
                        print(f"警告: 无法加载图像 '{image_path}'")
                        continue
                    self.images[filename] = image
                    
    def get_template(self, name: str) -> Optional[np.ndarray]:
        """获取指定名称的模板图像"""
        return self.images.get(name)

class TemplateMatcher:
    def __init__(self, folder_path: str = 'template'):
        self.image_loader = ImageLoader(folder_path)
        self.template_configs = self._init_template_configs()

    def _init_template_configs(self) -> Dict[str, TemplateConfig]:
        """初始化默认模板配置"""
        configs = {
            'self_blood': TemplateConfig(
                pos=TemplateRegion.BLOOD_AREA,
                confidence=Confidence.HIGH
            ),
            'enemy_blood': TemplateConfig(
                pos=TemplateRegion.ENEMY_BLOOD_AREA,
                confidence=Confidence.HIGH
            ),
            'houyi': TemplateConfig(
                pos=TemplateRegion.CHARACTER_AREA,
                confidence=Confidence.LOW
            ),
            'game_start': TemplateConfig(
                pos=TemplateRegion.GAME_START_AREA,
                confidence=Confidence.HIGH
            ),
            'victory': TemplateConfig(
                pos=TemplateRegion.VICTORY_AREA,
                confidence=Confidence.HIGH
            ),
            'defeat': TemplateConfig(
                pos=TemplateRegion.DEFEAT_AREA,
                confidence=Confidence.HIGH
            ),
            'back_home': TemplateConfig(
                pos=TemplateRegion.BACK_HOME_AREA,
                confidence=Confidence.HIGH
            ),
            'mode': TemplateConfig(
                pos=TemplateRegion.MODE_AREA,
                confidence=Confidence.HIGH
            ),
            'start': TemplateConfig(
                pos=TemplateRegion.START_AREA,
                confidence=Confidence.HIGH
            ),
            'icon': TemplateConfig(
                pos=TemplateRegion.ICON_AREA,
                confidence=Confidence.HIGH
            ),
            'fayulu': TemplateConfig(
                pos=TemplateRegion.FAYULU_AREA,
                confidence=Confidence.HIGH
            ),
            'houyi_ico': TemplateConfig(
                pos=TemplateRegion.HOUYI_ICO_AREA,
                confidence=Confidence.HIGH
            ),
            'confirm': TemplateConfig(
                pos=TemplateRegion.CONFIRM_AREA,
                confidence=Confidence.HIGH
            ),
            'self_death1': TemplateConfig(
                pos=TemplateRegion.SELF_DEATH_AREA1,
                confidence=Confidence.HIGH
            ),
            'self_death2': TemplateConfig(
                pos=TemplateRegion.SELF_DEATH_AREA2,
                confidence=Confidence.HIGH
            )
        }
        
        # 添加野怪模板配置
        for template_name in MONSTER_TEMPLATES:
            configs[template_name] = TemplateConfig(
                pos=TemplateRegion.MONSTER_AREA,
                confidence=Confidence.MEDIUM
            )
            
        return configs

    def add_template_config(self, name: str, config: TemplateConfig) -> None:
        """添加或更新模板配置"""
        self.template_configs[name] = config

    def match_template(self, 
                      image: np.ndarray, 
                      temp_name: str) -> MatchResult:
        if temp_name not in self.template_configs:
            raise ValueError(f"模板 '{temp_name}' 未配置")
            
        config = self.template_configs[temp_name]

        return self.match(
            image=image,
            temp_name=temp_name,
            pos=config.pos,
            confidence=config.confidence,
            algorithm=config.algorithm
        )

    def match(self, 
              image: np.ndarray, 
              temp_name: str, 
              pos: Optional[Tuple[int, int, int, int]] = None,
              confidence: float = Confidence.HIGH,
              algorithm: MatchAlgorithm = MatchAlgorithm.CCOEFF_NORMED
              ) -> MatchResult:
        if image is None:
            return MatchResult(False)

        # 转换为灰度图
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        template = self.image_loader.get_template(temp_name)

        if template is None:
            return MatchResult(False)

        roi = self._get_roi(image, pos)

        result = cv2.matchTemplate(roi, template, algorithm.value)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        if max_val >= confidence:
            x, y = self._calculate_coordinates(max_loc, pos)
            return MatchResult(True, x, y)
        
        return MatchResult(False)

    def _get_roi(self, 
                 image: np.ndarray, 
                 pos: Optional[Tuple[int, int, int, int]]
                 ) -> np.ndarray:
        """获取感兴趣区域"""
        if pos is None:
            return image
        x1, y1, x2, y2 = pos
        # 保存 ROI 图像到指定文件夹
        save_folder = "../img"
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        roi_file_path = os.path.join(save_folder, f"roi.png")
        roi=image[y1:y2, x1:x2]
        cv2.imwrite(roi_file_path, roi)
        return roi

    def _calculate_coordinates(self, 
                             max_loc: Tuple[int, int], 
                             pos: Optional[Tuple[int, int, int, int]]
                             ) -> Tuple[int, int]:
        """计算匹配坐标"""
        if pos is None:
            return max_loc
        x1, y1, _, _ = pos
        return x1 + max_loc[0], y1 + max_loc[1]