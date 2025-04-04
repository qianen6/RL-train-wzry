import cv2
import numpy as np
import time
from dataclasses import dataclass, field
from typing import List
from new_wzry_ai.utils.template_match import TemplateMatcher

def default_self_blood_hsv_lower() -> np.ndarray:
    """返回默认的自身血条HSV下限值"""
    return np.array([40, 100, 100])

def default_self_blood_hsv_upper() -> np.ndarray:
    """返回默认的自身血条HSV上限值"""
    return np.array([70, 255, 255])

def default_enemy_blood_hsv_lower1() -> np.ndarray:
    """返回默认的敌人血条HSV下限值1"""
    return np.array([0, 50, 50])

def default_enemy_blood_hsv_upper1() -> np.ndarray:
    """返回默认的敌人血条HSV上限值1"""
    return np.array([10, 255, 255])

def default_enemy_blood_hsv_lower2() -> np.ndarray:
    """返回默认的敌人血条HSV下限值2"""
    return np.array([170, 50, 50])

def default_enemy_blood_hsv_upper2() -> np.ndarray:
    """返回默认的敌人血条HSV上限值2"""
    return np.array([180, 255, 255])

@dataclass
class BloodDetectorConfig:
    change_duration: float = 0.5
    self_blood_hsv_lower: np.ndarray = field(default_factory=default_self_blood_hsv_lower)
    self_blood_hsv_upper: np.ndarray = field(default_factory=default_self_blood_hsv_upper)
    enemy_blood_hsv_lower1: np.ndarray = field(default_factory=default_enemy_blood_hsv_lower1)
    enemy_blood_hsv_upper1: np.ndarray = field(default_factory=default_enemy_blood_hsv_upper1)
    enemy_blood_hsv_lower2: np.ndarray = field(default_factory=default_enemy_blood_hsv_lower2)
    enemy_blood_hsv_upper2: np.ndarray = field(default_factory=default_enemy_blood_hsv_upper2)

class BloodDetector:
    def __init__(self, config: BloodDetectorConfig = BloodDetectorConfig()):
        """初始化血量检测器"""
        self.template_matcher = TemplateMatcher()
        self.config = config
        self.enemy_blood_changed = False
        self.last_check_time = 0
        self.last_enemy_blood = 0.0

    def get_self_blood(self, image: np.ndarray) -> float:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        res = self.template_matcher.match_template(gray_image, 'self_blood')
        if not res.success:
            return 0.0
            
        x1, y1 ,x2,y2= 744,264,867,272
        
        if y2 <= image.shape[0] and x2 <= image.shape[1]:
            blood_roi = image[y1:y2, x1:x2]
            hsv = cv2.cvtColor(blood_roi, cv2.COLOR_BGR2HSV)
            
            mask = cv2.inRange(hsv, 
                             self.config.self_blood_hsv_lower,
                             self.config.self_blood_hsv_upper)
            
            kernel = np.ones((4,4), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            total_pixels = mask.shape[0] * mask.shape[1]
            non_zero_pixels = np.sum(mask > 0)
            return non_zero_pixels / total_pixels
            
        return 0.0

    def get_enemy_blood(self, image: np.ndarray) -> float:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        res = self.template_matcher.match_template(gray_image, 'enemy_blood')
        if not res.success:
            return 0.0
        blood_roi_color = image[15:30, 818:1003]
        hsv = cv2.cvtColor(blood_roi_color, cv2.COLOR_BGR2HSV)
        
        mask1 = cv2.inRange(hsv, 
                           self.config.enemy_blood_hsv_lower1,
                           self.config.enemy_blood_hsv_upper1)
        mask2 = cv2.inRange(hsv, 
                           self.config.enemy_blood_hsv_lower2,
                           self.config.enemy_blood_hsv_upper2)
        mask = cv2.bitwise_or(mask1, mask2)
        
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        total_pixels = mask.shape[0] * mask.shape[1]
        non_zero_pixels = np.sum(mask > 0)
        return non_zero_pixels / total_pixels

    def is_enemy_blood_changed(self, image: np.ndarray) -> bool:
        current_blood = self.get_enemy_blood(image)
        current_time = time.time()
        
        # 如果当前血量为0，重置状态
        if current_blood == 0.0:
            self.last_enemy_blood = 0.0
            self.enemy_blood_changed = False
            return False
            
        # 如果是第一次检测到血量，更新记录但返回False
        if self.last_enemy_blood == 0.0:
            self.last_enemy_blood = current_blood
            return False
            
        # 检查血量是否发生变化（使用阈值避免微小波动）
        if abs(current_blood - self.last_enemy_blood) > 0.01:
            self.enemy_blood_changed = True
            self.last_check_time = current_time
        elif current_time - self.last_check_time > self.config.change_duration:
            self.enemy_blood_changed = False
        
        self.last_enemy_blood = current_blood
        return self.enemy_blood_changed 