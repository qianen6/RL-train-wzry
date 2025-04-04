import cv2
import numpy as np
from typing import Tuple, Dict, Optional
from dataclasses import dataclass
from new_wzry_ai.utils.template_match import TemplateMatcher

@dataclass
class Destination:
    name: str
    position: Tuple[int, int]
    threshold: int

class RoutingGuide:
    # 默认目标点配置
    DEFAULT_DESTINATIONS = {
        "我方红buff": Destination("我方红buff", (203, 179), 13),
        "我方野怪鸡": Destination("我方野怪鸡", (232, 194), 10),
        "我方野怪刺猬": Destination("我方野怪刺猬", (166, 182), 10),
        "我方野怪猪": Destination("我方野怪猪", (191, 155), 10),
        "我方蓝buff": Destination("我方蓝buff", (143, 107), 13),
        "我方野怪双狼": Destination("我方野怪双狼", (128, 134), 10),
        "我方野怪蜥蜴": Destination("我方野怪蜥蜴", (112, 101), 10),
        "我方穿山甲": Destination("我方穿山甲", (121, 68), 10),
        "敌方红buff": Destination("敌方红buff", (187, 47), 13),
        "敌方野怪鸡": Destination("敌方野怪鸡", (157, 33), 10),
        "敌方野怪刺猬": Destination("敌方野怪刺猬", (225, 46), 10),
        "敌方野怪猪": Destination("敌方野怪猪", (201, 74), 10),
        "敌方蓝buff": Destination("敌方蓝buff", (247, 119), 13),
        "敌方野怪双狼": Destination("敌方野怪双狼", (263, 92), 10),
        "敌方野怪蜥蜴": Destination("敌方野怪蜥蜴", (277, 126), 10),
        "敌方穿山甲": Destination("敌方穿山甲", (270, 160), 10),
        "大龙": Destination("大龙", (157, 65), 15),
        "小龙": Destination("小龙", (235, 161), 15),
        "对抗路精灵": Destination("对抗路精灵", (95, 14), 10),
        "发育路小鸟": Destination("发育路小鸟", (289, 209), 10),
    }

    def __init__(self):
        """初始化路径导航器"""
        self.template_matcher = TemplateMatcher()
        self.destinations = self.DEFAULT_DESTINATIONS.copy()
        self.last_position = None#记录上一次位置
        
    def add_destination(self, 
                       name: str, 
                       position: Tuple[int, int], 
                       threshold: int = 10) -> None:
        self.destinations[name] = Destination(name, position, threshold)
        
    def get_houyi_position(self, image: np.ndarray) -> Optional[Tuple[int, int]]:
        if image is None:
            return None
            
        result = self.template_matcher.match_template(image, 'houyi')
        if result.success:
            return (result.x+7, result.y+3)
        return None
        
    def is_moving_to_target(self, 
                           image: np.ndarray,
                           target_pos: Tuple[int, int]) -> bool:
        current_pos = self.get_houyi_position(image)
        if current_pos is None or self.last_position is None:
            self.last_position = current_pos
            return False
            
        current_distance = self._calculate_distance(current_pos, target_pos)
        last_distance = self._calculate_distance(self.last_position, target_pos)
        
        self.last_position = current_pos
        return current_distance < last_distance

    def is_at_destination(self, 
                         image: np.ndarray,
                         destination_name: str) -> bool:
        if destination_name not in self.destinations:
            raise ValueError(f"未找到目标点: {destination_name}")
            
        current_pos = self.get_houyi_position(image)
        if current_pos is None:
            return False
            
        dest = self.destinations[destination_name]
        distance = self._calculate_distance(current_pos, dest.position)
        return distance <= dest.threshold

    def is_at_any_destination(self, 
                            image: np.ndarray, 
                            current_pos: Optional[Tuple[int, int]] = None
                            ) -> Tuple[bool, Optional[str]]:
        if current_pos is None:
            current_pos = self.get_houyi_position(image)
            if current_pos is None:
                return False, None
            
        for dest in self.destinations.values():
            distance = self._calculate_distance(current_pos, dest.position)
            if distance <= dest.threshold:
                return True, dest.name
                
        return False, None

    def detect_any_monster(self, image: np.ndarray) -> bool:
        if image is None:
            return False
            
        for template_name in self.template_matcher.template_configs:
            if template_name.startswith(('chuanshanjia', 'ciwei', 'dalong', 'hongbuff', 'ji',
                                       'jingling', 'lanbuff', 'lang', 'longwang', 'xialong',
                                       'xiaoniao', 'xiyi', 'zhu')):
                result = self.template_matcher.match_template(image, template_name)
                if result.success:
                    return True
                    
        return False

    def _calculate_distance(self, 
                          pos1: Tuple[int, int], 
                          pos2: Tuple[int, int]) -> float:
        """计算两点间欧几里得距离"""
        return np.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2) 