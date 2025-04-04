from typing import Dict, Tuple
import cv2
import numpy as np
from dataclasses import dataclass
import pytesseract


@dataclass
class SkillConfig:
    position: Tuple[int, int]
    threshold: int
    region: Tuple[int, int, int, int]


class CooldownDetector:
    # 技能配置
    DEFAULT_SKILLS_CONFIG: Dict[int, SkillConfig] = {
        1: SkillConfig((1219, 649), 200, (1180, 615, 1223, 650)),  # 1技能
        2: SkillConfig((1290, 508), 200, (1258, 479, 1301, 514)),  # 2技能
        3: SkillConfig((1395, 421), 200, (1391, 403, 1434, 438)),  # 3技能
        5: SkillConfig((974, 665), 200, (967, 631, 1011, 660)),  # 恢复
        6: SkillConfig((1061, 665), 200, (1062, 630, 1118, 661)),  # 闪现
    }

    def __init__(self, sample_size: int = 1):
        self.skills_config = self.DEFAULT_SKILLS_CONFIG.copy()
        self.last_status = {skill_id: True for skill_id in self.skills_config}
        self.sample_buffer = {skill_id: [] for skill_id in self.skills_config}
        self.sample_size = sample_size

    def check_skills_status(self, image: np.ndarray) -> Dict[int, bool]:
        if image is None:
            return {skill_id: False for skill_id in self.skills_config}

        if len(image.shape) > 2:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image

        result = {}

        for skill_id, config in self.skills_config.items():
            if not self._is_valid_position(gray_image, *config.position):
                result[skill_id] = False
                continue

            # 获取当前灰度值
            current_gray = gray_image[config.position[1], config.position[0]]
            current_status = current_gray > config.threshold

            if current_status != self.last_status[skill_id]:
                # 状态发生变化，添加到采样缓冲区
                self.sample_buffer[skill_id].append(current_status)

                if len(self.sample_buffer[skill_id]) >= self.sample_size:
                    # 采样足够，计算主要状态
                    true_count = sum(1 for x in self.sample_buffer[skill_id] if x)
                    final_status = true_count > len(self.sample_buffer[skill_id]) // 2

                    # 更新状态并清空缓冲区
                    self.last_status[skill_id] = final_status
                    self.sample_buffer[skill_id] = []
                    result[skill_id] = final_status
                else:
                    # 采样不足，保持上一次的状态
                    result[skill_id] = self.last_status[skill_id]
            else:
                # 状态没有变化，清空采样缓冲区
                self.sample_buffer[skill_id] = []
                result[skill_id] = current_status
                self.last_status[skill_id] = current_status

        return result

    def get_skill_cooldown(self, image: np.ndarray):
        skills_cooldown = {
            1: 0,
            2: 0,
            3: 0,
            4: 0,
            5: 0,
            6: 0,
        }
        if image is None:
            return skills_cooldown

        skills_status = self.check_skills_status(image)

        for key, value in skills_status.items():
            if not value:
                skills_cooldown[key] = self.get_cool_down(image, self.skills_config[key].region)

        return skills_cooldown

    @staticmethod
    def get_cool_down(image, region):

        x1, y1, x2, y2 = region

        image = image[y1:y2, x1:x2]

        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        lower_white = np.array([0, 0, 200])  # 低值

        upper_white = np.array([180, 20, 255])  # 高值

        image = cv2.inRange(image, lower_white, upper_white)

        text = pytesseract.image_to_string(image, config='--psm 6 outputbase digi')

        # 如果无法识别任何内容，返回 0
        try:
            return int(text.strip()) if text.strip().isdigit() else 0
        except ValueError:
            return 0

    def _is_valid_position(self, image: np.ndarray, x: int, y: int) -> bool:
        """检查坐标是否在图像范围内"""
        height, width = image.shape[:2]
        return 0 <= x < width and 0 <= y < height
