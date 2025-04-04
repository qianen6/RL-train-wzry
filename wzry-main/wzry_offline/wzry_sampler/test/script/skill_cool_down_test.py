import pytest
import random
import numpy as np
import cv2
from typing import Tuple
from cooldown_detector import CooldownDetector, SkillConfig  # 假设原代码在 cooldown_detector.py 中

# 固定随机种子以保证测试可复现
random.seed(42)


def draw_random_number_in_region(image: np.ndarray, region: Tuple[int, int, int, int]) -> int:
    """
    在给定的区域内绘制一个随机的两位数（0~99），并返回这个数字。
    区域格式为 (x1, y1, x2, y2)。
    为确保 OCR 能识别，该数字以白色绘制，字体和大小需适当。
    """
    number = random.randint(0, 99)
    text = str(number)
    font = cv2.FONT_HERSHEY_SIMPLEX
    x1, y1, x2, y2 = region
    width = x2 - x1
    height = y2 - y1

    # 设置字体大小与粗细，确保数字能在区域内清晰显示
    font_scale = 0.7
    thickness = 2
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_w, text_h = text_size
    # 计算文本的起始坐标，使数字大致居中
    origin = (x1 + (width - text_w) // 2, y1 + (height + text_h) // 2)
    # 在图像上绘制白色文本
    cv2.putText(image, text, origin, font, font_scale, (255, 255, 255), thickness)
    return number


@pytest.fixture
def dummy_image():
    """
    创建一幅 800×1500 的测试图像，背景设为暗灰色（(50,50,50)），这样经过 HSV 转换与 inRange 后背景为黑。
    """
    img = np.full((720, 1600, 3), 50, dtype=np.uint8)
    return img


def test_get_skill_cooldown_with_random_numbers(dummy_image):
    """
    测试 get_skill_cooldown 方法：
    - 对于技能配置中要求冷却的技能（技能1、3、6），在其区域内随机绘制一个两位数，
      并将对应位置像素设置为低值（例如 100）模拟冷却状态。
    - 对于技能2、5，设置为正常状态（像素值 255），不绘制数字。
    - 验证 get_skill_cooldown 返回的字典中，冷却状态技能的值与绘制的数字一致，其它技能返回 0。
    """
    detector = CooldownDetector(sample_size=1)
    # 设置窗口信息为固定值，保证 _find_window 不被调用（或已经调用过）
    detector.skills_config = detector.DEFAULT_SKILLS_CONFIG.copy()

    # 模拟各技能对应的关键像素（position）:
    # - 技能1、3、6：设为 100 表示冷却（<= threshold 200）
    # - 技能2、5：设为 255 表示正常 (> threshold 200)
    for skill_id, config in detector.skills_config.items():
        x, y = config.position
        if skill_id in [1, 3, 6]:
            dummy_image[y, x] = 100  # 冷却状态
        else:
            dummy_image[y, x] = 255  # 正常状态

    # 对于处于冷却状态的技能，在其 region 内绘制随机两位数，并记录预期冷却值
    expected_cooldowns = {}
    for skill_id in [1, 3, 6]:
        config: SkillConfig = detector.skills_config[skill_id]
        # 在图像中绘制随机数字
        drawn_number = draw_random_number_in_region(dummy_image, config.region)
        expected_cooldowns[skill_id] = drawn_number

    # 调用 get_skill_cooldown，注意该方法会调用 check_skills_status 并调用 get_cool_down 对冷却技能读取数字
    actual_cooldowns = detector.get_skill_cooldown(dummy_image)

    # 验证：
    # 对于冷却状态技能（1,3,6），预期返回值为绘制的随机数字；
    # 对于技能2、5（正常）和技能4（未配置），预期返回 0。
    for skill_id, expected in expected_cooldowns.items():
        assert actual_cooldowns[
                   skill_id] == expected, f"技能 {skill_id} 预期冷却 {expected}，实际 {actual_cooldowns[skill_id]}"
    for skill_id in [2, 5, 4]:
        assert actual_cooldowns[skill_id] == 0, f"技能 {skill_id} 预期冷却 0，实际 {actual_cooldowns[skill_id]}"


if __name__ == "__main__":
    # 运行测试
    pytest.main(["-v", "--tb=short"])
