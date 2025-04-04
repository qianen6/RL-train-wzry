import cv2
import numpy as np
from new_wzry_ai.utils.screen import ScreenCapture
from new_wzry_ai.utils.skill_cooldown import CooldownDetector
from typing import Dict


# 假设 SkillConfig 类定义
class SkillConfig:
    def __init__(self, region: tuple, radius: int, box: tuple):
        self.region = region  # 区域坐标
        self.radius = radius  # 半径
        self.box = box  # 截取区域的坐标


# 默认技能配置
DEFAULT_SKILLS_CONFIG: Dict[int, SkillConfig] = {
    1: SkillConfig((1219, 649), 200, (1180, 615, 1223, 650)),  # 1技能
    2: SkillConfig((1290, 508), 200, (1258, 479, 1301, 514)),  # 2技能
    3: SkillConfig((1395, 421), 200, (1391, 403, 1434, 438)),  # 3技能
    5: SkillConfig((974, 665), 200, (967, 631, 1011, 660)),  # 恢复
    6: SkillConfig((1061, 665), 200, (1062, 630, 1118, 661)),  # 闪现
}

if __name__ == '__main__':
    screen = ScreenCapture()
    cooldown = CooldownDetector()

    while True:
        # 捕获屏幕
        image = screen.capture()

        skills_cooldown = cooldown.get_skill_cooldown(image)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        lower_white = np.array([0, 0, 200])  # 低值

        upper_white = np.array([180, 20, 255])  # 高值

        image = cv2.inRange(image, lower_white, upper_white)

        # 创建空白的拼接图像
        all_skills_image = []

        # 获取所有图像的最大高度
        max_height = 0
        skill_images = []

        for skill_id, skill_config in DEFAULT_SKILLS_CONFIG.items():
            # 根据 box 截取区域
            x1, y1, x2, y2 = skill_config.box
            skill_image = image[y1:y2, x1:x2]
            skill_images.append(skill_image)

            # 计算当前技能图像的高度，更新最大高度
            max_height = max(max_height, skill_image.shape[0])

        # 调整所有图像的高度到最大高度，保持宽度比例
        for i in range(len(skill_images)):
            height, width = skill_images[i].shape[:2]
            new_width = int(width * (max_height / height))
            resized_image = cv2.resize(skill_images[i], (new_width, max_height))
            all_skills_image.append(resized_image)

        # 拼接所有技能的图像
        concatenated_image = np.hstack(all_skills_image)  # 横向拼接


        for key,value in skills_cooldown.items():
            print(str(key) + " " + str(value))

        # 显示拼接后的图像
        cv2.imshow("Skills", concatenated_image)

        # 按 'q' 键退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
