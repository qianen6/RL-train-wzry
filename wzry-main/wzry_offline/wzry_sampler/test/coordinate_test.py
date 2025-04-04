import cv2
import time

from new_wzry_ai.config.default_config import TemplateRegion
from new_wzry_ai.utils.screen import ScreenCapture


def show_regions_live():
    """
    实时捕获屏幕并显示带有区域标记的图像
    """
    # 创建窗口
    cv2.namedWindow("Screen with Regions", cv2.WINDOW_NORMAL)
    screen = ScreenCapture()
    # 定义颜色和线宽
    color = (0, 255, 0)  # 绿色
    thickness = 2  # 线宽

    # 获取所有区域
    regions = [
        ("MONSTER_AREA", TemplateRegion.MONSTER_AREA),
        ("BLOOD_AREA", TemplateRegion.BLOOD_AREA),
        ("ENEMY_BLOOD_AREA", TemplateRegion.ENEMY_BLOOD_AREA),
        ("CHARACTER_AREA", TemplateRegion.CHARACTER_AREA),
        ("GAME_START_AREA", TemplateRegion.GAME_START_AREA),
        ("VICTORY_AREA", TemplateRegion.VICTORY_AREA),
        ("DEFEAT_AREA", TemplateRegion.DEFEAT_AREA),
        ("BACK_HOME_AREA", TemplateRegion.BACK_HOME_AREA),
        ("MODE_AREA", TemplateRegion.MODE_AREA),
        ("START_AREA", TemplateRegion.START_AREA),
        ("ICON_AREA", TemplateRegion.ICON_AREA),
        ("FAYULU_AREA", TemplateRegion.FAYULU_AREA),
        ("HOUYI_ICO_AREA", TemplateRegion.HOUYI_ICO_AREA),
        ("CONFIRM_AREA", TemplateRegion.CONFIRM_AREA),
        ("SELF_DEATH_AREA1", TemplateRegion.SELF_DEATH_AREA1),
        ("SELF_DEATH_AREA2", TemplateRegion.SELF_DEATH_AREA2)
    ]

    try:
        while True:
            # 捕获屏幕
            image = screen.capture()
            if image is None:
                print("无法捕获屏幕内容")
                break

            # 创建副本用于绘制
            image = image.copy()

            # 绘制所有区域
            for name, (x1, y1, x2, y2) in regions:
                # 绘制矩形
                cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

                # 添加标签
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                cv2.putText(image, name, (x1, y1 - 5), font, font_scale, color, thickness)

            # 显示图像
            cv2.imshow("Screen with Regions", image)

            # 检查用户是否按下了ESC键
            if cv2.waitKey(1) == 27:  # 27是ESC键的ASCII码
                break

            # 控制帧率，避免CPU占用过高
            time.sleep(0.05)

    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()


# 使用示例
if __name__ == "__main__":
    show_regions_live()