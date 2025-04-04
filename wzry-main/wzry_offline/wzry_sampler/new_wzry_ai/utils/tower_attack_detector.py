import cv2
import numpy as np
from dataclasses import dataclass, field
import math


@dataclass
class TowerAttackDetectorConfig:
    # 红色范围（覆盖低饱和度警示）
    lower_red: np.ndarray = field(default_factory=lambda: np.array([0, 220, 170]))  # 扩展色相和饱和度范围
    upper_red: np.ndarray = field(default_factory=lambda: np.array([6, 255, 255]))

    # 其他参数
    red_ratio_threshold: float = 0.3  # 提高阈值避免误检
    min_contour_area: int = 5  # 最小轮廓面积（过滤噪点）


class TowerAttackDetector:
    def __init__(self, config: TowerAttackDetectorConfig = TowerAttackDetectorConfig()):
        self.config = config
        self.confidence = 0.8
        self.red_point_detect_area = (797, 368, 802, 372)  # ROI区域坐标
        self.red_circle_detect_area = (772, 343, 825, 395)

    def _get_red_point_roi(self, frame):
        """获取检测区域"""
        x1, y1, x2, y2 = self.red_point_detect_area
        return frame[y1:y2, x1:x2]

    def _create_red_mask(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        return cv2.inRange(hsv, self.config.lower_red, self.config.upper_red)

    def _get_red_circle_roi(self, frame):
        x1, y1, x2, y2 = self.red_circle_detect_area
        return frame[y1:y2, x1:x2]

    def has_red_point(self, red_mask: np.ndarray) -> bool:
        """检测指定区域是否包含红色像素点"""
        red_mask_roi = self._get_red_point_roi(red_mask)

        pixels = cv2.countNonZero(red_mask_roi)
        roi_area = red_mask_roi.shape[0] * red_mask_roi.shape[1]
        red_ratio = pixels / (roi_area + 1e-6)

        return red_ratio > self.config.red_ratio_threshold

    def is_red_line_through_point(self, red_mask, point, visual_debug=False):
        """
        检测图像中是否存在红色直线穿过指定点
        :param red_mask: 红色掩膜
        :param point: 目标点坐标 (x, y)
        :param visual_debug: 是否显示调试视图
        :return: (bool, 可视化图像)
        """
        # 形态学优化（连接断裂区域）
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        processed_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)

        # 边缘检测（降低阈值以检测弱边缘）
        edges = cv2.Canny(processed_mask, 50, 150)

        # 霍夫直线检测参数
        rho_resolution = 1  # 像素精度
        angle_resolution = np.pi / 180  # 角度精度（1度）
        hough_threshold = 30  # 投票阈值
        min_line_length = 15  # 最小线段长度
        max_line_gap = 20  # 最大线段间隔

        # 执行直线检测
        lines = cv2.HoughLinesP(
            edges,
            rho=rho_resolution,
            theta=angle_resolution,
            threshold=hough_threshold,
            minLineLength=min_line_length,
            maxLineGap=max_line_gap
        )

        found = False
        debug_frame = red_mask.copy()

        # 在调试视图中标记目标点
        cv2.circle(debug_frame, point, 8, (0, 255, 0), -1)

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]

                # 计算直线方程：ax + by + c = 0
                a = y2 - y1
                b = x1 - x2
                c = x2 * y1 - x1 * y2

                # 计算点到直线的距离
                px, py = point
                distance = abs(a * px + b * py + c) / math.sqrt(a ** 2 + b ** 2)

                # 距离阈值（根据图像分辨率调整）
                distance_threshold = 3.0

                # 判断是否在直线上
                if distance <= distance_threshold:
                    found = True
                    if visual_debug:
                        # 绘制检测到的直线
                        cv2.line(debug_frame, (x1, y1), (x2, y2), (0, 0, 0), 2)
                        # 绘制距离标记
                        cv2.putText(debug_frame, f"{distance:.1f}px",
                                  (px + 10, py - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                  0.6, (255, 255, 255), 2)

        return found, debug_frame

    def detect(self, frame: np.ndarray) -> bool:
        if frame is None:
            return False

        red_mask = self._create_red_mask(frame)
        # 基础红点检测
        current_status = self.has_red_point(red_mask)

        # 如果红点检测失败，执行附加检测
        if not current_status:
            current_status, _ = self.is_red_line_through_point(red_mask, (800, 360))

        return current_status

    def visualize_roi(self, frame: np.ndarray) -> np.ndarray:
        """只显示ROI区域"""
        roi = self._create_red_mask(frame)
        return roi

# 使用示例
if __name__ == "__main__":
    from new_wzry_ai.utils.screen import ScreenCapture
    detector = TowerAttackDetector()
    screen = ScreenCapture()
    while True:
        frame = screen.capture()
        if frame is None:
            continue

        attack_detected = detector.detect(frame)
        print(attack_detected)

        key = cv2.waitKey(1)
        if key == 27:  # ESC退出
            break
    cv2.destroyAllWindows()