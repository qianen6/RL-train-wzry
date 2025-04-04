import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
from dataclasses import dataclass
from typing import Final, Tuple

from new_wzry_ai.config.default_config import TemplateRegion
from new_wzry_ai.utils.screen import ScreenCapture



class CaptureApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("区域截图工具")
        self.setGeometry(100, 100, 800, 600)

        # 初始化屏幕捕获
        self.screen = ScreenCapture()

        # 主窗口组件
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # 创建布局
        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)

        # 截图按钮
        self.capture_button = QPushButton("获取", self)
        self.capture_button.clicked.connect(self.capture_defeat_area)
        self.layout.addWidget(self.capture_button)

        # 显示区域
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.image_label)

    def capture_defeat_area(self):
        """捕获DEFEAT_AREA区域并保存"""
        try:
            # 获取屏幕截图
            image = self.screen.capture()

            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # 检查图像是否有效
            if image is None or image.size == 0:
                print("获取屏幕截图失败")
                return

            # 获取DEFEAT_AREA区域
            x1, y1, x2, y2 = TemplateRegion.CHARACTER_AREA
            defeat_area = image[y1:y2, x1:x2]

            # 保存图像
            cv2.imwrite("minimap.png", defeat_area)

            # 在窗口中显示截图
            self.display_image(defeat_area)

        except Exception as e:
            print(f"截图失败: {e}")

    def display_image(self, image):
        """在QLabel中显示图像"""
        # 将BGR转换为RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, channel = image_rgb.shape
        bytes_per_line = 3 * width
        q_img = QImage(image_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(q_img))


if __name__ == "__main__":
    app = QApplication([])
    window = CaptureApp()
    window.show()
    app.exec_()