import pytesseract
from PIL import Image
import numpy as np
from typing import Tuple, Optional
from PIL import Image, ImageEnhance, ImageFilter
from new_wzry_ai.utils.screen import ScreenCapture
import time
import os
import cv2

class EconomicGrowth:
    def __init__(self, region1: Tuple[int, int, int, int] = (1436, 38, 1492, 58), region2: Tuple[int, int, int, int] = (1206, 27, 1230, 49)):
        self.screen = ScreenCapture()
        """
        初始化经济增长检测器
        :param region: 指定区域的坐标 (left, upper, right, lower)，默认值为示例区域
        """
        self.region1 = region1
        self.region2 = region2

    def _get_roi(self, image: np.ndarray, pos: Optional[Tuple[int, int, int, int]]) -> Image:
        """获取感兴趣区域 (ROI)"""
        if pos is None:
            return Image.fromarray(image)  # 如果没有区域，直接返回整个图像作为 PIL 图像

        x1, y1, x2, y2 = pos
        # 提取图像的 ROI 区域
        roi = image[y1:y2, x1:x2]

        # 转换为 PIL 图像并返回
        return Image.fromarray(roi)

    def save_roi_image(self, roi_image: Image):
        """将 ROI 图像保存到当前目录的 image 文件夹中"""
        # 确保 image 文件夹存在
        output_dir = './image'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 使用时间戳作为文件名，避免文件名重复
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        file_path = os.path.join(output_dir, f"roi_{timestamp}.png")

        # 保存图像
        roi_image.save(file_path)
        print(f"保存ROI图像到: {file_path}")

    def preprocess_image(self, roi_image: Image) -> Image:
        """对图像进行预处理，二值化和增强"""
        # 转换为灰度图像
        gray_image = roi_image.convert('L')

        # 应用二值化处理，使用固定阈值
        threshold = 150  # 你可以根据需要调整这个值
        binary_image = gray_image.point(lambda p: p > threshold and 255)

        # 增强图像对比度（如果需要）
        enhancer = ImageEnhance.Contrast(binary_image)
        enhanced_image = enhancer.enhance(2.0)  # 增强倍数可以根据需要调整

        # 返回处理后的图像
        return enhanced_image

    def get_economic_value(self, image: np.ndarray) -> int:
        """
        获取图像中指定区域的经济数字
        :param image: 传入的图像（BGR 格式）
        :return: 识别出的经济数字
        """
        # 获取 ROI 图像
        roi_image = self._get_roi(image, self.region1)
        processed_image = self.preprocess_image(roi_image)
        # self.save_roi_image(processed_image)
        # 使用 pytesseract 进行 OCR 识别，识别数字
        text = pytesseract.image_to_string(processed_image, config='--psm 6 outputbase digi')

        # 如果无法识别任何内容，返回 0
        try:
            return int(text.strip()) if text.strip().isdigit() else 0
        except ValueError:
            return 0

    def get_kills_value(self, image: np.ndarray) -> int:
        """
        获取图像中指定区域的玩家击杀数(个人击杀数)
        :param image: 传入的图像（BGR 格式）
        :return: 玩家的击杀数
        """
        # 获取 ROI 图像
        roi_image = self._get_roi(image, self.region2)
        processed_image = self.preprocess_image(roi_image)
        # self.save_roi_image(processed_image)
        # 使用 pytesseract 进行 OCR 识别，识别数字
        text = pytesseract.image_to_string(processed_image, config='--psm 6 outputbase digi')

        # 如果无法识别任何内容，返回 0
        try:
            return int(text.strip()) if text.strip().isdigit() else 0
        except ValueError:
            return 0

def main():
    # 捕获窗口截图
    screen=ScreenCapture()
    economic_growth = EconomicGrowth()
    while(True):

        image = screen.capture()
        economic_value = economic_growth.get_economic_value(image)
        kills_value = economic_growth.get_kills_value(image)
        # 打印 ROI 和识别的经济值
        print(f"识别的经济值：{economic_value}")
        print(f"kills value: {kills_value}")
        time.sleep(1)





if __name__ == "__main__":
    main()