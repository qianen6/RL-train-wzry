import base64
from io import BytesIO
import numpy as np
from PIL import Image

def image_to_base64(image):
    """
    将图像对象转换为 Base64 编码字符串。
    :param image: 图像对象，可以是 PIL.Image 或其他支持的图像对象
    :return: Base64 编码字符串
    """
    if isinstance(image, np.ndarray):  # 检查是否是 numpy.ndarray
        image = Image.fromarray(image)  # 转换为 PIL.Image 对象
    buffered = BytesIO()
    image.save(buffered, format="PNG")  # 根据需要修改格式
    return base64.b64encode(buffered.getvalue()).decode("utf-8")