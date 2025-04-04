from PIL import Image, ImageEnhance, ImageFilter
import pytesseract

# 读取图片
img = Image.open('test_img/img_2.png')

# # 应用二值化处理
# threshold = 130  # 阈值可以根据图像调整
# img = img.point(lambda p: p > threshold and 255)
#
# # 显示处理后的图像
# img.show()
#
# 使用 Tesseract 进行 OCR 识别，仅识别数字
text = pytesseract.image_to_string(img, config='--psm 6 outputbase digi')

# 输出结果
if not text:
    print("None")
else:
    print("识别的数字是:", text)
number = int(text)

