from typing import Optional
import cv2
import numpy as np
import glob

from new_wzry_ai.utils.screen import ScreenCapture


def save_captured_image(image: np.ndarray, filename: str) -> None:
    """将捕获的图像保存到本地"""
    cv2.imwrite(filename, image)
    print(f"图片已保存至 {filename}")


def read_image(filename: str) -> Optional[np.ndarray]:
    """从本地读取图像"""
    img = cv2.imread(filename)
    if img is None:
        print(f"无法读取 {filename}")
    return img


def main():
    sc = ScreenCapture()

    # 初始化截图计数器
    existing_files = glob.glob("screenshot_*.png")
    screenshot_count = 0
    try:
        screenshot_count = max([int(f.split("_")[1].split(".")[0]) for f in existing_files])
    except (ValueError, IndexError):
        pass

    # 创建控制面板窗口
    cv2.namedWindow("Control Panel", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Control Panel", 400, 200)

    # 界面参数
    dummy = np.zeros((200, 400, 3), dtype=np.uint8)
    capture_btn = ((50, 30), (350, 80))
    quit_btn = ((50, 120), (350, 170))
    running = True

    def mouse_callback(event, x, y, flags, param):
        nonlocal screenshot_count, running
        if event == cv2.EVENT_LBUTTONDOWN:
            # 捕获按钮点击
            if (capture_btn[0][0] <= x <= capture_btn[1][0] and
                    capture_btn[0][1] <= y <= capture_btn[1][1]):
                try:
                    # 修改后的正确判断方式
                    captured_img = sc.capture()
                    if captured_img is not None and captured_img.any():  # 双重验证
                        screenshot_count += 1
                        filename = f"screenshot_{screenshot_count}.png"
                        save_captured_image(captured_img, filename)
                        # 显示截图预览（自动关闭旧窗口）
                        cv2.imshow("Captured", captured_img)
                        cv2.moveWindow("Captured", 400, 0)  # 将预览窗口移动到控制面板右侧
                    else:
                        print("捕获到空图像")
                except Exception as e:
                    print(f"截图异常: {str(e)}")
            # 退出按钮点击
            elif (quit_btn[0][0] <= x <= quit_btn[1][0] and
                  quit_btn[0][1] <= y <= quit_btn[1][1]):
                running = False

    cv2.setMouseCallback("Control Panel", mouse_callback)

    while running:
        # 绘制动态界面
        dummy.fill(230)  # 浅灰色背景
        # 绘制带阴影的按钮
        cv2.rectangle(dummy, (capture_btn[0][0] + 2, capture_btn[0][1] + 2),
                      (capture_btn[1][0] + 2, capture_btn[1][1] + 2), (100, 100, 100), -1)
        cv2.rectangle(dummy, capture_btn[0], capture_btn[1], (0, 200, 0), -1)
        cv2.putText(dummy, f"Capture ({screenshot_count + 1})", (60, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.rectangle(dummy, (quit_btn[0][0] + 2, quit_btn[0][1] + 2),
                      (quit_btn[1][0] + 2, quit_btn[1][1] + 2), (50, 50, 50), -1)
        cv2.rectangle(dummy, quit_btn[0], quit_btn[1], (0, 0, 200), -1)
        cv2.putText(dummy, "Quit", (60, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Control Panel", dummy)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            running = False

    cv2.destroyAllWindows()
    if screenshot_count > 0:
        if (img := read_image(f"screenshot_{screenshot_count}.png")) is not None:
            cv2.imshow("Last Capture", img)
            cv2.waitKey(2000)  # 显示2秒后自动关闭


if __name__ == "__main__":
    main()