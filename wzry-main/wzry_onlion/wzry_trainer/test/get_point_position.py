import cv2
from PIL import Image, ImageTk

import tkinter as tk

from new_wzry_ai.utils.screen import ScreenCapture


def on_click(event):
    """鼠标点击事件回调函数"""
    print(f"点击坐标：x={event.x}, y={event.y}")

def main():
    # 创建屏幕捕获对象
    screen = ScreenCapture()
    image = screen.capture()

    # 将 NumPy 数组转换为 PIL Image（注意：需要先将 BGR 转换为 RGB）
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # 创建 Tkinter 主窗口
    root = tk.Tk()
    root.title("点击查看坐标")

    # 创建画布并将画布大小设为图片尺寸
    canvas = tk.Canvas(root, width=pil_image.width, height=pil_image.height)
    canvas.pack()

    # 转换为 Tkinter 可用的图片格式
    tk_image = ImageTk.PhotoImage(pil_image)

    # 在画布上显示图片，并保存图片 id 以便后续更新
    image_id = canvas.create_image(0, 0, anchor=tk.NW, image=tk_image)
    # 保持引用，防止被垃圾回收
    canvas.image = tk_image

    # 绑定鼠标点击事件
    canvas.bind("<Button-1>", on_click)

    # 定义刷新函数，用于重新捕获屏幕并更新画布上的图片
    def refresh():
        new_image = screen.capture()
        new_pil_image = Image.fromarray(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
        new_tk_image = ImageTk.PhotoImage(new_pil_image)
        # 更新画布上的图片
        canvas.itemconfig(image_id, image=new_tk_image)
        # 更新引用
        canvas.image = new_tk_image

    # 添加刷新按钮，点击后调用 refresh 函数
    refresh_button = tk.Button(root, text="刷新", command=refresh)
    refresh_button.pack()

    root.mainloop()

if __name__ == "__main__":
    main()
