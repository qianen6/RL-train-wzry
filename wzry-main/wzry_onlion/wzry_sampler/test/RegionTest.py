import cv2
import numpy as np
import os

# 全局变量
dragging = False
dragging_region = None
drag_offset = (0, 0)

resizing = False
resizing_region = None
resizing_corner = None

threshold = 10  # 调整大小的角点检测阈值（像素）
save_folder = "template"  # 截图保存的文件夹

# 确保保存文件夹存在
os.makedirs(save_folder, exist_ok=True)

frame = None  # 全局存储当前帧

# 计算当前已有的最大序号
def get_next_filename():
    existing_files = [f for f in os.listdir(save_folder) if f.startswith("region_") and f.endswith(".png")]
    existing_numbers = [int(f[7:-4]) for f in existing_files if f[7:-4].isdigit()]
    next_number = max(existing_numbers, default=0) + 1
    return os.path.join(save_folder, f"region_{next_number:03d}.png")

def save_cropped_region(region_coords):
    """保存裁剪后的灰度区域"""
    global frame  # 获取最新的 frame
    if frame is None:
        print("⚠ Warning: Frame is None, skipping save.")
        return

    x1, y1, x2, y2 = region_coords
    cropped = frame[y1:y2, x1:x2]  # 截取区域
    gray_cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)  # 转换为灰度

    save_path = get_next_filename()
    cv2.imwrite(save_path, gray_cropped)  # 保存灰度图像
    print(f"✅ Saved gray cropped region: {save_path}")


def mouse_callback(event, x, y, flags, param):
    global dragging, dragging_region, drag_offset
    global resizing, resizing_region, resizing_corner
    global regions

    if event == cv2.EVENT_LBUTTONDOWN:
        for name, rect in regions.items():
            x1, y1, x2, y2 = rect
            corners = {
                "tl": (x1, y1),
                "tr": (x2, y1),
                "bl": (x1, y2),
                "br": (x2, y2)
            }
            for corner, (cx, cy) in corners.items():
                if ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5 < threshold:
                    resizing = True
                    resizing_region = name
                    resizing_corner = corner
                    return

        for name, rect in regions.items():
            x1, y1, x2, y2 = rect
            if x1 <= x <= x2 and y1 <= y <= y2:
                dragging = True
                dragging_region = name
                drag_offset = (x - x1, y - y1)
                return

    elif event == cv2.EVENT_MOUSEMOVE:
        if resizing and resizing_region:
            x1, y1, x2, y2 = regions[resizing_region]
            if resizing_corner == "tl":
                regions[resizing_region] = [min(x, x2 - 1), min(y, y2 - 1), x2, y2]
            elif resizing_corner == "tr":
                regions[resizing_region] = [x1, min(y, y2 - 1), max(x, x1 + 1), y2]
            elif resizing_corner == "bl":
                regions[resizing_region] = [min(x, x2 - 1), y1, x2, max(y, y1 + 1)]
            elif resizing_corner == "br":
                regions[resizing_region] = [x1, y1, max(x, x1 + 1), max(y, y1 + 1)]

        elif dragging and dragging_region:
            rect = regions[dragging_region]
            width, height = rect[2] - rect[0], rect[3] - rect[1]
            regions[dragging_region] = [x - drag_offset[0], y - drag_offset[1], x - drag_offset[0] + width, y - drag_offset[1] + height]

    elif event == cv2.EVENT_LBUTTONUP:
        if resizing and resizing_region:
            print(f"🔄 Updated {resizing_region} (resized): ({regions[resizing_region][0]}, {regions[resizing_region][1]}, {regions[resizing_region][2]}, {regions[resizing_region][3]})")
            save_cropped_region(regions[resizing_region])  # 保存截图
            resizing = False
            resizing_region = None
            resizing_corner = None
        elif dragging and dragging_region:
            print(f"🔄 Updated {dragging_region} (dragged): ({regions[dragging_region][0]}, {regions[dragging_region][1]}, {regions[dragging_region][2]}, {regions[dragging_region][3]})")
            save_cropped_region(regions[dragging_region])  # 保存截图
            dragging = False
            dragging_region = None

if __name__ == "__main__":
    from new_wzry_ai.utils.screen import ScreenCapture
    from new_wzry_ai.config.default_config import TemplateRegion

    # 初始化区域
    regions = {name: list(rect) for name, rect in TemplateRegion.__dict__.items() if not name.startswith("__")}

    screen = ScreenCapture()

    cv2.namedWindow("Annotated Frame")
    cv2.setMouseCallback("Annotated Frame", mouse_callback)

    while True:
        frame = screen.capture()  # 直接更新全局 frame
        if frame is None:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        annotated_frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        for name, (x1, y1, x2, y2) in regions.items():
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(annotated_frame, name, (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imshow("Annotated Frame", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
