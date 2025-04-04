import cv2
import win32gui
import win32ui
import win32con
import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass
from new_wzry_ai.config.default_config import GameConfig

@dataclass
class WindowInfo:
    parent_hwnd: int
    child_hwnd: int
    rect: Tuple[int, int, int, int]

class ScreenCaptureError(Exception):
    """截图相关异常"""
    pass

class ScreenCapture:
    def __init__(self, window_title: str = GameConfig.WINDOW_TITLE):
        self.window_title = window_title
        self._window_info: Optional[WindowInfo] = None

    def _find_window(self) -> WindowInfo:
        parent_hwnd = win32gui.FindWindow(None, self.window_title)
        if not parent_hwnd:
            raise ScreenCaptureError(f'未找到{self.window_title}窗口')

        child_hwnd = None

        def callback(hwnd, _):
            nonlocal child_hwnd
            if win32gui.GetClassName(hwnd) == 'nemuwin':
                child_hwnd = hwnd
                return False
            return True

        win32gui.EnumChildWindows(parent_hwnd, callback, None)
        if not child_hwnd:
            raise ScreenCaptureError('未找到模拟器显示窗口')

        window_rect = win32gui.GetWindowRect(child_hwnd)
        return WindowInfo(parent_hwnd, child_hwnd, window_rect)

    def _get_absolute_position(self, 
                             pos: Optional[Tuple[int, int, int, int]] = None
                             ) -> Tuple[int, int, int, int]:
        if not self._window_info:
            self._window_info = self._find_window()

        window_left, window_top, window_right, window_bottom = self._window_info.rect

        if pos:
            left, top, right, bottom = pos
            return (
                left + window_left,
                top + window_top,
                right - left,
                bottom - top
            )

        return (
            window_left,
            window_top,
            window_right - window_left,
            window_bottom - window_top
        )

    def capture(self, 
                pos: Optional[Tuple[int, int, int, int]] = None
                ) -> Optional[np.ndarray]:
        if not self._window_info:
            self._window_info = self._find_window()

        abs_left, abs_top, width, height = self._get_absolute_position(pos)
        #print(f"{abs_left},{abs_top},{width},{height}")
        hwnd_dc = win32gui.GetWindowDC(self._window_info.child_hwnd)
        mfc_dc = win32ui.CreateDCFromHandle(hwnd_dc)
        save_dc = mfc_dc.CreateCompatibleDC()
        save_bitmap = win32ui.CreateBitmap()
        save_bitmap.CreateCompatibleBitmap(mfc_dc, width, height)
        save_dc.SelectObject(save_bitmap)

        try:
            # 捕获屏幕内容
            save_dc.BitBlt(
                (0, 0),
                (width, height),
                mfc_dc,
                (abs_left - self._window_info.rect[0], 
                 abs_top - self._window_info.rect[1]),
                win32con.SRCCOPY
            )

            # 转换为numpy数组
            bmpstr = save_bitmap.GetBitmapBits(True)
            img = np.frombuffer(bmpstr, dtype='uint8')
            img.shape = (height, width, 4)
            return img[:, :, :3]  # 返回BGR格式

        except Exception as e:
            raise ScreenCaptureError(f"截图失败: {str(e)}")

        finally:
            # 清理资源
            win32gui.DeleteObject(save_bitmap.GetHandle())
            save_dc.DeleteDC()
            mfc_dc.DeleteDC()
            win32gui.ReleaseDC(self._window_info.child_hwnd, hwnd_dc) 