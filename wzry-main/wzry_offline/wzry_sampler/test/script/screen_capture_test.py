import unittest
import win32gui
import win32con
import cv2
import numpy as np
from unittest.mock import patch
from new_wzry_ai.utils.screen import ScreenCapture, ScreenCaptureError


class TestScreenCapture(unittest.TestCase):
    def setUp(self):
        # 创建一个 dummy 父窗口（使用系统内置的 Static 类）
        self.window_title = "wzry_ai"
        self.parent_hwnd = win32gui.CreateWindowEx(
            0, "Static", self.window_title,
            win32con.WS_OVERLAPPEDWINDOW,
            0, 0, 800, 600, 0, 0, 0, None
        )
        # 创建一个 dummy 子窗口，设置为父窗口的子窗口
        self.child_hwnd = win32gui.CreateWindowEx(
            0, "Static", "nemuwin",  # 注意：这里为了让 GetClassName 返回 'nemuwin'
            win32con.WS_CHILD,
            0, 0, 800, 600, self.parent_hwnd, 0, 0, None
        )

    def tearDown(self):
        # 销毁创建的窗口
        try:
            win32gui.DestroyWindow(self.child_hwnd)
        except Exception:
            pass
        try:
            win32gui.DestroyWindow(self.parent_hwnd)
        except Exception:
            pass

    @patch('win32gui.GetClassName')
    def test_full_screen_capture(self, mock_get_class_name):
        # 当传入的句柄为子窗口句柄时，返回 'nemuwin'
        def get_class(hwnd):
            if hwnd == self.child_hwnd:
                return 'nemuwin'
            return 'Static'

        mock_get_class_name.side_effect = get_class

        with patch('win32gui.FindWindow', return_value=self.parent_hwnd), \
                patch('win32gui.EnumChildWindows',
                      side_effect=lambda hwnd, callback, param: callback(self.child_hwnd, param) or True), \
                patch('win32gui.GetWindowRect', return_value=(0, 0, 800, 600)), \
                patch('win32gui.GetWindowDC', return_value=54321), \
                patch('win32ui.CreateDCFromHandle') as mock_create_dc, \
                patch('win32ui.CreateBitmap') as mock_create_bitmap:
            # 模拟设备上下文和位图
            mock_mfc_dc = unittest.mock.MagicMock()
            mock_create_dc.return_value = mock_mfc_dc
            mock_save_dc = unittest.mock.MagicMock()
            mock_mfc_dc.CreateCompatibleDC.return_value = mock_save_dc
            mock_bitmap = unittest.mock.MagicMock()
            mock_create_bitmap.return_value = mock_bitmap
            # 构造一幅全白图像，尺寸为 (600,800,4)
            white_image = np.ones((600, 800, 4), dtype='uint8') * 255
            mock_bitmap.GetBitmapBits.return_value = white_image.tobytes()

            # 实例化 ScreenCapture 并调用 capture 方法
            capture = ScreenCapture(self.window_title)
            img = capture.capture()

            # 断言返回的图像尺寸与颜色
            self.assertEqual(img.shape, (600, 800, 3))
            self.assertTrue(np.all(img == 255))


if __name__ == '__main__':
    unittest.main()
