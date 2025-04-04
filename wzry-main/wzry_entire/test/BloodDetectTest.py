import time

from PIL.ImageChops import screen

from new_wzry_ai.config.default_config import TemplateRegion
from new_wzry_ai.utils.blood_detect import BloodDetector
from new_wzry_ai.utils.screen import ScreenCapture

if __name__ == '__main__':
    screen = ScreenCapture()
    blood_detector = BloodDetector()
    while True:
        image = screen.capture()

        print(blood_detector.get_self_blood(image))
        print(blood_detector.get_enemy_blood(image))
        print(blood_detector.is_enemy_blood_changed(image))
        time.sleep(1)

