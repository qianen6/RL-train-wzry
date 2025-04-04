from PIL.ImageChops import screen

from new_wzry_ai.config.default_config import TemplateRegion
from new_wzry_ai.utils.heros_position_detector import HeroPositionDetector
from new_wzry_ai.utils.screen import ScreenCapture

if __name__ == '__main__':
    screen = ScreenCapture()
    hero_position_detector = HeroPositionDetector()
    while True:
        image = screen.capture()
        x1,y1,x2,y2 = TemplateRegion.CHARACTER_AREA
        image = image[y1:y2,x1:x2]
        p1,p2,p3 = hero_position_detector.get_hero_position(image)
        print(p1,p2,p3)

