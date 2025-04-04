import cv2
from new_wzry_ai.utils.screen import ScreenCapture
from dataclasses import dataclass

@dataclass
class Destination:
    name: str
    position: tuple[int, int]
    radius: int

if __name__ == '__main__':
    screen = ScreenCapture()
    destinations = {
        "我方红buff": Destination("我方红buff", (203, 179), 13),
        "我方野怪鸡": Destination("我方野怪鸡", (232, 194), 10),
        "我方野怪刺猬": Destination("我方野怪刺猬", (166, 182), 10),
        "我方野怪猪": Destination("我方野怪猪", (191, 155), 10),
        "我方蓝buff": Destination("我方蓝buff", (143, 107), 13),
        "我方野怪双狼": Destination("我方野怪双狼", (128, 134), 10),
        "我方野怪蜥蜴": Destination("我方野怪蜥蜴", (112, 101), 10),
        "我方穿山甲": Destination("我方穿山甲", (121, 68), 10),
        "敌方红buff": Destination("敌方红buff", (187, 47), 13),
        "敌方野怪鸡": Destination("敌方野怪鸡", (157, 33), 10),
        "敌方野怪刺猬": Destination("敌方野怪刺猬", (225, 46), 10),
        "敌方野怪猪": Destination("敌方野怪猪", (201, 74), 10),
        "敌方蓝buff": Destination("敌方蓝buff", (247, 119), 13),
        "敌方野怪双狼": Destination("敌方野怪双狼", (263, 92), 10),
        "敌方野怪蜥蜴": Destination("敌方野怪蜥蜴", (277, 126), 10),
        "敌方穿山甲": Destination("敌方穿山甲", (270, 160), 10),
        "大龙": Destination("大龙", (157, 65), 15),
        "小龙": Destination("小龙", (235, 161), 15),
        "对抗路精灵": Destination("对抗路精灵", (95, 14), 10),
        "发育路小鸟": Destination("发育路小鸟", (289, 209), 10),
    }

    while True:
        image = screen.capture()
        image = image.copy() # Make a copy

        for dest in destinations.values():
            cv2.circle(image, dest.position, dest.radius, (0, 0, 255), 2)
            cv2.putText(image, dest.name, (dest.position[0] + 10, dest.position[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        cv2.imshow('Map Points', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()