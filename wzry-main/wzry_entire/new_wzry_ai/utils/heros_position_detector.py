import math
import os

import torch
from ultralytics import YOLO
import cv2
import numpy as np
import math
from new_wzry_ai.config.default_config import TrainingConfig,TemplateRegion
import logging
logging.getLogger("ultralytics").setLevel(logging.ERROR)

class HeroPositionDetector:
    def __init__(self):
        # 加载模型
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = YOLO(TrainingConfig.YOLO_MODEL_PATH).to(self.device)  # 增加模型设备转移
        self.class_names= ["g", "b", "r"]  # 假设类别索引 0 对应 g(my)，1 对应 b(teammate)，2 对应 r(enemy)
        self.x_offset= -3
        self.y_offset = 2

    def get_hero_position(self,image):
        x1, y1, x2, y2 = TemplateRegion.CHARACTER_AREA
        # 保存 ROI 图像到指定文件夹
        frame = np.array(image, dtype=np.uint8)

        frame = torch.tensor(frame, dtype=torch.float32).to(self.device)  # 更高效的转换方式

        frame = frame.permute(2, 0, 1)  # 添加通道维度调整 (HWC -> CHW)

        frame = frame.unsqueeze(0)  # 添加batch维度
        results = self.model(frame)
        my_position = None  # 存储 g 的中心点
        teammate_position = []  # 存储所有 b 的中心点
        enemy_position = []  # 存储所有 r 的中心点


        try:
            # 获取检测框和跟踪ID（检查是否有检测结果）
            if results[0].boxes is not None and len(results[0].boxes) > 0:
                boxes = results[0].boxes.xywh.cpu().numpy()  # 获取检测框
                class_ids = results[0].boxes.cls.int().cpu().tolist()  # 获取分类ID
                # 遍历每个检测到的对象
                for i, box in enumerate(boxes):
                    # 获取中心点 (x, y)
                    center_x, center_y = box[0], box[1]
                    class_id = class_ids[i]  # 获取分类ID
                    class_name = self.class_names[class_id]  # 获取分类名称

                    # 如果是 g 类别，记录其中心点
                    if class_name == "g":
                        my_position = [center_x+x1+self.x_offset, center_y+y1+self.y_offset]

                    # 如果是 b 类别，保存 b 的中心点
                    elif class_name == "b":
                        teammate_position.append([center_x+x1+self.x_offset, center_y+y1+self.y_offset])
                        # 如果是 b 类别，保存 b 的中心点
                    elif class_name == "r":
                        enemy_position.append([center_x+x1+self.x_offset, center_y+y1+self.y_offset])
            return my_position,teammate_position,enemy_position
        except Exception as e:
            print("================== error ========================")
            print(e)
            print("================== error ========================")
            return None,[],[]

    def get_ally_positions(self,image,radius):#获取在radius范围内的英雄位置与数量
        my_team=[]#存储己方英雄与我的位置
        my_enemy=[]#存储敌方英雄与我的位置
        my_team2=[]#存储在范围内的己方英雄与我的位置
        my_enemy2=[]#存储在范围内的敌方英雄与我的位置
        my_teamnum=[]#存储在范围内的己方英雄数量
        my_enemynum=[]#存储在范围内的敌方英雄数量
        my_position,teammate_position,enemy_position=self.get_hero_position(image)#获取英雄位置
        #print(f"my_position: {my_position},teammate_position: {teammate_position},enemy_position: {enemy_position}")
        for item in teammate_position:
            my_team.append([math.sqrt((my_position[0] - teammate_position[0]) ** 2 + (my_position[1] - teammate_position[1]) ** 2)])#计算距离
        for item in enemy_position:
            my_enemy.append([math.sqrt((my_position[0] - enemy_position[0]) ** 2 + (my_position[1] - enemy_position[1]) ** 2)])
        my_team = sorted(my_team)#按照距离升序排序
        my_enemy = sorted(my_enemy)
        for item in my_team:
            if item <= radius:
                my_team2.append(item)
        for item in my_enemy:
            if item <= radius:  
                my_enemy2.append(item)  
        my_teamnum=len(my_team2)
        my_enemynum=len(my_enemy2)
        return my_team2,my_enemy2,my_teamnum,my_enemynum
        
hero_position_detector=HeroPositionDetector()

            