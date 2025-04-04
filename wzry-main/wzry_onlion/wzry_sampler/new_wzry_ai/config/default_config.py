from dataclasses import dataclass
from typing import Tuple, Final
import cv2
from enum import Enum



@dataclass(frozen=True)
class TrainingConfig:
    """训练相关配置"""
    BATCH_SIZE: Final[int] = 8#批次大小
    LEARNING_RATE: Final[float] = 0.0001#学习率
    GAMMA: Final[float] = 0.99#折扣因子
    EPSILON_START: Final[float] = 1.0#初始探索率
    EPSILON_MIN: Final[float] = 0.01#最小探索率
    EPSILON_DECAY: Final[float] = 0.9999#探索率衰减率
    UPDATE_TARGET_FREQ: Final[int] = 1000#更新目标网络频率
    MEMORY_SIZE: Final[int] = 100000#记忆库大小
    ALPHA: Final[float] = 0.6#优先级采样参数
    BETA_START: Final[float] = 0.4#重要性采样参数
    BETA_FRAMES: Final[int] = 100000#重要性采样帧数
    NUM_EPISODES: Final[int] = 10000#训练次数
    MAX_STEPS: Final[int] = 3000#最大步数
    SAVE_FREQ: Final[int] = 1#保存频率
    MODEL_PATH: Final[str] = '../models/wzry_model.pth'#模型保存路径
    YOLO_MODEL_PATH = 'runs/detect/train/weights/best.pt'#yolo模型路径


@dataclass(frozen=True)
class GameConfig:
    """游戏相关配置"""
    WINDOW_TITLE: Final[str] = 'wzry_ai'#窗口标题
    FRAME_STACK_SIZE: Final[int] = 4#帧栈大小
    FRAME_HEIGHT: Final[int] = 144#帧高度
    FRAME_WIDTH: Final[int] = 256#帧宽度
    MAX_WAIT_TIME: Final[int] = 300#最大等待时间
    RESPAWN_WAIT_TIME: Final[int] = 60#重生等待时间
    FRAME_TIME: Final[float] = 0.01#帧时间
    CorlorImage:bool=False  #帧栈中是否为彩色图像

@dataclass(frozen=False)
class Reward_Config:
    """奖励相关配置"""
    # 移动奖励
    MOVE_TO_TARGET = 1  # 向目标移动奖励
    MOVE_PENALTY = -0.3  # 远离目标惩罚
    MOVE_To_Edge=-10  #边界卡住惩罚

    # 攻击奖励 (前5分钟)
    ATTACK_MONSTER = 0.2  # 在范围内攻击野怪
    ATTACK_MONSTER_OUT = 0.0  # 脱战状态攻击野怪
    ATTACK_ENEMY = 3.0  # 攻击敌人

    # 攻击奖励 (5分钟后)
    LATE_ATTACK_MONSTER = 1.2  # 后期在范围内攻击野怪
    LATE_ATTACK_MONSTER_OUT = 0.2  # 后期脱战状态攻击野怪
    LATE_ATTACK_ENEMY = 2.0  # 后期攻击敌人

    # 技能奖励
    SKILL_USE= 8.0  # 有效技能使用奖励
    SKILL_NO_TARGET = -5  # 无目标使用技能惩罚
    RECALL_WITH_LOW_HP = 1.0  # 低血量回城奖励
    RECALL_WITH_High_HP = -0.5#高血量回城惩罚
    Heal_SKill_Good=5#有效回复奖励
    Heal_SKill_Bad=-3#无效恢复惩罚

    # 经济增加奖励
    ECONOMIC_GROWTH_FACTOR = 0.01 # 经济增加奖励
    Kill_FACTOR = 10.0 # 击杀奖励


    # 游戏结果奖励
    DEATH_PENALTY = -50.0  # 死亡惩罚
    GAME_VICTORY = 1000.0  # 游戏胜利奖励
    GAME_DEFEAT = -1000.0  # 游戏失败惩罚

    #游戏因子
    DEVELOPMENT_FACTOR = 1.0 #发育因子，通过时间阶段以及敌我双方的位置综合判断
    ATTACK_FACTOR= 1  #进攻因子，通过时间阶段，敌我双方的位置，经济情况综合判断
    DEFENSE_FACTOR= 1  #防守因子，通过时间阶段，敌我双方的位置，经济情况综合判断
    MOVE_FACTOR = 1  #移动因子
    MOVERESTR_FACTOR= 1.0  #移动受限因子

    def get_DEVELOPMENT_FACTOR(self,steps: int ) -> float:#获取发育因子
        # 计算衰减因子
        self.DEVELOPMENT_FACTOR = max(0.1, 1 - steps / TrainingConfig.MAX_STEPS)
        
        return self.DEVELOPMENT_FACTOR

    def get_ATTACK_FACTOR(self,steps: int ) -> float:#获取进攻因子
        # 计算衰减因子
        self.ATTACK_FACTOR = max(0.1, 1 - steps / TrainingConfig.MAX_STEPS)
        
        return self.ATTACK_FACTOR

    def get_DEFENSE_FACTOR(self,steps: int ) -> float:#获取防守因子
        # 计算衰减因子
        self.DEFENSE_FACTOR = max(0.1, 1 - steps / TrainingConfig.MAX_STEPS)
        
        return self.DEFENSE_FACTOR

    def get_MOVE_FACTOR(self,episodes: int ) -> float:#获取移动因子
        # 计算衰减因子
        self.MOVE_FACTOR = max(0.1, 1 - episodes / TrainingConfig.NUM_EPISODES)
        
        return self.MOVE_FACTOR

    def get_MOVERESTR_FACTOR(self,episodes: int ) -> float:#获取移动受限因子
        # 计算衰减因子
        self.MOVERESTR_FACTOR = max(0.1, 1 - episodes / TrainingConfig.NUM_EPISODES)
        
        return self.MOVERESTR_FACTOR

reward_config=Reward_Config()



class MatchAlgorithm(Enum):
    """模板匹配算法"""
    CCORR_NORMED = cv2.TM_CCORR_NORMED#相关性匹配
    CCOEFF_NORMED = cv2.TM_CCOEFF_NORMED#相关系数匹配
    SQDIFF_NORMED = cv2.TM_SQDIFF_NORMED#平方差匹配


@dataclass(frozen=True)
class TemplateRegion:
    """模板匹配区域配置"""
    MONSTER_AREA: Final[Tuple[int, int, int, int]] = (992, 7, 1050, 51)#野怪头像
    BLOOD_AREA: Final[Tuple[int, int, int, int]] = (744,264,867,272) #自己血量
    ENEMY_BLOOD_AREA: Final[Tuple[int, int, int, int]] = (818,12, 1011, 58)#敌方血量
    CHARACTER_AREA: Final[Tuple[int, int, int, int]] = (61, 0, 317, 256)#小地图 匹配英雄位置
    GAME_START_AREA: Final[Tuple[int, int, int, int]] = (1443, 0, 1483, 39)#
    VICTORY_AREA: Final[Tuple[int, int, int, int]] = (724, 418, 867, 436)#
    DEFEAT_AREA: Final[Tuple[int, int, int, int]] = (722, 384, 869, 410)#
    BACK_HOME_AREA: Final[Tuple[int, int, int, int]] = (453, 629, 625, 675)#返回大厅
    MODE_AREA: Final[Tuple[int, int, int, int]] = (151, 280, 304, 351)#5v5
    START_AREA: Final[Tuple[int, int, int, int]] = (1002, 590, 1145, 643)#开始按钮
    ICON_AREA: Final[Tuple[int, int, int, int]] = (248, 286, 282, 381)#icon 无关
    FAYULU_AREA: Final[Tuple[int, int, int, int]] = (656, 3, 776, 62)#无关
    HOUYI_ICO_AREA: Final[Tuple[int, int, int, int]] = (970, 72, 1069, 144)#无关
    CONFIRM_AREA: Final[Tuple[int, int, int, int]] = (1121, 640, 1231, 694)#无关
    SELF_DEATH_AREA1: Final[Tuple[int, int, int, int]] = (731, 168, 869, 201)  #查看自己是否死亡1    3
    SELF_DEATH_AREA2: Final[Tuple[int, int, int, int]] = (751, 0, 847, 14)  #查看自己是否死亡2       3


@dataclass(frozen=True)
class Confidence:
    """置信度配置"""
    HIGH: Final[float] = 0.98
    MEDIUM: Final[float] = 0.9
    LOW: Final[float] = 0.8


# 野怪模板名称列表
MONSTER_TEMPLATES: Final[list[str]] = [
    'chuanshanjia', 'ciwei', 'dalong', 'hongbuff', 'ji',
    'jingling', 'lanbuff', 'lang1', 'lang2', 'longwang',
    'xialong', 'xiaoniao', 'xiyi', 'zhu1', 'zhu2',
    'hongta', 'hongshuijing'
]


other_status:dict={
    "if_alive":True,
    "if_game_start":False,
    "if_first_episode_have_start":False,
    "move_target":(249,219),#移动目标，初始值为我方发育路一塔
    "move_reason":"无",   #大语言模型给出的当前局势分析
    "time":0,
    "my_position":None
}

other_click_position={
    "update_first_skill":(1131,556),
    "update_second_skill":(1211,423),
    "update_third_skill":(1345,345),
    "buy_equipment":(1389,106)
}
