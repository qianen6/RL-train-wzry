from typing import Tuple

import numpy as np
import time
from new_wzry_ai.config.default_config import TemplateRegion
from new_wzry_ai.environment.routing import RoutingGuide
from new_wzry_ai.utils.blood_detect import BloodDetector
from new_wzry_ai.utils.economic_growth import EconomicGrowth
from new_wzry_ai.utils.heros_position_detector import HeroPositionDetector
from new_wzry_ai.utils.skill_cooldown import CooldownDetector
from new_wzry_ai.utils.template_match import TemplateMatcher
from new_wzry_ai.utils.tower_attack_detector import TowerAttackDetector
from new_wzry_ai.utils.reward_state import reward_state

class StateVector:
    def __init__(self):
        self.blood_detector = BloodDetector()
        self.routing_guide = RoutingGuide()
        self.cooldown_detector = CooldownDetector()
        self.economic_growth = EconomicGrowth()
        self.position_detector = HeroPositionDetector()
        self.state_vector = None
        self.template_matcher = TemplateMatcher()
        self.tower_attack_detector = TowerAttackDetector()
        self.routing_guide = RoutingGuide()

    def initial_dict(self, frame):
        current_pos = self.routing_guide.get_houyi_position(frame)
        return {
            "isDead": self.is_dead(frame),
            "isEnemyBloodChanged": self.is_enemy_blood_changed(frame),
            "isVictory": self.is_victory(frame),
            "isDefeat": self.is_defeat(frame),
            "isTowerAttacking": self.is_tower_attacking(frame),
            "isAtDest": self.routing_guide.is_at_any_destination(frame, current_pos),
            "isAttackingMonster": self.routing_guide.detect_any_monster(frame),
            "selfBlood": self.get_self_blood(frame),
            "enemyBlood": self.get_enemy_blood(frame),
            "economicValue": self.get_economic_value(frame),
            "selfKillsValue": self.get_self_kills_value(frame),
            "positions": self.get_positions(frame),
            "skillCooldown": self.get_skill_cooldown(frame),
            "action": (8,7),
            "reward": 0
        }

    def update(self, frame, action : Tuple[int, int], reward : float):
        if self.state_vector is None:
            self.state_vector = self.initial_dict(frame)
            return
        other_start_time = time.time()
        # 更新状态向量


        current_pos = self.routing_guide.get_houyi_position(frame)

        self.state_vector["isDead"] = reward_state.isDead
        self.state_vector["isEnemyBloodChanged"] = reward_state.isEnemyBloodChanged
        self.state_vector["isVictory"] = reward_state.isVictory
        self.state_vector["isDefeat"] = reward_state.isDefeat
        self.state_vector["isTowerAttacking"] = reward_state.isTowerAttacking
        self.state_vector["isAtDest"] = reward_state.isAtDest
        self.state_vector["isAttackingMonster"] = reward_state.isAttackingMonster
        self.state_vector["selfBlood"] = reward_state.selfBlood
        self.state_vector["enemyBlood"] = reward_state.enemyBlood
        self.state_vector["economicValue"] = reward_state.economicValue
        self.state_vector["selfKillsValue"] = reward_state.selfKillsValue
        self.state_vector["positions"] = self.get_positions(frame)
        self.state_vector["skillCooldown"] = self.get_skill_cooldown(frame)
        self.state_vector["action"] = (action[0], action[1])
        self.state_vector["reward"] = reward
        other_end_time = time.time()
        print(f"更新状态向量时间: {other_end_time - other_start_time:.6f} 秒")

    @staticmethod
    def normalize(l, value: float):
        if not l:  # 处理空列表/字典
            return []
        if isinstance(l, dict):
            return [v / value for v in l.values()]
        else:  # 如果不是字典，就假设它是列表
            for i in range(len(l)):
                l[i] = l[i] / value
            return l


    @staticmethod
    def bool_to_int(value: bool):
        return 1 if value else 0

    def get_state(self):
        # 创建初始的状态向量
        vector = [
            self.bool_to_int(self.state_vector["isDead"]),
            self.bool_to_int(self.state_vector["isEnemyBloodChanged"]),
            self.bool_to_int(self.state_vector["isVictory"]),
            self.bool_to_int(self.state_vector["isDefeat"]),
            self.bool_to_int(self.state_vector["isTowerAttacking"]),
            self.bool_to_int(self.state_vector["isAtDest"]),
            self.bool_to_int(self.state_vector["isAttackingMonster"]),
            self.state_vector["selfBlood"],
            self.state_vector["enemyBlood"],
            self.state_vector["economicValue"] / 200,
            self.state_vector["selfKillsValue"] / 20,
        ]

        # 添加归一化后的位置信息
        vector = vector + self.normalize(self.state_vector["positions"], 256)

        # 添加归一化后的技能冷却时间
        vector = vector + self.normalize(self.state_vector["skillCooldown"], 60)

        vector.append(self.state_vector["action"][0] / 8)

        vector.append(self.state_vector["action"][1] / 7)

        vector.append(self.state_vector["reward"] / 2000)
        # 填充至固定大小
        padding_size = 200 - len(vector)
        vector += [0] * padding_size

        # 转换为 numpy 数组并返回
        return np.array(vector, dtype=np.float32)

    def is_dead(self, frame):
        return self.template_matcher.match_template(frame, 'self_death1').success \
            or self.template_matcher.match_template(frame, 'self_death2').success

    def get_positions(self, frame):
        x1, y1, x2, y2 = TemplateRegion.CHARACTER_AREA

        frame = frame[y1:y2, x1:x2]

        # 获取英雄的位置
        my_position, teammate_position, enemy_position = self.position_detector.get_hero_position(frame)

        my_position = my_position if my_position is not None else []

        teammate_position = teammate_position if teammate_position is not None else []

        enemy_position = enemy_position if enemy_position is not None else []

        # 展开所有二元组并转换为 float 类型数组
        positions = [coord for pos in [my_position] + teammate_position + enemy_position for coord in pos]

        return positions

    def get_self_blood(self, frame):
        return self.blood_detector.get_self_blood(frame)

    def is_enemy_blood_changed(self, frame):
        return self.blood_detector.is_enemy_blood_changed(frame)

    def is_victory(self, frame):
        return self.template_matcher.match_template(frame, 'victory').success

    def is_defeat(self, frame):
        return self.template_matcher.match_template(frame, 'defeat').success

    def is_tower_attacking(self, frame):
        return self.tower_attack_detector.detect(frame)

    def get_enemy_blood(self, frame):
        return self.blood_detector.get_enemy_blood(frame)

    def get_economic_value(self, frame):
        return self.economic_growth.get_economic_value(frame)

    def get_self_kills_value(self, frame):
        return self.economic_growth.get_kills_value(frame)

    def get_skill_cooldown(self, frame):
        return self.cooldown_detector.get_skill_cooldown(frame)
