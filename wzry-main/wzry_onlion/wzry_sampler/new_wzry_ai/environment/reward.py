"""
奖励计算模块
"""
import time
import math
from typing import Tuple, Optional
from dataclasses import dataclass

# 使用绝对导入
from new_wzry_ai.utils.blood_detect import BloodDetector
from new_wzry_ai.utils.template_match import TemplateMatcher
from new_wzry_ai.utils.skill_cooldown import CooldownDetector
from new_wzry_ai.utils.economic_growth import EconomicGrowth
from new_wzry_ai.environment.routing import RoutingGuide
from new_wzry_ai.config.default_config import reward_config
from new_wzry_ai.config.default_config import other_status

@dataclass
class RewardState:
    total_episodes: int = 0
    victories: int = 0
    defeats: int = 0
    start_time: float = 0

class RewardCalculator:
    def __init__(self):
        """初始化奖励计算器"""
        self.blood_detector = BloodDetector()
        self.routing_guide = RoutingGuide()
        self.cooldown_detector = CooldownDetector()
        self.template_matcher = TemplateMatcher()
        self.state = RewardState()
        self.state.start_time = time.time()
        self.economic_growth = EconomicGrowth()  # 使用默认区域
        """初始化经济奖励"""
        self.prev_economic_value = 0  # 记录上次的经济
        self.prev_kills_value = 0  # 记录上次的击杀数
        self.first_run_region2 = True  # 用于区分第一次获取的击杀数

    def _is_late_game(self) -> bool:
        """检查是否到达游戏5分钟后阶段"""
        return (time.time() - self.state.start_time) > 300  # 5分钟 = 300秒

    def _is_late_game2(self) -> bool:
        """检查是否到达游戏10分钟后阶段"""
        return (time.time() - self.state.start_time) > 600  # 10分钟 = 600秒

    def _get_movement_target(self) -> Tuple[str, Tuple[int, int]]:
        """获取当前移动目标"""
        return other_status["move_target"]



    def calculate_movement_reward(self, image, self_blood: float,left_action,episodes: int ) -> float:
        """计算移动奖励"""
        current_pos = self.routing_guide.get_houyi_position(image)#获取后羿坐标
 
        if current_pos is not None:
            other_status["my_position"]=current_pos

        if current_pos is None or self.routing_guide.last_position is None:
            self.routing_guide.last_position = current_pos
            return 0.0

        # 获取目标位置
        if self_blood <= 0.2 and not self._is_late_game2:#前期血量较低时回泉水
            target_pos = (85, 226)  # 泉水位置
        else:
            target_pos = self._get_movement_target()  # 使用同一个目标获取函数
        
        # 计算距离变化
        current_distance = self._calculate_distance(current_pos, target_pos)#目标位置到上一次当前位置的距离
        last_distance = self._calculate_distance(self.routing_guide.last_position, target_pos)#目标位置到当前位置的距离
        distance_changed = current_distance - last_distance#距离变化
        
        if distance_changed==0  and left_action in [1,2,3,4,5,6,7]:
            print("kazhu")
            return reward_config.MOVE_To_Edge*reward_config.get_MOVERESTR_FACTOR(episodes)
        # 更新上一次位置
        self.routing_guide.last_position = current_pos

        # 血量低于0.2时
        if self_blood <= 0.2 and not self._is_late_game2:
            if distance_changed < 0:  # 向泉水移动
                return -1*distance_changed*reward_config.MOVE_TO_TARGET*reward_config.get_MOVE_FACTOR(episodes)
            elif distance_changed > 0:  # 远离泉水
                return -1*distance_changed*reward_config.MOVE_PENALTY*reward_config.get_MOVE_FACTOR(episodes)
        # 血量高于0.2时
        else:
            if distance_changed < 0:  # 向目标移动
                return -1*distance_changed*reward_config.MOVE_TO_TARGET*reward_config.get_MOVE_FACTOR(episodes)
            elif distance_changed > 0:  # 远离目标
                return -1*distance_changed*reward_config.MOVE_PENALTY*reward_config.get_MOVE_FACTOR(episodes)
        return 0.0

    def calculate_attack_reward(self, image, is_blood_changed: bool, current_pos: Optional[Tuple[int, int]]) -> float:
        """计算攻击奖励"""
        if not is_blood_changed:
            return 0.0#敌人血量没有变化返回0
        
        # 检测是否有野怪
        has_monster = self.routing_guide.detect_any_monster(image)
        is_at_dest, _ = self.routing_guide.is_at_any_destination(image, current_pos)
        
        # 检测是否在攻击野怪
        if has_monster:
            if is_at_dest:
                # 在目标范围内攻击野怪
                return (reward_config.LATE_ATTACK_MONSTER if self._is_late_game()
                       else reward_config.ATTACK_MONSTER)
            else:
                # 脱战状态
                return (reward_config.LATE_ATTACK_MONSTER_OUT if self._is_late_game()
                       else reward_config.ATTACK_MONSTER_OUT)
        # 检测是否在攻击敌人
        elif self.blood_detector.get_enemy_blood(image) > 0:
            return (reward_config.LATE_ATTACK_ENEMY if self._is_late_game()
                   else reward_config.ATTACK_ENEMY)
        
        return 0.0

    def calculate_skill_reward(self, image, right_action: int, left_action: int, self_blood: float) -> float:
        """计算技能使用奖励"""
        reward = 0.0
        
        # 检查是否在攻击状态
        is_attacking = (self.blood_detector.is_enemy_blood_changed(image) and 
                       (self.blood_detector.get_enemy_blood(image) > 0 or 
                        self.routing_guide.detect_any_monster(image)))
        
        # 获取技能状态
        skills_status = self.cooldown_detector.check_skills_status(image)
        
        # 处理技能使用 (只处理1,2,3,5,6技能)
        if right_action in [1, 2, 3, 6]:
            if is_attacking and skills_status.get(right_action, True):  # 在攻击且技能可用
                reward += reward_config.SKILL_USE
            elif not is_attacking:  # 不在攻击状态使用技能
                reward += reward_config.SKILL_NO_TARGET

        if right_action in [5]:
            if self_blood<0.9:
                reward += reward_config.Heal_SKill_Good
            else:
                reward += reward_config.Heal_SKill_Bad
        
        # 处理回城
        if right_action == 4:
            if not self._is_late_game2() and self_blood < 0.1 and left_action == 8:
                reward += reward_config.RECALL_WITH_LOW_HP
            else:
                reward+=reward_config.RECALL_WITH_High_HP
        
        return reward

    def calcuate_economic_reward(self, image) -> float:
        """
        计算经济增长的奖励
        :param image: 游戏界面的图像，用于提取经济数字
        :return: 经济增长奖励
        """

        # 获取当前经济值
        current_economic_value_region1 = self.economic_growth.get_economic_value(image)
        current_economic_value_region2 = self.economic_growth.get_kills_value(image)

        # 初始化奖励
        reward1 = 0
        reward2 = 0

        # 如果region1的数值非0且有增加，则奖励1
        if current_economic_value_region1 != 0:
            delta_region1 = current_economic_value_region1 - self.prev_economic_value
            if delta_region1 > 50:  # 增量超过 50 时才进行奖励计算
                reward1 = delta_region1 * reward_config.ECONOMIC_GROWTH_FACTOR

        # 如果region2的数值非0且有增加，则奖励2
        if current_economic_value_region2 != 0 or not self.first_run_region2:
            if self.first_run_region2:
                self.first_run_region2 = False  # 第一次执行时设置为 False
            delta_region2 = current_economic_value_region2 - self.prev_kills_value
            if delta_region2 > 0:  # region2 每增加 1 就给奖励
                reward2 = delta_region2 * reward_config.Kill_FACTOR

        # 更新prev_economic_value和prev_kills_value
        self.prev_economic_value = current_economic_value_region1
        self.prev_kills_value = current_economic_value_region2

        # 返回奖励之和
        return float(reward1 + reward2)





    def check_game_result(self, image) -> Tuple[bool, float]:
        """检查游戏结果"""
        if self.template_matcher.match_template(image, 'victory').success:
            self.state.victories += 1
            return True, reward_config.GAME_VICTORY
            
        if self.template_matcher.match_template(image, 'defeat').success:
            self.state.defeats += 1
            return True, reward_config.GAME_DEFEAT
            
        return False, 0.0



    def check_death(self, image) -> float:
        """检查是否死亡"""
        if self.template_matcher.match_template(image, 'self_death1').success or self.template_matcher.match_template(image, 'self_death2').success:
            other_status['if_alive']=False
            return reward_config.DEATH_PENALTY
        return 0

    def calculate_total_reward(self, image, right_action: int, left_action: int, episodes: int,min_map) -> Tuple[float, bool]:
        """计算总奖励值"""
        self_blood = self.blood_detector.get_self_blood(image)
        is_blood_changed = self.blood_detector.is_enemy_blood_changed(image)
        current_pos = self.routing_guide.get_houyi_position(image)
        
        # 计算各项奖励
        move_reward = self.calculate_movement_reward(image, self_blood,left_action , episodes)
        attack_reward = self.calculate_attack_reward(image, is_blood_changed, current_pos)
        skill_reward = self.calculate_skill_reward(image, right_action, left_action, self_blood)
        economic_reward = self.calcuate_economic_reward(image)
        # 检查死亡
        death_reward = self.check_death(image)
        
        # 检查游戏结果
        game_over, result_reward = self.check_game_result(image)
        
        total_reward = move_reward + attack_reward + skill_reward + death_reward + result_reward + economic_reward
        return total_reward, game_over

    def _calculate_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """计算两点间欧几里得距离"""
        return math.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2) 