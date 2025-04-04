# reward_state.py
class RewardStateManager:
    _instance = None  # 用于存储唯一实例

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RewardStateManager, cls).__new__(cls)
            cls._instance.init_state()
        return cls._instance

    def init_state(self):
        """初始化状态变量"""
        self.isDead = False
        self.isEnemyBloodChanged = False
        self.isVictory = False
        self.isDefeat = False
        self.isTowerAttacking = False
        self.isAtDest = (False, None)
        self.isAttackingMonster = False
        self.selfBlood = 0.0
        self.enemyBlood = 0.0
        self.economicValue = 0
        self.selfKillsValue = 0


    def reset(self):
        """重置状态"""
        self.init_state()


# 创建一个全局实例
reward_state = RewardStateManager()
