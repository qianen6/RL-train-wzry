"""
环境相关异常定义
"""

class EnvironmentError(Exception):
    """环境相关异常基类"""
    pass

class GameNotReadyError(EnvironmentError):
    """游戏未准备好异常"""
    pass

class GameStateError(EnvironmentError):
    """游戏状态异常"""
    pass 