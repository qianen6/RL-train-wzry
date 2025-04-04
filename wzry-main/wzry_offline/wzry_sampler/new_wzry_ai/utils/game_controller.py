from typing import Dict, List, Optional, Set, Union
from pynput.keyboard import Key, Controller, KeyCode
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from concurrent.futures import ThreadPoolExecutor, Future, wait
from new_wzry_ai.config.default_config import other_status
# 类型别名
KeyType = Union[str, Key, KeyCode]

class KeyboardError(Exception):
    """键盘操作相关异常基类"""
    pass

class InvalidActionError(KeyboardError):
    """无效动作异常"""
    pass

class KeyboardOperationError(KeyboardError):
    """键盘操作执行异常"""
    pass

class KeyOperation(Enum):
    """键盘操作枚举"""
    PRESS = auto()
    RELEASE = auto()

@dataclass
class HandAction:
    keys: Optional[List[str]]
    is_click: bool = False

def default_letter_keys() -> Dict[str, str]:
    return {
        'A': 'a', 'B': 'b', 'C': 'c', 'D': 'd', 'E': 'e',
        'F': 'f', 'G': 'g', 'H': 'h', 'I': 'i', 'J': 'j',
        'K': 'k', 'L': 'l', 'M': 'm', 'N': 'n', 'O': 'o',
        'P': 'p', 'Q': 'q', 'R': 'r', 'S': 's', 'T': 't',
        'U': 'u', 'V': 'v', 'W': 'w', 'X': 'x', 'Y': 'y',
        'Z': 'z'
    }

def default_navigation_keys() -> Dict[str, Key]:
    return {
        'up': Key.up, 'down': Key.down,
        'left': Key.left, 'right': Key.right
    }

def default_control_keys() -> Dict[str, Key]:
    return {
        'space': Key.space, 'enter': Key.enter,
        'esc': Key.esc, 'tab': Key.tab
    }

def default_modifier_keys() -> Dict[str, Key]:
    return {
        'shift': Key.shift,
        'ctrl': Key.ctrl,
        'alt': Key.alt
    }

def default_left_actions() -> Dict[int, HandAction]:
    return {
        0: HandAction(['W']),
        1: HandAction(['A']),
        2: HandAction(['S']),
        3: HandAction(['D']),
        4: HandAction(['W', 'A']),
        5: HandAction(['W', 'D']),
        6: HandAction(['S', 'A']),
        7: HandAction(['S', 'D']),
        8: HandAction(None)
    }

def default_right_actions() -> Dict[int, HandAction]:
    return {
        0: HandAction(['J'], True),
        1: HandAction(['T'], True),
        2: HandAction(['E'], True),
        3: HandAction(['R'], True),
        4: HandAction(['B'], True),
        5: HandAction(['C'], True),
        6: HandAction(['F'], True),
        7: HandAction(None)
    }

@dataclass(frozen=True)
class KeyboardConstants:
    """键盘常量和映射定义"""
    # 基础按键映射
    LETTER_KEYS: Dict[str, str] = field(default_factory=default_letter_keys)
    NAVIGATION_KEYS: Dict[str, Key] = field(default_factory=default_navigation_keys)
    CONTROL_KEYS: Dict[str, Key] = field(default_factory=default_control_keys)
    MODIFIER_KEYS: Dict[str, Key] = field(default_factory=default_modifier_keys)
    LEFT_ACTIONS: Dict[int, HandAction] = field(default_factory=default_left_actions)
    RIGHT_ACTIONS: Dict[int, HandAction] = field(default_factory=default_right_actions)

class ActionExecutor:
    def __init__(self):
        """初始化动作执行器"""
        self.keyboard = Controller()

        self.keyboard_lock = threading.Lock()
        self.constants = KeyboardConstants()
        self._all_mappings = {
            **self.constants.LETTER_KEYS,
            **self.constants.NAVIGATION_KEYS,
            **self.constants.CONTROL_KEYS,
            **self.constants.MODIFIER_KEYS
        }
        self.current_left_keys: Set[str] = set()
        self.current_right_keys: Set[str] = set()
        self.action_pool = ThreadPoolExecutor(max_workers=2)
        self.click_pool1 = ThreadPoolExecutor(max_workers=2)
        self.click_future1 = self.click_pool1.submit(self._click_worker)
        self.click_pool2 = ThreadPoolExecutor(max_workers=2)
        self.click_future2 = self.click_pool2.submit(self._new_game_click)

        # New attributes for continuous clicking
        self.continuous_click_active = False
        self.continuous_click_lock = threading.Lock()
        self.continuous_click_thread = None

    def _click_worker(self):
        try:

            while True:
                if other_status['if_game_start']:
                    for key in ['O', 'U', 'I', 'P']:
                        key=[key]
                        self._handle_click_action(key)
                        #print(f"Pressed {key[0]}")
                        time.sleep(0.5)
                # 等待10秒
                time.sleep(10)
        except Exception as e:
            print("====================== error =======================")
            print(e)
            print("====================== error =======================")

    def _new_game_click(self):
        try:

            while True:
                if not other_status['if_game_start'] and other_status['if_first_episode_have_start']:
                    time.sleep(20)
                    for key in ['B', 'B', 'B', 'B','I','Z']:
                        key=[key]
                        self._handle_click_action(key)
                        print(f"Pressed {key[0]}")
                        time.sleep(10)
                    time.sleep(10)
                # 等待10秒
                time.sleep(10)
        except Exception as e:
            print("====================== error =======================")
            print(e)
            print("====================== error =======================")

    def _perform_key_operation(self, operation: KeyOperation, key: str) -> None:
        try:
            mapped_key = self._all_mappings.get(key, key)
            operation_func = {
                KeyOperation.PRESS: self.keyboard.press,
                KeyOperation.RELEASE: self.keyboard.release
            }[operation]
            operation_func(mapped_key)
        except Exception as e:
            raise KeyboardOperationError(f"按键操作失败: {key}") from e

    def _safe_key_operation(self, operation: KeyOperation, keys: List[str]) -> None:
        """线程安全地执行键盘操作"""
        with self.keyboard_lock:
            for key in keys:
                self._perform_key_operation(operation, key)

    def _handle_continuous_action(self, new_keys: Set[str], current_keys: Set[str]) -> None:
        """处理持续性按键动作"""
        keys_to_release = current_keys - new_keys
        keys_to_press = new_keys - current_keys

        if keys_to_release:
            self._safe_key_operation(KeyOperation.RELEASE, list(keys_to_release))
        if keys_to_press:
            self._safe_key_operation(KeyOperation.PRESS, list(keys_to_press))

    def     _handle_click_action(self, keys: List[str]) -> None:
        """处理点击式按键动作"""
        self._safe_key_operation(KeyOperation.PRESS, keys)
        time.sleep(0.1)
        self._safe_key_operation(KeyOperation.RELEASE, keys)

    def _continuous_click_worker(self):
        """Worker thread for continuous clicking"""
        while True:
            with self.continuous_click_lock:
                if not self.continuous_click_active or not other_status["if_game_start"]:
                    break
                self._handle_click_action(['Z'])  # Continuous click on 'Z'
            time.sleep(0.5)  # Click every 0.5 seconds

    def _execute_hand_action(self, action: int, is_left: bool) -> None:
        """Execute a hand action"""
        actions_map = self.constants.LEFT_ACTIONS if is_left else self.constants.RIGHT_ACTIONS

        if action not in actions_map:
            raise InvalidActionError(f"Invalid action ID: {action}")

        hand_action = actions_map[action]

        if is_left:
            if hand_action.keys:
                new_keys = set(hand_action.keys)
                self._handle_continuous_action(new_keys, self.current_left_keys)
                self.current_left_keys = new_keys
            else:
                # If left action is None, release all current left keys
                self._handle_continuous_action(set(), self.current_left_keys)
                self.current_left_keys = set()
        else:
            if action == 0:
                # Start continuous clicking
                with self.continuous_click_lock:
                    self.continuous_click_active = True
                if self.continuous_click_thread is None or not self.continuous_click_thread.is_alive():
                    self.continuous_click_thread = threading.Thread(target=self._continuous_click_worker)
                    self.continuous_click_thread.start()
            else:
                # Stop continuous clicking for any non-0 action, including 7
                with self.continuous_click_lock:
                    self.continuous_click_active = False
                if hand_action.keys and hand_action.is_click:
                    self._handle_click_action(hand_action.keys)
                # If action is 7, hand_action.keys is None, so no click action is performed


    def execute_action_async(self, left_action: int, right_action: int) -> None:
        """异步执行p动作"""
        left_future = self.action_pool.submit(self._execute_hand_action, left_action, True)
        right_future = self.action_pool.submit(self._execute_hand_action, right_action, False)
        # 等待两个动作都执行完成
        done, _ = wait([left_future, right_future])
        # 检查是否有异常
        for future in done:
            if future.exception():
                raise future.exception()

    def close(self):
        """Close the action executor"""
        self.click_future1.cancel()
        self.click_future2.cancel()
        with self.continuous_click_lock:
            self.continuous_click_active = False
        if self.continuous_click_thread is not None:
            self.continuous_click_thread.join()
        self.action_pool.shutdown(wait=True)
        self.click_pool1.shutdown(wait=False)
        self.click_pool2.shutdown(wait=False)