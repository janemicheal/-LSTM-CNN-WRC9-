"""
输入模拟模块 (InputSimulator)

核心职责：将模型预测的操作指令转为游戏输入
支持两种输入方式：vgamepad（虚拟手柄）和 pynput（键盘）
"""
import time
import logging
from typing import Dict, Optional, Tuple
import threading

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='../logs/input_simulator.log'
)
logger = logging.getLogger('InputSimulator')

# 尝试导入vgamepad和pynput
try:
    import vgamepad as vg
    VGAMEPAD_AVAILABLE = True
except ImportError:
    logger.warning("vgamepad库未安装，将使用键盘模拟")
    VGAMEPAD_AVAILABLE = False

try:
    from pynput.keyboard import Key, Controller as KeyboardController
    PYNPUT_AVAILABLE = True
except ImportError:
    logger.warning("pynput库未安装")
    PYNPUT_AVAILABLE = False

class InputSimulator:
    """
    输入模拟器类，负责将控制指令转换为游戏输入
    """
    
    @staticmethod
    def is_gamepad_available():
        """
        静态方法：检查游戏手柄是否可用
        
        Returns:
            bool: 游戏手柄是否可用
        """
        global VGAMEPAD_AVAILABLE
        return VGAMEPAD_AVAILABLE
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化输入模拟器
        
        Args:
            config: 配置字典
        """
        # 默认配置
        self.default_config = {
            'input_method': 'keyboard',  # 强制使用键盘输入，确保输入可靠发送
            'enable_smoothing': True,  # 是否启用平滑
            'smoothing_factor': 0.4,  # 提高平滑因子，减少延迟感，使控制更灵敏
            'polling_rate': 100,  # 提高输入更新频率，使控制更流畅
            'throttle_axis': 4,  # 油门轴索引 (Xbox 360)
            'brake_axis': 5,  # 刹车轴索引 (Xbox 360)
            'steering_axis': 0,  # 转向轴索引 (Xbox 360)
            'keyboard_layout': {
                'steering_left': Key.left,  # 左转向按键
                'steering_right': Key.right,  # 右转向按键
                'throttle': 'w',  # 油门按键
                'brake': 's',  # 刹车按键
                'handbrake': 'space',  # 手刹按键
                'gear_up': 'e',  # 升档按键
                'gear_down': 'q',  # 降档按键
                'pause': Key.esc  # 暂停按键
            },
            'continuous_input': True,  # 是否持续输入
            'debug': True  # 启用调试模式，方便诊断问题
        }
        
        # 合并配置
        self.config = self.default_config.copy()
        if config:
            self.config.update(config)
        
        # 初始化输入设备
        self.input_method = self.config['input_method'].lower()
        self.gamepad = None
        self.keyboard = None
        self.initialized = False
        
        # 状态变量
        self.current_steering = 0.0  # 当前转向值 (-1.0 to 1.0)
        self.current_throttle = 0.0  # 当前油门值 (0.0 to 1.0)
        self.current_brake = 0.0  # 当前刹车值 (0.0 to 1.0)
        self.current_handbrake = False  # 当前手刹状态
        
        # 按键状态跟踪，用于保持按键连续性
        self.currently_pressed_keys = {
            'steering_left': False,
            'steering_right': False,
            'throttle': False,
            'brake': False
        }
        
        # 输入更新线程
        self.running = False
        self.input_thread = None
        self.control_queue = []
        self.control_queue_lock = threading.Lock()
        
        # 初始化输入设备
        self.initialize()
    
    def initialize(self) -> bool:
        """
        初始化输入设备
        
        Returns:
            是否成功初始化
        """
        try:
            if self.input_method == 'gamepad' and VGAMEPAD_AVAILABLE:
                # 初始化虚拟游戏手柄
                self.gamepad = vg.VX360Gamepad()
                logger.info("已初始化虚拟Xbox 360手柄")
            else:
                # 回退到键盘模式
                self.input_method = 'keyboard'
                if PYNPUT_AVAILABLE:
                    self.keyboard = KeyboardController()
                    logger.info("已初始化键盘控制器")
                else:
                    logger.error("无法初始化任何输入设备")
                    return False
            
            self.initialized = True
            return True
            
        except Exception as e:
            logger.error(f"初始化输入设备时出错: {e}")
            self.initialized = False
            return False
    
    def _apply_smoothing(self, target_steering: float, target_throttle: float, target_brake: float) -> Tuple[float, float, float]:
        """
        应用平滑算法到控制指令
        
        Args:
            target_steering: 目标转向值
            target_throttle: 目标油门值
            target_brake: 目标刹车值
            
        Returns:
            平滑后的控制值
        """
        if not self.config['enable_smoothing']:
            return target_steering, target_throttle, target_brake
        
        # 获取平滑因子
        factor = self.config['smoothing_factor']
        
        # 应用平滑
        smoothed_steering = self.current_steering + factor * (target_steering - self.current_steering)
        smoothed_throttle = self.current_throttle + factor * (target_throttle - self.current_throttle)
        smoothed_brake = self.current_brake + factor * (target_brake - self.current_brake)
        
        # 限制范围
        smoothed_steering = max(-1.0, min(1.0, smoothed_steering))
        smoothed_throttle = max(0.0, min(1.0, smoothed_throttle))
        smoothed_brake = max(0.0, min(1.0, smoothed_brake))
        
        # 特殊处理：如果方向改变，直接应用一部分目标值
        if (self.current_steering > 0 and target_steering < 0) or (self.current_steering < 0 and target_steering > 0):
            # 方向改变，增加平滑因子
            smoothed_steering = self.current_steering + min(factor * 2, 1.0) * (target_steering - self.current_steering)
        
        return smoothed_steering, smoothed_throttle, smoothed_brake
    
    def _send_gamepad_input(self, steering: float, throttle: float, brake: float):
        """
        发送游戏手柄输入
        
        Args:
            steering: 转向值 (-1.0 to 1.0)
            throttle: 油门值 (0.0 to 1.0)
            brake: 刹车值 (0.0 to 1.0)
        """
        if not self.gamepad or not VGAMEPAD_AVAILABLE:
            logger.error("游戏手柄未初始化")
            return
        
        try:
            # 转换为游戏手柄轴值范围 (-32768 to 32767)
            steering_value = int(steering * 32767)
            throttle_value = int((throttle - 0.5) * 65534)  # 将0-1映射到-32767到32767
            brake_value = int((brake - 0.5) * 65534)
            
            # 设置模拟轴
            self.gamepad.left_joystick_float(steering_value / 32767.0, 0.0)
            self.gamepad.right_trigger_float(throttle)  # 使用右扳机作为油门
            self.gamepad.left_trigger_float(brake)  # 使用左扳机作为刹车
            
            # 更新游戏手柄状态
            self.gamepad.update()
            
            if self.config['debug']:
                logger.debug(f"发送手柄输入: 转向={steering_value}, 油门={throttle_value}, 刹车={brake_value}")
                
        except Exception as e:
            logger.error(f"发送游戏手柄输入时出错: {e}")
    
    def _send_keyboard_input(self, steering: float, throttle: float, brake: float):
        """
        发送键盘输入，优化版：只在按键状态变化时才改变按键状态
        
        Args:
            steering: 转向值 (-1.0 to 1.0)
            throttle: 油门值 (0.0 to 1.0)
            brake: 刹车值 (0.0 to 1.0)
        """
        if not self.keyboard or not PYNPUT_AVAILABLE:
            logger.error("键盘控制器未初始化")
            return
        
        try:
            # 打印调试信息
            print(f"发送键盘输入: 转向={steering:.2f}, 油门={throttle:.2f}, 刹车={brake:.2f}")
            
            # 计算需要按下的按键
            need_steering_left = steering > 0.06
            need_steering_right = steering < -0.05
            need_throttle = throttle >= 0.25
            need_brake = brake >= 0.07
            
            # 处理转向按键
            # 释放不再需要的转向键
            if self.currently_pressed_keys['steering_left'] and not need_steering_left:
                self.keyboard.release(self.config['keyboard_layout']['steering_left'])
                self.currently_pressed_keys['steering_left'] = False
                print("释放左转向键")
            
            if self.currently_pressed_keys['steering_right'] and not need_steering_right:
                self.keyboard.release(self.config['keyboard_layout']['steering_right'])
                self.currently_pressed_keys['steering_right'] = False
                print("释放右转向键")
            
            # 按下需要的转向键
            if not self.currently_pressed_keys['steering_left'] and need_steering_left:
                self.keyboard.press(self.config['keyboard_layout']['steering_left'])
                self.currently_pressed_keys['steering_left'] = True
                print("按下左转向键")
            
            if not self.currently_pressed_keys['steering_right'] and need_steering_right:
                self.keyboard.press(self.config['keyboard_layout']['steering_right'])
                self.currently_pressed_keys['steering_right'] = True
                print("按下右转向键")
            
            # 处理油门按键
            if self.currently_pressed_keys['throttle'] and not need_throttle:
                self.keyboard.release(self.config['keyboard_layout']['throttle'])
                self.currently_pressed_keys['throttle'] = False
                print("释放油门键")
            
            if not self.currently_pressed_keys['throttle'] and need_throttle:
                self.keyboard.press(self.config['keyboard_layout']['throttle'])
                self.currently_pressed_keys['throttle'] = True
                print("按下油门键")
            
            # 处理刹车按键
            if self.currently_pressed_keys['brake'] and not need_brake:
                self.keyboard.release(self.config['keyboard_layout']['brake'])
                self.currently_pressed_keys['brake'] = False
                print("释放刹车键")
            
            if not self.currently_pressed_keys['brake'] and need_brake:
                self.keyboard.press(self.config['keyboard_layout']['brake'])
                self.currently_pressed_keys['brake'] = True
                print("按下刹车键")
                
        except Exception as e:
            logger.error(f"发送键盘输入时出错: {e}")
    
    def _release_all_keys(self):
        """
        释放所有按键
        """
        if not self.keyboard or not PYNPUT_AVAILABLE:
            return
        
        try:
            # 释放转向按键
            self.keyboard.release(self.config['keyboard_layout']['steering_left'])
            self.keyboard.release(self.config['keyboard_layout']['steering_right'])
            
            # 释放油门和刹车按键
            self.keyboard.release(self.config['keyboard_layout']['throttle'])
            self.keyboard.release(self.config['keyboard_layout']['brake'])
            
            # 更新按键状态跟踪
            self.currently_pressed_keys = {
                'steering_left': False,
                'steering_right': False,
                'throttle': False,
                'brake': False
            }
            
            print("已释放所有按键")
            
        except Exception as e:
            logger.error(f"释放按键时出错: {e}")
    
    def send_input(self, steering: float, throttle: float, brake: float, handbrake: bool = False):
        """
        发送输入指令到游戏
        
        Args:
            steering: 转向值 (-1.0 to 1.0)
            throttle: 油门值 (0.0 to 1.0)
            brake: 刹车值 (0.0 to 1.0)
            handbrake: 是否拉手刹
        """
        if not self.initialized:
            logger.error("输入模拟器未初始化")
            return
        
        # 限制值范围
        steering = max(-1.0, min(1.0, steering))
        throttle = max(0.0, min(1.0, throttle))
        brake = max(0.0, min(1.0, brake))
        
        # 应用平滑
        smoothed_steering, smoothed_throttle, smoothed_brake = self._apply_smoothing(
            steering, throttle, brake
        )
        
        # 更新当前状态
        self.current_steering = smoothed_steering
        self.current_throttle = smoothed_throttle
        self.current_brake = smoothed_brake
        self.current_handbrake = handbrake
        
        # 发送输入
        if self.input_method == 'gamepad' and VGAMEPAD_AVAILABLE:
            self._send_gamepad_input(smoothed_steering, smoothed_throttle, smoothed_brake)
        else:
            self._send_keyboard_input(smoothed_steering, smoothed_throttle, smoothed_brake)
        
        # 处理手刹
        if handbrake and self.input_method == 'gamepad' and VGAMEPAD_AVAILABLE:
            self.gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_B)
            self.gamepad.update()
        elif not handbrake and self.input_method == 'gamepad' and VGAMEPAD_AVAILABLE:
            self.gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_B)
            self.gamepad.update()
    
    def _input_thread_function(self):
        """
        输入更新线程函数
        """
        update_interval = 1.0 / self.config['polling_rate']
        
        while self.running:
            start_time = time.time()
            
            # 检查控制队列
            with self.control_queue_lock:
                if self.control_queue:
                    # 获取最新的控制指令
                    controls = self.control_queue.pop(0)
                    
                    # 发送输入
                    self.send_input(
                        controls.get('steering', 0.0),
                        controls.get('throttle', 0.0),
                        controls.get('brake', 0.0),
                        controls.get('handbrake', False)
                    )
                elif not self.config['continuous_input']:
                    # 如果不需要持续输入且队列为空，释放所有输入
                    self._release_all_keys()
                    
                    # 如果是手柄，重置所有轴
                    if self.gamepad and VGAMEPAD_AVAILABLE:
                        self.gamepad.reset()
                        self.gamepad.update()
            
            # 等待下一次更新
            elapsed = time.time() - start_time
            sleep_time = max(0, update_interval - elapsed)
            time.sleep(sleep_time)
    
    def start(self):
        """
        启动输入模拟器
        """
        if self.running:
            logger.warning("输入模拟器已经在运行")
            return
        
        self.running = True
        self.input_thread = threading.Thread(target=self._input_thread_function, daemon=True)
        self.input_thread.start()
        logger.info("输入模拟器已启动")
    
    def stop(self):
        """
        停止输入模拟器
        """
        if not self.running:
            logger.warning("输入模拟器未运行")
            return
        
        self.running = False
        if self.input_thread:
            self.input_thread.join(timeout=2.0)
            logger.info("输入模拟器已停止")
        
        # 释放所有输入
        self._release_all_keys()
        
        # 重置游戏手柄
        if self.gamepad and VGAMEPAD_AVAILABLE:
            self.gamepad.reset()
            self.gamepad.update()
    
    def queue_input(self, steering: float, throttle: float, brake: float, handbrake: bool = False):
        """
        将输入指令加入队列
        
        Args:
            steering: 转向值 (-1.0 to 1.0)
            throttle: 油门值 (0.0 to 1.0)
            brake: 刹车值 (0.0 to 1.0)
            handbrake: 是否拉手刹
        """
        controls = {
            'steering': steering,
            'throttle': throttle,
            'brake': brake,
            'handbrake': handbrake
        }
        
        with self.control_queue_lock:
            # 限制队列大小
            if len(self.control_queue) >= 10:
                self.control_queue.pop(0)
            
            # 添加到队列
            self.control_queue.append(controls)
    
    def send_single_input(self, steering: float, throttle: float, brake: float, handbrake: bool = False, duration: float = 0.1):
        """
        发送单次输入（非持续）
        
        Args:
            steering: 转向值 (-1.0 to 1.0)
            throttle: 油门值 (0.0 to 1.0)
            brake: 刹车值 (0.0 to 1.0)
            handbrake: 是否拉手刹
            duration: 输入持续时间 (秒)
        """
        original_continuous = self.config['continuous_input']
        
        # 临时设置为非持续输入
        self.config['continuous_input'] = False
        
        try:
            # 发送输入
            self.send_input(steering, throttle, brake, handbrake)
            
            # 等待指定时间
            time.sleep(duration)
            
            # 释放输入
            self.send_input(0.0, 0.0, 0.0, False)
            
        finally:
            # 恢复原始设置
            self.config['continuous_input'] = original_continuous
    
    def press_key(self, key: str, duration: float = 0.1):
        """
        按指定的键盘按键
        
        Args:
            key: 要按的键
            duration: 按键持续时间 (秒)
        """
        if not self.keyboard or not PYNPUT_AVAILABLE:
            logger.error("键盘控制器未初始化")
            return
        
        try:
            self.keyboard.press(key)
            time.sleep(duration)
            self.keyboard.release(key)
            
            if self.config['debug']:
                logger.debug(f"按键: {key}, 持续时间: {duration}秒")
                
        except Exception as e:
            logger.error(f"按键时出错: {e}")
    
    def send_gear_input(self, gear_change: str):
        """
        发送档位输入
        
        Args:
            gear_change: 档位变化 ('up', 'down')
        """
        if gear_change == 'up':
            self.press_key(self.config['keyboard_layout']['gear_up'])
        elif gear_change == 'down':
            self.press_key(self.config['keyboard_layout']['gear_down'])
    
    def reset_input(self):
        """
        重置所有输入
        """
        # 重置状态变量
        self.current_steering = 0.0
        self.current_throttle = 0.0
        self.current_brake = 0.0
        self.current_handbrake = False
        
        # 清空控制队列
        with self.control_queue_lock:
            self.control_queue.clear()
        
        # 释放所有按键
        self._release_all_keys()
        
        # 重置游戏手柄
        if self.gamepad and VGAMEPAD_AVAILABLE:
            self.gamepad.reset()
            self.gamepad.update()
        
        logger.info("输入已重置")
    
    def update_config(self, new_config: Dict):
        """
        更新配置
        
        Args:
            new_config: 新的配置字典
        """
        # 如果更改了输入方法，需要重新初始化
        old_method = self.input_method
        
        # 更新配置
        self.config.update(new_config)
        
        # 检查输入方法是否改变
        if 'input_method' in new_config:
            self.input_method = new_config['input_method'].lower()
            if self.input_method != old_method:
                logger.info(f"输入方法已切换: {old_method} -> {self.input_method}")
                # 停止当前输入
                self.stop()
                # 重新初始化
                self.initialize()
                # 重新启动
                if self.running:
                    self.start()
    
    def get_input_method(self) -> str:
        """
        获取当前输入方法
        
        Returns:
            当前输入方法
        """
        return self.input_method
    
    def get_current_state(self) -> Dict:
        """
        获取当前输入状态
        
        Returns:
            当前输入状态字典
        """
        return {
            'steering': self.current_steering,
            'throttle': self.current_throttle,
            'brake': self.current_brake,
            'handbrake': self.current_handbrake,
            'input_method': self.input_method,
            'running': self.running
        }
    
    def calibrate_input(self, steering_center: float = 0.0, throttle_zero: float = 0.0, brake_zero: float = 0.0):
        """
        校准输入设备
        
        Args:
            steering_center: 转向中心偏移
            throttle_zero: 油门零点偏移
            brake_zero: 刹车零点偏移
        """
        # 这里可以实现输入校准逻辑
        # 例如存储校准参数，在发送输入时应用
        logger.info(f"输入校准已应用: 转向中心={steering_center}, 油门零点={throttle_zero}, 刹车零点={brake_zero}")

# 示例用法
def example_usage():
    # 创建输入模拟器实例
    config = {
        'input_method': 'keyboard',  # 或者 'gamepad'
        'enable_smoothing': True,
        'smoothing_factor': 0.3,
        'debug': True
    }
    
    simulator = InputSimulator(config)
    
    # 启动输入模拟器
    simulator.start()
    
    try:
        # 模拟基本操作
        print("测试基本控制...")
        
        # 直行加速
        simulator.queue_input(0.0, 0.8, 0.0)
        time.sleep(2)
        
        # 左转
        simulator.queue_input(-0.5, 0.6, 0.0)
        time.sleep(1)
        
        # 右转
        simulator.queue_input(0.5, 0.6, 0.0)
        time.sleep(1)
        
        # 减速
        simulator.queue_input(0.0, 0.2, 0.4)
        time.sleep(2)
        
        print("测试完成")
        
    finally:
        # 停止输入模拟器
        simulator.stop()
        simulator.reset_input()

if __name__ == "__main__":
    example_usage()