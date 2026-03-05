"""
画面捕获模块 (ScreenCapture)

核心职责：实时捕获游戏画面，为模型推理提供输入数据。
功能：捕获游戏画面帧、预处理、ROI提取、画面增强等。
"""
import time
import threading
import numpy as np
import cv2
from PIL import Image
import logging
from typing import Dict, Tuple, Optional, List

# 尝试导入mss模块，添加异常处理
try:
    import mss
    import mss.tools
    MSS_AVAILABLE = True
except ImportError:
    MSS_AVAILABLE = False
    logging.warning("未找到mss模块，无法进行屏幕捕获。请使用 'pip install mss' 安装。")

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='../logs/screen_capture.log'
)
logger = logging.getLogger('ScreenCapture')

class ScreenCapture:
    """
    游戏画面捕获类
    使用mss库进行高效屏幕捕获，支持多显示器、ROI选择、画面预处理等功能
    """
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化画面捕获模块
        
        Args:
            config: 配置字典，包含捕获参数
        """
        # 默认配置
        self.default_config = {
            'monitor_id': 1,            # 显示器ID
            'capture_region': None,     # 捕获区域 [左, 上, 宽, 高]，None表示全屏
            'target_resolution': (960, 540),  # 输出分辨率，与训练一致
            'fps': 30,                  # 目标帧率
            'grayscale': False,         # 是否转换为灰度图
            'apply_enhancement': False, # 是否应用画面增强
            'enable_roi': True,         # 启用ROI提取
            'roi_region': (430, 30, 100, 80),  # ROI区域 [左, 上, 宽, 高]，适配路书位置：水平430~530，垂直30~110
            'flip_image': False,        # 是否翻转图像
            'enable_multi_thread': True  # 是否使用多线程捕获
        }
        
        # 合并配置
        self.config = self.default_config.copy()
        if config:
            self.config.update(config)
        
        # 初始化mss（如果可用）
        self.sct = None
        if MSS_AVAILABLE:
            try:
                self.sct = mss.mss()
            except Exception as e:
                logger.error(f"初始化mss失败: {e}")
                self.sct = None
        else:
            logger.warning("mss模块不可用，无法进行屏幕捕获")
        
        # 初始化运行状态
        self.running = False
        
        # 获取显示器信息
        self.monitors = []
        if self.sct is not None:
            try:
                self.monitors = self.sct.monitors
                logger.info(f"找到 {len(self.monitors) - 1} 个显示器 (索引从1开始)")
                
                # 检查显示器ID是否有效
                if self.config['monitor_id'] < 1 or self.config['monitor_id'] >= len(self.monitors):
                    logger.warning(f"无效的显示器ID: {self.config['monitor_id']}, 使用默认显示器1")
                    self.config['monitor_id'] = 1
            except Exception as e:
                logger.error(f"获取显示器信息失败: {e}")
        else:
            logger.error("无法获取显示器信息，mss未初始化")
        
        # 设置捕获区域
        self._setup_capture_region()
        
        # 计算ROI坐标
        if self.config['enable_roi']:
            self._compute_roi_coordinates()
        
        # 初始化变量
        self.last_frame = None
        self.last_capture_time = 0
        self.fps_history = []
        self.running = False
        self.capture_thread = None
        self.frame_lock = threading.RLock()  # 用于线程安全的帧访问
    
    def _setup_capture_region(self):
        """
        设置捕获区域
        """
        # 默认捕获区域
        self.capture_region = {
            "top": 0,
            "left": 0,
            "width": 960,
            "height": 540
        }
        
        # 尝试从显示器获取信息
        if self.monitors and self.config['monitor_id'] < len(self.monitors):
            monitor = self.monitors[self.config['monitor_id']]
            
            if self.config['capture_region'] is None:
                # 全屏捕获
                self.capture_region = {
                    "top": monitor["top"],
                    "left": monitor["left"],
                    "width": monitor["width"],
                    "height": monitor["height"]
                }
            else:
                # 自定义区域捕获
                left, top, width, height = self.config['capture_region']
                self.capture_region = {
                    "top": monitor["top"] + top,
                    "left": monitor["left"] + left,
                    "width": width,
                    "height": height
                }
        else:
            logger.warning(f"无法获取有效显示器信息，使用默认捕获区域 {self.capture_region}")
        
        logger.info(f"设置捕获区域: {self.capture_region}")
        logger.info(f"原始分辨率: {self.capture_region['width']}x{self.capture_region['height']}")
        logger.info(f"目标分辨率: {self.config['target_resolution'][0]}x{self.config['target_resolution'][1]}")
    
    def _compute_roi_coordinates(self):
        """
        计算ROI坐标
        """
        roi_left, roi_top, roi_width, roi_height = self.config['roi_region']
        
        # 确保ROI在有效范围内
        monitor = {"left": 0, "top": 0, "width": 960, "height": 540}
        if self.monitors and self.config['monitor_id'] < len(self.monitors):
            monitor = self.monitors[self.config['monitor_id']]
        
        # 转换为相对于捕获区域的坐标
        if self.config['capture_region'] is None:
            # 相对于全屏的ROI
            self.roi_coords = (
                max(0, roi_left - monitor["left"]),
                max(0, roi_top - monitor["top"]),
                min(roi_width, monitor["width"]),
                min(roi_height, monitor["height"])
            )
        else:
            # 相对于自定义捕获区域的ROI
            cap_left = self.capture_region["left"]
            cap_top = self.capture_region["top"]
            
            # 计算ROI在捕获区域内的相对位置
            roi_rel_left = max(0, roi_left - cap_left)
            roi_rel_top = max(0, roi_top - cap_top)
            
            # 确保ROI不超出捕获区域
            roi_rel_width = min(roi_width, self.capture_region["width"] - roi_rel_left)
            roi_rel_height = min(roi_height, self.capture_region["height"] - roi_rel_top)
            
            self.roi_coords = (roi_rel_left, roi_rel_top, roi_rel_width, roi_rel_height)
        
        logger.info(f"计算ROI坐标: {self.roi_coords}")
    
    def _apply_enhancement(self, frame: np.ndarray) -> np.ndarray:
        """
        应用画面增强
        
        Args:
            frame: 输入图像帧
            
        Returns:
            增强后的图像
        """
        # 转换为HLS色彩空间以增强对比度和亮度
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            hls = cv2.cvtColor(frame, cv2.COLOR_RGB2HLS)
            
            # 分离通道
            h, l, s = cv2.split(hls)
            
            # 对比度增强 (使用CLAHE - 对比度受限自适应直方图均衡化)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            
            # 合并通道
            enhanced_hls = cv2.merge((h, cl, s))
            enhanced_frame = cv2.cvtColor(enhanced_hls, cv2.COLOR_HLS2RGB)
        else:
            # 灰度图处理
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced_frame = clahe.apply(frame)
        
        return enhanced_frame
    
    def _resize_frame(self, frame: np.ndarray, target_resolution: Tuple[int, int]) -> np.ndarray:
        """
        调整帧大小
        
        Args:
            frame: 输入图像帧
            target_resolution: 目标分辨率 (宽度, 高度)
            
        Returns:
            调整大小后的图像
        """
        # 使用INTER_AREA插值方法，通常在缩小图像时效果更好
        return cv2.resize(frame, target_resolution, interpolation=cv2.INTER_AREA)
    
    def _extract_roi(self, frame: np.ndarray) -> np.ndarray:
        """
        提取ROI区域
        
        Args:
            frame: 输入图像帧
            
        Returns:
            ROI区域图像
        """
        x, y, w, h = self.roi_coords
        return frame[y:y+h, x:x+w]
    
    def _capture_single_frame(self) -> np.ndarray:
        """
        捕获单个帧
        
        Returns:
            处理后的图像帧
        """
        # 检查mss是否可用
        if not MSS_AVAILABLE or self.sct is None:
            logger.warning("mss不可用，无法捕获画面")
            return None
            
        try:
            # 捕获屏幕
            sct_img = self.sct.grab(self.capture_region)
            
            # 转换为NumPy数组
            frame = np.array(sct_img)
            
            # 如果需要ROI，先提取ROI
            if self.config['enable_roi']:
                frame = self._extract_roi(frame)
            
            # 转换BGR为RGB (如果需要)
            # MSS返回的是BGRA格式，但我们只需要BGR/RGB
            if len(frame.shape) == 3 and frame.shape[2] == 4:
                frame = frame[:, :, :3]  # 移除alpha通道
            
            # 调整大小
            if self.config['target_resolution'] is not None:
                frame = self._resize_frame(frame, self.config['target_resolution'])
            
            # 应用画面增强
            if self.config['apply_enhancement']:
                frame = self._apply_enhancement(frame)
            
            # 转换为灰度图
            if self.config['grayscale'] and len(frame.shape) == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                frame = np.expand_dims(frame, axis=-1)  # 保持通道维度
            
            # 翻转图像
            if self.config['flip_image']:
                frame = cv2.flip(frame, 1)  # 水平翻转
            
            return frame
            
        except Exception as e:
            logger.error(f"捕获帧时出错: {e}")
            return None
    
    def _threaded_capture(self):
        """
        多线程捕获循环
        """
        self.running = True
        frame_interval = 1.0 / self.config['fps']
        
        while self.running:
            start_time = time.time()
            
            # 捕获帧
            frame = self._capture_single_frame()
            
            if frame is not None:
                # 线程安全地更新最新帧
                with self.frame_lock:
                    self.last_frame = frame
                    self.last_capture_time = start_time
                
                # 计算FPS
                elapsed = time.time() - start_time
                current_fps = 1.0 / elapsed if elapsed > 0 else 0
                
                # 更新FPS历史
                self.fps_history.append(current_fps)
                if len(self.fps_history) > 100:  # 保持最近100个FPS记录
                    self.fps_history.pop(0)
                
                # 根据目标FPS调整捕获间隔
                sleep_time = max(0, frame_interval - elapsed)
                time.sleep(sleep_time)
            else:
                # 如果捕获失败，稍作延迟再试
                time.sleep(frame_interval)
    
    def start(self):
        """
        开始捕获画面
        """
        if self.running:
            logger.warning("捕获已经在运行")
            return False
        
        try:
            if self.config['enable_multi_thread']:
                # 启动捕获线程
                self.capture_thread = threading.Thread(target=self._threaded_capture, daemon=True)
                self.capture_thread.start()
                logger.info("启动多线程捕获")
            else:
                # 单线程模式直接设置运行标志
                self.running = True
                logger.info("启动单线程捕获")
            
            return True
            
        except Exception as e:
            logger.error(f"启动捕获线程时出错: {e}")
            self.running = False
            return False
    
    def stop(self):
        """
        停止捕获画面
        """
        if not self.running:
            logger.warning("捕获未运行")
            return False
        
        try:
            self.running = False
            
            # 等待线程结束
            if self.capture_thread and self.capture_thread.is_alive():
                self.capture_thread.join(timeout=2.0)
                logger.info("捕获线程已停止")
            
            return True
            
        except Exception as e:
            logger.error(f"停止捕获时出错: {e}")
            return False
    
    def get_latest_frame(self) -> Optional[np.ndarray]:
        """
        获取最新的捕获帧
        
        Returns:
            最新的图像帧，如果没有帧则返回None
        """
        # 单线程模式下直接捕获
        if not self.config['enable_multi_thread'] and self.running:
            frame = self._capture_single_frame()
            if frame is not None:
                self.last_frame = frame
                self.last_capture_time = time.time()
        
        # 线程安全地获取最后一帧
        with self.frame_lock:
            return self.last_frame.copy() if self.last_frame is not None else None
    
    def capture_and_save(self, output_path: str) -> bool:
        """
        捕获一帧并保存到文件
        
        Args:
            output_path: 输出文件路径
            
        Returns:
            是否成功
        """
        try:
            frame = self._capture_single_frame()
            if frame is None:
                return False
            
            # 确保输出路径有效
            if not output_path:
                output_path = f"screenshot_{int(time.time())}.png"
            
            # 转换为PIL图像并保存
            if len(frame.shape) == 3 and frame.shape[2] == 1:
                # 灰度图
                pil_img = Image.fromarray(frame[:, :, 0], mode='L')
            else:
                # 彩色图
                pil_img = Image.fromarray(frame)
            
            pil_img.save(output_path)
            logger.info(f"截图已保存到: {output_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"保存截图时出错: {e}")
            return False
    
    def get_current_fps(self) -> float:
        """
        获取当前FPS
        
        Returns:
            平均FPS值
        """
        if not self.fps_history:
            return 0.0
        
        # 计算最近10帧的平均FPS
        recent_fps = self.fps_history[-10:] if len(self.fps_history) >= 10 else self.fps_history
        return sum(recent_fps) / len(recent_fps)
    
    def get_monitor_info(self, monitor_id: int) -> Optional[Dict]:
        """
        获取指定显示器的信息
        
        Args:
            monitor_id: 显示器ID (从1开始)
            
        Returns:
            显示器信息字典，如果ID无效则返回None
        """
        if monitor_id < 1 or monitor_id >= len(self.monitors):
            logger.warning(f"无效的显示器ID: {monitor_id}")
            return None
        
        return self.monitors[monitor_id]
    
    def list_available_monitors(self) -> List[Dict]:
        """
        列出所有可用的显示器
        
        Returns:
            显示器信息列表
        """
        # 跳过第一个监视器（索引0），它是虚拟的全桌面监视器
        return self.monitors[1:]
    
    def calibrate_capture_region(self, window_title: str = None) -> bool:
        """
        校准捕获区域（可选功能）
        可以通过窗口标题自动定位游戏窗口
        
        Args:
            window_title: 窗口标题，如果为None则不执行校准
            
        Returns:
            是否成功校准
        """
        # 这个功能需要额外的窗口管理库，如pygetwindow或win32gui
        # 这里提供一个简单的实现框架
        if not window_title:
            logger.warning("未提供窗口标题，跳过校准")
            return False
        
        try:
            # 尝试导入窗口管理库
            try:
                import win32gui
                import win32con
                
                # 查找窗口
                hwnd = win32gui.FindWindow(None, window_title)
                if hwnd:
                    # 获取窗口矩形
                    left, top, right, bottom = win32gui.GetWindowRect(hwnd)
                    width = right - left
                    height = bottom - top
                    
                    # 更新捕获区域
                    self.config['capture_region'] = (left, top, width, height)
                    self._setup_capture_region()
                    
                    logger.info(f"成功校准捕获区域为窗口 '{window_title}': {self.capture_region}")
                    return True
                else:
                    logger.warning(f"未找到窗口: {window_title}")
                    return False
                    
            except ImportError:
                logger.warning("需要安装pywin32库以使用窗口校准功能")
                return False
                
        except Exception as e:
            logger.error(f"校准捕获区域时出错: {e}")
            return False
    
    def __del__(self):
        """
        析构函数，确保资源正确释放
        """
        try:
            self.stop()
            if hasattr(self, 'sct') and self.sct is not None:
                try:
                    self.sct.close()
                except Exception as e:
                    logger.error(f"关闭sct时发生错误: {e}")
        except Exception as e:
            logger.error(f"析构函数中发生错误: {e}")

# 示例用法
def example_usage():
    # 创建捕获实例
    capture_config = {
        'monitor_id': 1,
        'target_resolution': (800, 450),
        'fps': 30,
        'enable_roi': True,
        'roi_region': (0, 100, 1920, 900)  # 假设游戏画面在这个区域
    }
    
    screen_capture = ScreenCapture(capture_config)
    
    try:
        # 开始捕获
        if screen_capture.start():
            print("画面捕获已开始")
            
            # 捕获几帧
            for i in range(5):
                frame = screen_capture.get_latest_frame()
                if frame is not None:
                    print(f"捕获到帧 {i+1}: {frame.shape}")
                    # 保存帧
                    screen_capture.capture_and_save(f"test_frame_{i+1}.png")
                time.sleep(1)
            
            # 获取FPS
            print(f"当前FPS: {screen_capture.get_current_fps():.2f}")
            
        else:
            print("无法启动画面捕获")
            
    finally:
        # 确保停止捕获
        screen_capture.stop()
        print("画面捕获已停止")

if __name__ == "__main__":
    example_usage()