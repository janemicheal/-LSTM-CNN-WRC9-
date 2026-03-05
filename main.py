import os
import sys
import time
import logging
import threading
import numpy as np
import mss
import mss.tools
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union
import cv2
import pygetwindow as gw
import keyboard
from PIL import Image

# 导入tkinter用于简单UI
import tkinter as tk
from tkinter import ttk

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# 导入实际实现的类
from src.input_simulator import InputSimulator
from src.screen_capture import ScreenCapture
from src.model_inference import ModelInference
from src.roadbook_recognizer import RoadbookRecognizer

# 配置日志

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("wrc9_autodrive.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("WRC9AutoDrive")

# 已从src目录导入实际实现的类

class WRC9AutoDrive:
    def __init__(self):
        self.running = False
        self.stop_event = threading.Event()
        self.auto_driving = False  # 添加auto_driving状态变量
        self.recording = False  # 添加recording状态变量
        self.screen_capture = ScreenCapture()
        self.model_inference = ModelInference()
        self.input_simulator = InputSimulator()
        self.roadbook_recognizer = RoadbookRecognizer()
        self.config = {"debug": False, "stop_key": "q", "roadbook": {"enabled": True}}
        self.thread_list = []
        self.inference_thread = None
        
        # 初始化数据录制器
        self.data_recorder = None
        
        # 初始化UI窗口
        self.ui_root = None
        self.status_label = None
        self.recording_label = None
        self.create_status_window()
    
    def create_status_window(self):
        """
        创建一个简单的状态显示窗口
        """
        try:
            # 创建主窗口
            self.ui_root = tk.Tk()
            self.ui_root.title("WRC9自动驾驶状态")
            self.ui_root.geometry("300x200")
            self.ui_root.resizable(False, False)
            self.ui_root.attributes('-topmost', True)  # 窗口置顶
            self.ui_root.configure(bg='#f0f0f0')
            
            # 创建标题
            title_label = tk.Label(self.ui_root, text="WRC9自动驾驶系统", font=("Arial", 14, "bold"), bg='#f0f0f0')
            title_label.pack(pady=10)
            
            # 创建自动驾驶状态标签
            self.status_label = tk.Label(self.ui_root, text="自动驾驶: 未启动", font=("Arial", 12), bg='#f0f0f0')
            self.status_label.pack(pady=5)
            
            # 创建录制状态标签
            self.recording_label = tk.Label(self.ui_root, text="录制状态: 未录制", font=("Arial", 12), bg='#f0f0f0')
            self.recording_label.pack(pady=5)
            
            # 创建状态指示灯
            self.status_indicator = tk.Label(self.ui_root, width=20, height=2, bg="red", relief="sunken")
            self.status_indicator.pack(pady=5)
            
            # 创建录制指示灯
            self.recording_indicator = tk.Label(self.ui_root, width=20, height=2, bg="red", relief="sunken")
            self.recording_indicator.pack(pady=5)
            
            # 创建提示标签
            hint_label = tk.Label(self.ui_root, text="按K键切换自动驾驶，按J键切换录制", font=("Arial", 10), fg="gray", bg='#f0f0f0')
            hint_label.pack(pady=5)
            
            # 启动UI线程
            ui_thread = threading.Thread(target=self._run_ui, daemon=True)
            ui_thread.start()
            self.thread_list.append(ui_thread)
            
            print("✅ 状态显示窗口已创建")
        except Exception as e:
            print(f"❌ 创建状态窗口时出错: {e}")
            self.ui_root = None
    
    def _run_ui(self):
        """
        运行UI主循环
        """
        if self.ui_root:
            self.ui_root.mainloop()
    
    def update_status_window(self):
        """
        更新状态窗口显示
        """
        if self.ui_root and self.status_label and self.recording_label:
            try:
                # 更新自动驾驶状态
                if self.auto_driving:
                    auto_status_text = "自动驾驶: 🚗 已启动"
                    auto_indicator_color = "green"
                else:
                    auto_status_text = "自动驾驶: 🛑 未启动"
                    auto_indicator_color = "red"
                
                # 更新录制状态
                if self.recording:
                    recording_status_text = "录制状态: 📹 录制中"
                    recording_indicator_color = "yellow"
                else:
                    recording_status_text = "录制状态: � 未录制"
                    recording_indicator_color = "red"
                
                # 更新UI
                self.status_label.config(text=auto_status_text)
                self.status_indicator.config(bg=auto_indicator_color)
                self.recording_label.config(text=recording_status_text)
                self.recording_indicator.config(bg=recording_indicator_color)
                self.ui_root.update()
            except Exception as e:
                print(f"❌ 更新状态窗口时出错: {e}")
    
    class ControlPanel:
        def __init__(self):
            self.auto_drive_active = False  # 跟踪自动驾驶状态
            self.status_label = "就绪"
            self.auto_drive_button = "启动自动驾驶"
    
    def run_inference(self) -> bool:
        """
        执行自动驾驶推理模式
        """
        # 初始化性能计数器
        frame_count = 0
        total_time = 0
        fps_history = []
        capture_failures = 0
        inference_success = False
        
        max_failures = 5
        target_fps = 30  # 目标帧率
        frame_time = 1.0 / target_fps  # 目标帧时间
        
        # 创建备用的mss实例以提高稳定性
        backup_mss = None
        backup_mss_available = False
        try:
            import mss
            backup_mss = mss()
            backup_mss_available = True
            print("备用MSS实例创建成功")
            logger.info("备用MSS实例创建成功")
        except Exception as e:
            print(f"备用MSS实例创建失败: {e}")
            logger.warning(f"备用MSS实例创建失败: {e}")
        
        # 预定义捕获区域（避免每次重新创建）
        capture_region = {'top': 0, 'left': 0, 'width': 800, 'height': 450}
        
        try:
            # 检查核心模块是否初始化
            print("检查核心模块...")
            logger.info("检查推理模式所需模块是否初始化")
            if not self.screen_capture:
                print("错误：屏幕捕获模块未初始化")
                logger.error("屏幕捕获模块未初始化")
                return False
            if not self.model_inference:
                print("错误：模型推理模块未初始化")
                logger.error("模型推理模块未初始化")
                return False
            if not self.input_simulator:
                print("错误：输入模拟模块未初始化")
                logger.error("输入模拟模块未初始化")
                return False
            
            print("所有核心模块检查通过")
            logger.info("所有核心模块检查通过")
            print("自动驾驶模式已启动")
            print("请确保WRC9游戏已运行")
            print(f"按 '{self.config.get('stop_key', 'q')}' 键退出")
            
            # 启动输入模拟器
            print("尝试启动输入模拟器...")
            if hasattr(self.input_simulator, 'start'):
                try:
                    self.input_simulator.start()
                    print("输入模拟器启动成功")
                    logger.info("输入模拟器已启动")
                except Exception as e:
                    print(f"警告：启动输入模拟器时出错: {e}")
                    logger.warning(f"启动输入模拟器时出错: {e}")
            else:
                print("输入模拟器没有start方法，跳过启动")
            
            # 启动键盘监控线程
            print("启动键盘监控线程...")
            self.start_keyboard_monitor()
            print("键盘监控线程已启动")
            
            # 调试：打印输入模拟器信息
            print("===== 输入模拟器信息 =====")
            if self.input_simulator:
                sim_type = type(self.input_simulator).__name__
                print(f"输入模拟器类型: {sim_type}")
                if hasattr(self.input_simulator, '_keyboard_layout'):
                    print(f"键盘布局配置: {self.input_simulator._keyboard_layout}")
                else:
                    print("输入模拟器没有_keyboard_layout属性")
            else:
                print("输入模拟器未初始化")
            
            # 确保运行标志设置正确
            self.running = True
            print(f"运行标志设置: {self.running}, 停止事件: {self.stop_event.is_set()}")
            print("即将进入推理循环...")
            
            # 使用running标志控制循环
            counter = 0
            print("开始推理循环")
            print("按K键启动/停止自动驾驶")
            
            while self.running and not self.stop_event.is_set():
                # 正常执行循环
                counter += 1
                print(f"\n===== 推理循环迭代 #{counter} =====")
                print(f"当前自动驾驶状态: {'已启动' if self.auto_driving else '已停止'}")
                
                # 更新状态显示窗口
                self.update_status_window()
                
                # 捕获游戏画面
                print("\n===== 捕获屏幕画面 =====")
                frame = None
                capture_start = time.time()
                
                try:
                    # 确保ScreenCapture实例状态正确
                    if hasattr(self.screen_capture, 'running') and not self.screen_capture.running:
                        self.screen_capture.running = True
                    
                    # 直接使用单线程捕获方法
                    print("调用_capture_single_frame()进行单帧捕获...")
                    frame = self.screen_capture._capture_single_frame()
                    
                    if frame is not None and frame.size > 0:
                        print("成功捕获帧")
                        print(f"  - 形状: {frame.shape}")
                        capture_failures = 0  # 重置失败计数
                    else:
                        print(f"未捕获到有效帧")
                        capture_failures += 1
                        
                        # 尝试备用MSS实例
                        if backup_mss_available:
                            try:
                                print("尝试使用备用MSS实例捕获...")
                                screenshot = backup_mss.grab(capture_region)
                                frame = np.array(screenshot)
                                if len(frame.shape) == 3 and frame.shape[-1] == 4:  # BGRA
                                    frame = frame[:,:,:3]  # 转换为RGB
                                print(f"备用MSS成功捕获: {frame.shape}")
                                capture_failures = 0  # 重置失败计数
                            except Exception as e:
                                print(f"备用MSS捕获失败: {e}")
                except Exception as e:
                    print(f"捕获画面时出错: {e}")
                    capture_failures += 1
                
                # 检查是否达到最大失败次数
                if capture_failures >= max_failures:
                    print(f"错误：连续{max_failures}次捕获失败，尝试重新初始化屏幕捕获...")
                    try:
                        if hasattr(self, 'screen_capture') and self.screen_capture:
                            self.screen_capture.stop()
                            if hasattr(self.screen_capture, 'start'):
                                self.screen_capture.start()
                    except Exception as e:
                        print(f"重新初始化屏幕捕获失败: {e}")
                    capture_failures = 0
                    time.sleep(1.0)
                    continue
                
                # 检查是否需要录制帧
                if self.recording and self.data_recorder:
                    try:
                        # 将捕获的帧传递给录制器
                        # 使用SimpleRecorder的record方法，需要传递frame, roadbook_info, controls
                        self.data_recorder.record(frame=frame, roadbook_info="", controls={'steering': 0.0, 'throttle': 0.0, 'brake': 0.0})
                    except Exception as e:
                        print(f"录制帧时出错: {e}")
                
                # 只有当自动驾驶开启时才执行模型预测和控制
                if self.auto_driving:
                    # 执行路书识别
                    roadbook_info = None
                    roadbook_confidence = 0.0
                    if self.config.get('roadbook', {}).get('enabled', True) and self.roadbook_recognizer:
                        print("执行路书识别...")
                        print("路书识别配置: enabled=True")
                        try:
                            print("调用roadbook_recognizer.recognize_roadbook()...")
                            roadbook_info, roadbook_confidence = self.roadbook_recognizer.recognize_roadbook(frame)
                            print(f"成功识别路书: ({roadbook_info}, {roadbook_confidence})")
                        except Exception as e:
                            print(f"路书识别出错: {e}")
                    
                    # 执行模型预测
                    prediction_result = None
                    inference_time = 0.0
                    try:
                        print("执行模型预测...")
                        print("调用model_inference.predict()...")
                        # 确保roadbook_info是字典格式，避免类型错误
                        if isinstance(roadbook_info, dict):
                            roadbook_dict = roadbook_info
                        else:
                            # 如果不是字典，使用默认空字典
                            roadbook_dict = {}
                        prediction_result, inference_time = self.model_inference.predict(frame, roadbook=roadbook_dict)
                        print(f"成功完成模型预测，推理时间: {inference_time:.2f}ms")
                        # 添加预测结果输出
                        if prediction_result:
                            steering = prediction_result.get('steering', 0.0)
                            throttle = prediction_result.get('throttle', 0.0)
                            brake = prediction_result.get('brake', 0.0)
                            print(f"模型预测结果: 转向={steering:.2f}, 油门={throttle:.2f}, 刹车={brake:.2f}")
                    except Exception as e:
                        print(f"模型预测出错: {e}")
                    
                    # 发送控制指令到输入模拟器
                    if prediction_result is not None and self.input_simulator:
                        try:
                            steering = prediction_result.get('steering', 0.0)
                            throttle = prediction_result.get('throttle', 0.0)
                            brake = prediction_result.get('brake', 0.0)
                            print(f"准备发送控制指令: 转向={steering:.2f}, 油门={throttle:.2f}, 刹车={brake:.2f}")
                            self.input_simulator.send_input(steering, throttle, brake)
                            print("成功发送控制指令")
                        except Exception as e:
                            print(f"执行输入模拟时出错: {e}")
                else:
                    print("自动驾驶未启动，跳过模型预测和控制指令发送")
                
                # 性能统计
                frame_count += 1
                
                # 定期显示性能信息
                if frame_count % 10 == 0:
                    avg_inference_time = 0
                    if hasattr(self.model_inference, 'get_average_inference_time'):
                        try:
                            avg_inference_time = self.model_inference.get_average_inference_time()
                        except Exception as e:
                            print(f"获取平均推理时间时出错: {e}")
                    
                    if self.config.get('debug', False):
                        print(f"FPS: 30.0, 推理时间: {avg_inference_time:.1f}ms")
                
                # 控制帧率
                time.sleep(0.01)
            
            # 循环正常结束
            print("推理循环正常结束")
            inference_success = True
            
        except KeyboardInterrupt:
            print("自动驾驶被用户中断")
            inference_success = True
        except Exception as e:
            print(f"自动驾驶运行出错: {e}")
            inference_success = False
        finally:
            print("执行推理模式清理操作")
            # 确保备用MSS实例被正确关闭
            if backup_mss_available and backup_mss is not None:
                try:
                    backup_mss.close()
                    print("备用MSS实例已关闭")
                except Exception as e:
                    print(f"关闭备用MSS实例时出错: {e}")
            # 调用清理方法
            self.stop_inference()
        
        print(f"推理模式结束，成功状态: {inference_success}")
        return inference_success
    
    def stop_inference(self):
        """
        停止自动驾驶推理
        """
        # 只设置停止标志，不调用_stop_auto_drive（避免线程join警告）
        self.running = False
        self.stop_event.set()
        print("推理模式停止标志已设置")
    
    def _start_auto_drive(self):
        """内部方法：启动自动驾驶"""
        self.auto_driving = True
        print("自动驾驶已启动")
        # 更新控制面板状态
        if hasattr(self, 'control_panel'):
            self.control_panel.auto_drive_active = True
            self.control_panel.auto_drive_button = "停止自动驾驶"
            self.control_panel.status_label = "自动驾驶中"
    
    def _stop_auto_drive(self):
        """内部方法：停止自动驾驶"""
        self.auto_driving = False
        print("自动驾驶已停止")
        # 停止输入模拟器
        if hasattr(self, 'input_simulator') and self.input_simulator:
            try:
                if hasattr(self.input_simulator, 'stop'):
                    self.input_simulator.stop()
                    print("输入模拟器已停止")
            except Exception as e:
                print(f"停止输入模拟器时出错: {e}")
        # 停止画面捕获
        if hasattr(self, 'screen_capture') and self.screen_capture:
            try:
                if hasattr(self.screen_capture, 'stop'):
                    self.screen_capture.stop()
                    print("画面捕获已停止")
            except Exception as e:
                print(f"停止画面捕获时出错: {e}")
        # 更新控制面板状态
        if hasattr(self, 'control_panel'):
            self.control_panel.auto_drive_active = False
            self.control_panel.auto_drive_button = "启动自动驾驶"
            self.control_panel.status_label = "已停止"
        # 等待推理线程结束，但避免在当前线程中join自身
        if hasattr(self, 'inference_thread') and self.inference_thread is not None and self.inference_thread.is_alive():
            # 检查当前线程是否为推理线程
            current_thread = threading.current_thread()
            if current_thread.ident == self.inference_thread.ident:
                print("警告：避免在推理线程内部join自身 - 跳过join操作")
            else:
                self.inference_thread.join(timeout=5.0)
                print("推理线程已停止")
        # 清理线程列表
        for thread in self.thread_list[:]:
            if thread.is_alive():
                try:
                    if hasattr(thread, 'join'):
                        thread.join(timeout=1.0)
                except Exception as e:
                    print(f"等待线程结束时出错: {e}")
            self.thread_list.remove(thread)
    
    def start_auto_drive(self):
        """公共方法：启动自动驾驶"""
        self._start_auto_drive()
    
    def stop_auto_drive(self):
        """公共方法：停止自动驾驶"""
        self._stop_auto_drive()
    
    def start_keyboard_monitor(self) -> bool:
        """
        启动键盘监控线程
        """
        print("启动键盘监控...")
        
        def on_key_press(event):
            """
            按键按下事件处理函数
            """
            try:
                # 检查退出键
                if event.name == 'esc':
                    print("检测到ESC键，准备退出...")
                    self.stop_event.set()
                    self.running = False
                
                # 检查录制模式键，使用J键避免与游戏冲突，对大小写不敏感
                elif event.name.lower() == 'j':
                    print("检测到J键，切换录制模式")
                    # 切换录制状态
                    self.recording = not self.recording
                    
                    if self.recording:
                        print("开始录制")
                        # 初始化数据录制器
                        try:
                            from src.simple_recorder import SimpleRecorder
                            self.data_recorder = SimpleRecorder()
                            # 直接调用start方法，不启动新线程
                            self.data_recorder.start()
                            print("录制已开始")
                            print("录制逻辑：开始录制后实时记录数据，按J键停止后自动保存到data/raw/目录")
                        except Exception as e:
                            print(f"初始化录制器失败: {e}")
                            self.recording = False
                    else:
                        print("停止录制")
                        # 停止录制
                        if self.data_recorder:
                            try:
                                self.data_recorder.stop()
                                print("录制已停止")
                                print("数据已保存到data/raw/目录")
                            except Exception as e:
                                print(f"停止录制失败: {e}")
                    # 更新状态窗口
                    self.update_status_window()
                
                # 检查自动驾驶键，对大小写不敏感
                elif event.name.lower() == 'k':
                    print("检测到K键，切换自动驾驶状态")
                    # 切换auto_driving状态
                    if not self.auto_driving:
                        self.auto_driving = True
                        print("开始自动驾驶")
                        # 调用start_auto_drive方法启动自动驾驶
                        self.start_auto_drive()
                        # 更新控制面板状态
                        if hasattr(self, 'control_panel'):
                            self.control_panel.auto_drive_active = True
                            self.control_panel.auto_drive_button = "停止自动驾驶"
                            self.control_panel.status_label = "自动驾驶中"
                            # 更新按钮背景色
                            # 这里假设有update_button_style方法
                            if hasattr(self.control_panel, 'update_button_style'):
                                self.control_panel.update_button_style('auto_drive_button', 'background-color', '#ff4444')
                    else:
                        self.auto_driving = False
                        print("停止自动驾驶")
                        # 调用stop_auto_drive方法停止自动驾驶
                        self.stop_auto_drive()
                        # 更新控制面板状态
                        if hasattr(self, 'control_panel'):
                            self.control_panel.auto_drive_active = False
                            self.control_panel.auto_drive_button = "启动自动驾驶"
                            self.control_panel.status_label = "已停止"
                            # 更新按钮背景色
                            if hasattr(self.control_panel, 'update_button_style'):
                                self.control_panel.update_button_style('auto_drive_button', 'background-color', '#44ff44')
            except Exception as e:
                print(f"处理按键事件时出错: {e}")
        
        def keyboard_monitor_thread():
            try:
                print("键盘监控线程已启动")
                print(f"按ESC键退出，按J键切换录制模式，按K键切换自动驾驶")
                
                # 注册按键事件监听器
                keyboard.on_press(on_key_press)
                
                # 保持线程运行
                while self.running and not self.stop_event.is_set():
                    time.sleep(0.1)  # 防止CPU占用过高
            except Exception as e:
                print(f"键盘监控线程错误: {e}")
            finally:
                # 清理事件监听器
                keyboard.unhook_all()
                print("键盘监控线程已退出")
        
        # 创建并启动线程
        try:
            monitor_thread = threading.Thread(target=keyboard_monitor_thread)
            monitor_thread.daemon = True
            monitor_thread.start()
            self.thread_list.append(monitor_thread)
            return True
        except Exception as e:
            print(f"启动键盘监控线程失败: {e}")
            return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='WRC9自动驾驶系统')
    parser.add_argument('--mode', choices=['inference', 'training', 'record'], default='inference', help='运行模式')
    parser.add_argument('--debug', action='store_true', help='启用调试模式')
    args = parser.parse_args()
    
    if args.mode == 'record':
        # 运行录制模式
        try:
            from src.data_recorder import DataRecorder
            recorder = DataRecorder()
            recorder.start_recording()
        except Exception as e:
            print(f"启动录制模式失败: {e}")
    else:
        auto_drive = WRC9AutoDrive()
        auto_drive.config['debug'] = args.debug
        
        # 添加调试信息，显示模型加载状态
        print("\n===== 系统初始化调试信息 =====")
        print(f"模型路径: {auto_drive.model_inference.config['model_path']}")
        print(f"模型是否加载成功: {'是' if auto_drive.model_inference.model_loaded else '否'}")
        print(f"输入方法: {auto_drive.input_simulator.input_method}")
        print(f"使用设备: {auto_drive.model_inference.device}")
        print(f"目标分辨率: {auto_drive.model_inference.config['target_resolution']}")
        print("==============================\n")
        
        if args.mode == 'inference':
            auto_drive.run_inference()
        else:
            print("训练模式暂未实现")
