"""
简单录制类 (SimpleRecorder)

核心职责：记录游戏画面、路书信息和操作指令，保存到指定目录
"""
import os
import time
import numpy as np
import cv2
from datetime import datetime
import threading

class SimpleRecorder:
    """
    简单的录制类，负责记录和保存数据
    """
    def __init__(self, output_dir="data/raw", fps=15):
        """
        初始化录制器
        
        Args:
            output_dir: 数据保存目录
            fps: 录制帧率
        """
        # 使用绝对路径，确保数据保存到正确位置
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.output_dir = os.path.join(self.base_dir, output_dir)
        self.fps = fps
        self.frame_interval = 1.0 / fps
        
        # 录制状态
        self.is_recording = False
        self.recording_data = []
        self.start_time = None
        self.last_frame_time = None
        
        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"✅ 录制保存目录: {self.output_dir}")
    
    def start(self):
        """
        开始录制
        """
        if self.is_recording:
            print("⚠️  已经在录制中")
            return
        
        self.is_recording = True
        self.recording_data = []  # 清空之前的数据
        self.start_time = time.time()
        self.last_frame_time = self.start_time
        
        print("📹 开始录制")
        print(f"📝 录制配置: 帧率={self.fps}")
    
    def stop(self):
        """
        停止录制并保存数据
        """
        if not self.is_recording:
            print("⚠️  没有在录制中")
            return
        
        self.is_recording = False
        
        if self.recording_data:
            # 保存数据
            self._save_data()
            print(f"✅ 录制完成，共 {len(self.recording_data)} 帧数据")
        else:
            print("❌ 没有录制到数据")
    
    def record(self, frame, roadbook_info, controls):
        """
        录制一帧数据
        
        Args:
            frame: 游戏画面帧
            roadbook_info: 路书信息
            controls: 操作指令
        """
        if not self.is_recording:
            return
        
        current_time = time.time()
        
        # 控制录制帧率
        if current_time - self.last_frame_time >= self.frame_interval:
            timestamp = current_time - self.start_time
            
            # 保存数据样本
            data_sample = {
                'timestamp': timestamp,
                'frame': frame,
                'roadbook_info': roadbook_info,
                'controls': controls
            }
            self.recording_data.append(data_sample)
            
            # 显示录制状态
            if len(self.recording_data) % 10 == 0:
                print(f"📊 已录制 {len(self.recording_data)} 帧数据")
            
            self.last_frame_time = current_time
    
    def _save_data(self):
        """
        保存录制的数据
        """
        try:
            # 生成唯一的文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.output_dir, f"recording_{timestamp}.npz")
            
            # 准备数据
            frames = []
            roadbook_infos = []
            steerings = []
            throttles = []
            brakes = []
            timestamps = []
            
            for sample in self.recording_data:
                frames.append(sample['frame'])
                roadbook_infos.append(str(sample['roadbook_info']))
                steerings.append(sample['controls'].get('steering', 0.0))
                throttles.append(sample['controls'].get('throttle', 0.0))
                brakes.append(sample['controls'].get('brake', 0.0))
                timestamps.append(sample['timestamp'])
            
            # 转换为numpy数组
            frames = np.array(frames)
            roadbook_infos = np.array(roadbook_infos, dtype=str)
            steerings = np.array(steerings, dtype=np.float32)
            throttles = np.array(throttles, dtype=np.float32)
            brakes = np.array(brakes, dtype=np.float32)
            timestamps = np.array(timestamps, dtype=np.float64)
            
            # 保存数据
            np.savez(filename, 
                    frames=frames,
                    roadbook_infos=roadbook_infos,
                    steerings=steerings,
                    throttles=throttles,
                    brakes=brakes,
                    timestamps=timestamps)
            
            print(f"💾 数据已保存到: {filename}")
            return True
        except Exception as e:
            print(f"❌ 保存数据失败: {e}")
            return False
    
    def get_status(self):
        """
        获取录制状态
        
        Returns:
            bool: 是否正在录制
        """
        return self.is_recording
    
    def get_recording_count(self):
        """
        获取已录制的帧数
        
        Returns:
            int: 已录制的帧数
        """
        return len(self.recording_data)
