"""
驾驶数据录制模块 (DataRecorder)

核心职责：同步录制游戏画面、路书信息、人类操作指令，生成训练原始数据。
"""
import time
import numpy as np
import h5py
import cv2
from datetime import datetime
import os

# 尝试导入pynput模块，添加异常处理
try:
    from pynput import keyboard
    PYNPUT_AVAILABLE = True
except ImportError:
    PYNPUT_AVAILABLE = False
    print("警告: 未找到pynput模块，无法进行键盘监听。请使用 'pip install pynput' 安装。")

# 尝试导入mss模块，添加异常处理
try:
    import mss
    import mss.tools
    MSS_AVAILABLE = True
except ImportError:
    MSS_AVAILABLE = False
    print("警告: 未找到mss模块，无法进行屏幕捕获。请使用 'pip install mss' 安装。")

# 尝试导入pytesseract模块，添加异常处理
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    print("警告: 未找到pytesseract模块，无法进行OCR识别。请使用 'pip install pytesseract' 安装。")

class DataRecorder:
    """驾驶数据录制器类"""
    
    def __init__(self, output_dir="data/raw", fps=15):
        """
        初始化数据录制器
        
        Args:
            output_dir: 数据保存目录
            fps: 录制帧率
        """
        # 使用绝对路径，确保数据保存到正确位置
        self.base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.output_dir = os.path.join(self.base_dir, output_dir)
        self.fps = fps
        self.frame_interval = 1.0 / fps
        self.running = False
        self.recording_data = []
        
        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"✅ 数据保存目录: {self.output_dir}")
        
        # 操作状态
        self.controls = {
            'steering': 0.0,  # -1.0 到 1.0
            'throttle': 0.0,   # 0.0 到 1.0
            'brake': 0.0       # 0.0 到 1.0
        }
    
    def _on_press(self, key):
        """键盘按下事件处理"""
        try:
            if key == keyboard.Key.left:
                self.controls['steering'] = -1.0
            elif key == keyboard.Key.right:
                self.controls['steering'] = 1.0
            elif key == keyboard.Key.up:
                self.controls['throttle'] = 1.0
            elif key == keyboard.Key.down:
                self.controls['brake'] = 1.0
            elif key == keyboard.Key.esc:
                self.stop_recording()
                return False
        except Exception as e:
            print(f"按键处理错误: {e}")
    
    def _on_release(self, key):
        """键盘释放事件处理"""
        try:
            if key == keyboard.Key.left or key == keyboard.Key.right:
                self.controls['steering'] = 0.0
            elif key == keyboard.Key.up:
                self.controls['throttle'] = 0.0
            elif key == keyboard.Key.down:
                self.controls['brake'] = 0.0
        except Exception as e:
            print(f"按键释放处理错误: {e}")
    
    def capture_screen(self):
        """捕获游戏画面帧"""
        screenshot = self.sct.grab(self.monitor)
        img = np.array(screenshot)
        # 转换为RGB格式 (mss默认返回BGRA)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        return img
    
    def recognize_roadbook(self, frame):
        """识别屏幕上路书文本
        
        Args:
            frame: 游戏画面帧
            
        Returns:
            识别出的路书文本
        """
        try:
            # 假设路书在屏幕顶部特定区域
            # 这里需要根据实际游戏界面调整ROI坐标
            roi = frame[10:80, 800:1100]  # 示例坐标，需要调整
            
            # 预处理图像以提高OCR准确率
            gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # 使用Tesseract进行OCR识别
            custom_config = r'--oem 3 --psm 6'
            roadbook_text = pytesseract.image_to_string(thresh, config=custom_config, lang='chi_sim')
            
            return roadbook_text.strip()
        except Exception as e:
            print(f"路书识别错误: {e}")
            return ""
    
    def start_recording(self):
        """开始录制数据"""
        print("开始录制数据...")
        
        # 不启动键盘监听器，由外部统一处理按键事件
        
        self.running = True
        self.recording_data = []  # 清空之前的录制数据
        self.start_time = time.time()
        self.last_frame_time = self.start_time
        
        print(f"✅ 录制已开始，保存目录: {self.output_dir}")
        print(f"📝 录制配置: 帧率={self.fps}")
    
    def record_frame(self, frame=None, roadbook_text="", controls=None):
        """录制一帧数据，接收外部已经捕获的帧
        
        Args:
            frame: 外部捕获的帧数据
            roadbook_text: 路书文本
            controls: 控制指令
        """
        if not self.running:
            return
        
        current_time = time.time()
        
        # 控制录制帧率
        if current_time - self.last_frame_time >= self.frame_interval:
            # 使用外部提供的数据或默认值
            if frame is None:
                print("警告：没有提供帧数据，跳过录制")
                return
            
            if controls is None:
                controls = self.controls.copy()
            
            # 保存数据样本
            timestamp = current_time - self.start_time
            data_sample = {
                'timestamp': timestamp,
                'frame': frame,
                'roadbook_text': roadbook_text,
                'controls': controls
            }
            self.recording_data.append(data_sample)
            
            # 显示录制状态
            if len(self.recording_data) % 10 == 0:
                print(f"已录制 {len(self.recording_data)} 帧数据")
            
            self.last_frame_time = current_time
    
    def stop_recording(self):
        """停止录制并保存数据"""
        if not self.running:
            return
        
        print("停止录制，正在保存数据...")
        
        # 停止键盘监听器（如果存在）
        if hasattr(self, 'key_listener') and self.key_listener:
            self.key_listener.stop()
            self.key_listener = None
        
        self.running = False
        
        # 保存数据到HDF5文件
        if self.recording_data:
            self._save_data()
            print(f"✅ 数据已保存，共 {len(self.recording_data)} 条样本")
        else:
            print("❌ 没有录制到数据")
    
    def _save_data(self):
        """保存录制的数据到HDF5文件"""
        # 生成唯一的文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.output_dir}/recording_{timestamp}.h5"
        
        try:
            with h5py.File(filename, 'w') as hf:
                # 创建数据集
                frames = hf.create_dataset('frames', (len(self.recording_data),) + self.recording_data[0]['frame'].shape,
                                          dtype=np.uint8)
                timestamps = hf.create_dataset('timestamps', (len(self.recording_data),), dtype=np.float64)
                roadbook_texts = hf.create_dataset('roadbook_texts', (len(self.recording_data),), dtype=h5py.string_dtype())
                
                # 转向数据
                steering = hf.create_dataset('steering', (len(self.recording_data),), dtype=np.float32)
                throttle = hf.create_dataset('throttle', (len(self.recording_data),), dtype=np.float32)
                brake = hf.create_dataset('brake', (len(self.recording_data),), dtype=np.float32)
                
                # 填充数据
                for i, sample in enumerate(self.recording_data):
                    frames[i] = sample['frame']
                    timestamps[i] = sample['timestamp']
                    roadbook_texts[i] = sample['roadbook_text']
                    steering[i] = sample['controls']['steering']
                    throttle[i] = sample['controls']['throttle']
                    brake[i] = sample['controls']['brake']
                
                # 添加元数据
                hf.attrs['fps'] = self.fps
                hf.attrs['recording_time'] = self.recording_data[-1]['timestamp'] - self.recording_data[0]['timestamp']
                hf.attrs['sample_count'] = len(self.recording_data)
                hf.attrs['creation_time'] = timestamp
            
        except Exception as e:
            print(f"保存数据时出错: {e}")
            # 尝试保存为NPZ格式作为备选
            try:
                np_filename = f"{self.output_dir}/recording_{timestamp}.npz"
                frames = np.array([sample['frame'] for sample in self.recording_data])
                timestamps = np.array([sample['timestamp'] for sample in self.recording_data])
                roadbook_texts = np.array([sample['roadbook_text'] for sample in self.recording_data])
                steering = np.array([sample['controls']['steering'] for sample in self.recording_data])
                throttle = np.array([sample['controls']['throttle'] for sample in self.recording_data])
                brake = np.array([sample['controls']['brake'] for sample in self.recording_data])
                
                np.savez(np_filename, frames=frames, timestamps=timestamps, roadbook_texts=roadbook_texts,
                         steering=steering, throttle=throttle, brake=brake)
                print(f"已保存为NPZ格式: {np_filename}")
            except Exception as e2:
                print(f"保存为NPZ格式也失败: {e2}")

# 示例用法
if __name__ == "__main__":
    recorder = DataRecorder()
    recorder.start_recording()