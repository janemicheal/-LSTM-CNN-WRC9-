"""
数据预处理模块 (DataPreprocessor)

核心职责：清洗和标准化原始录制数据，提升训练效率。
"""
import os
import numpy as np
import cv2
import re
from datetime import datetime

class DataPreprocessor:
    """数据预处理器类"""
    
    def __init__(self, input_dir="./data/raw", output_dir="./data/processed"):
        """
        初始化数据预处理器
        
        Args:
            input_dir: 原始数据目录
            output_dir: 处理后数据保存目录
        """
        # 使用绝对路径，确保目录正确
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.input_dir = os.path.join(self.base_dir, input_dir)
        self.output_dir = os.path.join(self.base_dir, output_dir)
        
        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 注意：当前版本使用正则表达式进行路书解析，无需NLP模型
        
        # 赛道区域ROI坐标（需要根据游戏界面调整）
        self.track_roi = {
            'top': 0,      # 顶部边界
            'bottom': 1080,  # 底部边界
            'left': 0,       # 左侧边界
            'right': 1920    # 右侧边界
        }
    
    def preprocess_frame(self, frame):
        """
        预处理游戏画面帧
        
        Args:
            frame: 原始RGB图像
            
        Returns:
            预处理后的图像
        """
        # 裁剪ROI（仅保留赛道区域）
        roi_frame = frame[
            self.track_roi['top']:self.track_roi['bottom'],
            self.track_roi['left']:self.track_roi['right']
        ]
        
        # 调整大小以减少计算量
        target_size = (960, 540)  # 适合CNN输入的大小
        resized = cv2.resize(roi_frame, target_size, interpolation=cv2.INTER_AREA)
        
        # 转换为RGB（确保格式一致）
        if len(resized.shape) == 2:
            resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
        elif resized.shape[2] == 4:
            resized = cv2.cvtColor(resized, cv2.COLOR_RGBA2RGB)
        
        # 不立即归一化，保留uint8类型以节省内存
        # 注意：在模型预测前会进行归一化处理
        return resized
    
    def parse_roadbook_text(self, roadbook_text):
        """
        解析路书文本为结构化数据
        
        Args:
            roadbook_text: 原始路书文本，如"左4坡上50"
            
        Returns:
            结构化路书特征字典
        """
        roadbook_features = {
            'direction': 'straight',  # 方向：left, right, straight
            'degree': 0,             # 弯道度数：1-10
            'speed': 0,              # 建议速度
            'slope': 'flat',         # 坡度：up, down, flat
            'special': None          # 特殊路况：water, sand, etc.
        }
        
        # 使用正则表达式提取信息
        text = roadbook_text.lower()
        
        # 提取方向
        if '左' in text or 'left' in text:
            roadbook_features['direction'] = 'left'
        elif '右' in text or 'right' in text:
            roadbook_features['direction'] = 'right'
        
        # 提取弯道度数
        degree_match = re.search(r'(左|右)?\s*(\d+)', text)
        if degree_match:
            try:
                degree = int(degree_match.group(2))
                roadbook_features['degree'] = min(max(degree, 1), 10)  # 限制在1-10范围内
            except:
                pass
        
        # 提取速度
        speed_match = re.search(r'(\d+)', text)
        if speed_match:
            try:
                speed = int(speed_match.group(0))
                roadbook_features['speed'] = speed
            except:
                pass
        
        # 提取坡度
        if '坡上' in text or '上坡' in text or 'up' in text:
            roadbook_features['slope'] = 'up'
        elif '坡下' in text or '下坡' in text or 'down' in text:
            roadbook_features['slope'] = 'down'
        
        # 提取特殊路况
        if '水' in text or 'water' in text:
            roadbook_features['special'] = 'water'
        elif '沙' in text or 'sand' in text:
            roadbook_features['special'] = 'sand'
        elif '石' in text or 'stone' in text:
            roadbook_features['special'] = 'stone'
        
        return roadbook_features
    
    def normalize_controls(self, controls):
        """
        归一化操作指令
        
        Args:
            controls: 原始操作字典
            
        Returns:
            归一化后的操作字典
        """
        normalized = {}
        
        # 转向归一化到[-1, 1]
        if 'steering' in controls:
            normalized['steering'] = max(min(controls['steering'], 1.0), -1.0)
        else:
            normalized['steering'] = 0.0
        
        # 油门归一化到[0, 1]
        if 'throttle' in controls:
            normalized['throttle'] = max(min(controls['throttle'], 1.0), 0.0)
        else:
            normalized['throttle'] = 0.0
        
        # 刹车归一化到[0, 1]
        if 'brake' in controls:
            normalized['brake'] = max(min(controls['brake'], 1.0), 0.0)
        else:
            normalized['brake'] = 0.0
        
        # 确保油门和刹车不会同时生效
        if normalized['throttle'] > 0 and normalized['brake'] > 0:
            # 如果同时有油门和刹车，优先使用其中较大的值
            if normalized['throttle'] >= normalized['brake']:
                normalized['brake'] = 0.0
            else:
                normalized['throttle'] = 0.0
        
        return normalized
    
    def filter_abnormal_frames(self, frames, controls, threshold=0.5):
        """
        过滤异常帧
        
        Args:
            frames: 帧序列
            controls: 控制序列
            threshold: 异常检测阈值
            
        Returns:
            过滤后的帧和控制序列
        """
        filtered_frames = []
        filtered_controls = []
        
        # 批量处理以减少内存占用
        batch_size = 50
        
        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i:i+batch_size]
            batch_controls = controls[i:i+batch_size] if i < len(controls) else []
            
            for j in range(len(batch_frames)):
                frame = batch_frames[j]
                
                # 检查帧是否有效
                if frame is None or frame.size == 0:
                    continue
                
                # 检查uint8图像数据格式是否有效
                if isinstance(frame, np.ndarray):
                    # 对于uint8格式的图像，确保值在合理范围内
                    if frame.dtype == np.uint8:
                        # 检查图像是否有有效数据（不是全黑）
                        if frame.max() == 0:
                            continue
                
                # 检查控制指令是否异常
                if j < len(batch_controls):
                    ctrl = batch_controls[j]
                    # 检查控制值是否在合理范围内
                    steering = ctrl.get('steering', 0.0)
                    throttle = ctrl.get('throttle', 0.0)
                    brake = ctrl.get('brake', 0.0)
                    
                    # 控制值范围检查
                    if abs(steering) > 1.0 or throttle < 0.0 or throttle > 1.0 or brake < 0.0 or brake > 1.0:
                        continue
                    
                    # 检查是否同时大油门和大刹车
                    if throttle > threshold and brake > threshold:
                        continue
                
                filtered_frames.append(frame)
                filtered_controls.append(batch_controls[j] if j < len(batch_controls) else controls[i+j])
            
            # 每处理完一个批次后尝试释放内存
            import gc
            gc.collect()
        
        return filtered_frames, filtered_controls
    
    def data_augmentation(self, frame, steering):
        """
        数据增强
        
        Args:
            frame: 图像帧
            steering: 转向角度
            
        Returns:
            增强后的数据列表 [(frame1, steering1), (frame2, steering2), ...]
        """
        augmented_data = [(frame, steering)]
        
        # 水平翻转（模拟反向赛道）
        flipped_frame = cv2.flip(frame, 1)
        flipped_steering = -steering
        augmented_data.append((flipped_frame, flipped_steering))
        
        # 亮度调整
        # 增加亮度
        hsv = cv2.cvtColor((frame * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
        hsv[:, :, 2] = np.minimum(hsv[:, :, 2] * 1.2, 255)
        bright_frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0
        augmented_data.append((bright_frame, steering))
        
        # 降低亮度
        hsv[:, :, 2] = np.maximum(hsv[:, :, 2] * 0.8, 0)
        dark_frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0
        augmented_data.append((dark_frame, steering))
        
        return augmented_data
    
    def load_h5_data(self, file_path):
        """
        加载HDF5格式的原始数据
        
        Args:
            file_path: 文件路径
            
        Returns:
            数据字典
        """
        import h5py
        
        try:
            with h5py.File(file_path, 'r') as hf:
                data = {
                    'frames': np.array(hf['frames']),
                    'timestamps': np.array(hf['timestamps']),
                    'roadbook_texts': np.array(hf['roadbook_texts']),
                    'steering': np.array(hf['steering']),
                    'throttle': np.array(hf['throttle']),
                    'brake': np.array(hf['brake'])
                }
            return data
        except Exception as e:
            print(f"加载HDF5文件错误: {e}")
            return None
    
    def load_npz_data(self, file_path):
        """
        加载NPZ格式的原始数据
        
        Args:
            file_path: 文件路径
            
        Returns:
            数据字典
        """
        try:
            with np.load(file_path) as npz:
                # 检查NPZ文件的键，适配SimpleRecorder生成的格式
                keys = list(npz.keys())
                
                # 适配不同格式的NPZ文件
                if 'roadbook_infos' in keys:
                    # SimpleRecorder生成的格式
                    data = {
                        'frames': npz['frames'],
                        'timestamps': npz['timestamps'],
                        'roadbook_texts': npz['roadbook_infos'],  # 映射roadbook_infos到roadbook_texts
                        'steering': npz['steerings'],  # 映射steerings到steering
                        'throttle': npz['throttles'],  # 映射throttles到throttle
                        'brake': npz['brakes']  # 映射brakes到brake
                    }
                else:
                    # 原始格式
                    data = {
                        'frames': npz['frames'],
                        'timestamps': npz['timestamps'],
                        'roadbook_texts': npz['roadbook_texts'],
                        'steering': npz['steering'],
                        'throttle': npz['throttle'],
                        'brake': npz['brake']
                    }
            return data
        except Exception as e:
            print(f"加载NPZ文件错误: {e}")
            return None
    
    def process_file(self, file_path):
        """
        处理单个数据文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            处理后的数据保存路径
        """
        print(f"开始处理文件: {file_path}")
        
        # 根据文件扩展名选择加载方法
        if file_path.endswith('.h5') or file_path.endswith('.hdf5'):
            data = self.load_h5_data(file_path)
        elif file_path.endswith('.npz'):
            data = self.load_npz_data(file_path)
        else:
            print(f"不支持的文件格式: {file_path}")
            return None
        
        if data is None:
            return None
        
        # 预处理数据
        processed_data = []
        
        for i in range(len(data['frames'])):
            # 预处理画面
            frame = data['frames'][i]
            preprocessed_frame = self.preprocess_frame(frame)
            
            # 解析路书
            roadbook_text = data['roadbook_texts'][i]
            roadbook_features = self.parse_roadbook_text(roadbook_text)
            
            # 准备控制指令
            raw_controls = {
                'steering': data['steering'][i],
                'throttle': data['throttle'][i],
                'brake': data['brake'][i]
            }
            normalized_controls = self.normalize_controls(raw_controls)
            
            # 构建样本
            sample = {
                'frame': preprocessed_frame,
                'roadbook': roadbook_features,
                'controls': normalized_controls,
                'timestamp': data['timestamps'][i]
            }
            
            processed_data.append(sample)
        
        # 过滤异常数据
        frames = [s['frame'] for s in processed_data]
        controls = [s['controls'] for s in processed_data]
        filtered_frames, filtered_controls = self.filter_abnormal_frames(frames, controls)
        
        # 重新构建处理后的数据
        filtered_data = []
        for frame, controls in zip(filtered_frames, filtered_controls):
            filtered_data.append({
                'frame': frame,
                'roadbook': roadbook_features,  # 这里简化处理，实际应该对应到每个样本
                'controls': controls
            })
        
        # 保存处理后的数据
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"processed_{os.path.basename(file_path).split('.')[0]}_{timestamp}.npz"
        output_path = os.path.join(self.output_dir, filename)
        
        try:
            # 准备保存数据
            save_frames = np.array([s['frame'] for s in filtered_data])
            save_roadbooks = np.array([s['roadbook'] for s in filtered_data], dtype=object)
            save_steering = np.array([s['controls']['steering'] for s in filtered_data])
            save_throttle = np.array([s['controls']['throttle'] for s in filtered_data])
            save_brake = np.array([s['controls']['brake'] for s in filtered_data])
            
            np.savez_compressed(output_path, 
                               frames=save_frames,
                               roadbooks=save_roadbooks,
                               steering=save_steering,
                               throttle=save_throttle,
                               brake=save_brake)
            
            print(f"处理完成，保存到: {output_path}")
            print(f"原始样本数: {len(processed_data)}, 过滤后样本数: {len(filtered_data)}")
            
            return output_path
        except Exception as e:
            print(f"保存处理后数据时出错: {e}")
            return None
    
    def process_directory(self):
        """
        处理目录中所有数据文件和视频文件
        
        Returns:
            处理后的文件列表
        """
        processed_files = []
        
        if not os.path.exists(self.input_dir):
            print(f"错误: 目录不存在: {self.input_dir}")
            return processed_files
            
        print(f"开始处理目录: {self.input_dir}")
        
        # 获取目录中的所有文件
        files = os.listdir(self.input_dir)
        print(f"目录中的文件数量: {len(files)}")
        
        # 支持的数据文件扩展名
        data_extensions = ('.h5', '.hdf5', '.npz')
        # 支持的视频文件扩展名
        video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.wmv')
        
        for filename in files:
            file_path = os.path.join(self.input_dir, filename)
            
            # 检查是否为文件（非目录）
            if os.path.isfile(file_path):
                # 根据文件扩展名选择处理方法
                if filename.lower().endswith(data_extensions):
                    print(f"处理数据文件: {filename}")
                    output_path = self.process_file(file_path)
                    if output_path:
                        processed_files.append(output_path)
                elif filename.lower().endswith(video_extensions):
                    print(f"处理视频文件: {filename}")
                    output_path = self.process_single_video(file_path)
                    if output_path:
                        processed_files.append(output_path)
                else:
                    print(f"跳过不支持的文件格式: {filename}")
            else:
                print(f"跳过子目录: {filename}")
        
        print(f"目录处理完成，成功处理的文件数: {len(processed_files)}")
        return processed_files
    
    def process_single_video(self, video_path):
        """
        处理单个视频文件
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            处理后的数据保存路径，失败返回None
        """
        if not os.path.exists(video_path):
            print(f"错误: 视频文件不存在: {video_path}")
            return None
        
        print(f"开始处理视频文件: {video_path}")
        
        try:
            # 创建一个临时的视频处理函数
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"错误: 无法打开视频文件: {video_path}")
                return None
            
            processed_data = []
            frame_count = 0
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            # 设置目标帧率为15帧/秒
            target_fps = 15
            # 计算采样间隔
            if fps > 0:
                sample_interval = int(fps / target_fps)
                # 确保至少采样1帧
                sample_interval = max(1, sample_interval)
            else:
                sample_interval = 1
            
            print(f"视频原始帧率: {fps}, 目标帧率: {target_fps}, 采样间隔: {sample_interval}")
            
            # 添加批量处理以减少内存占用
            batch_size = 30  # 每批处理的帧数
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # 根据采样间隔选择帧，只处理指定间隔的帧
                if frame_count % sample_interval == 0:
                    try:
                        # 预处理画面
                        # 注意：视频帧通常是BGR格式，需要转换为RGB
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        preprocessed_frame = self.preprocess_frame(rgb_frame)
                        
                        # 为演示目的创建默认控制指令
                        # 实际应用中可能需要从其他来源获取控制数据
                        normalized_controls = {
                            'steering': 0.0,
                            'throttle': 0.5,
                            'brake': 0.0
                        }
                        
                        # 构建样本 - 不再包含路书信息
                        sample = {
                            'frame': preprocessed_frame,
                            'controls': normalized_controls,
                            'timestamp': frame_count / fps if fps > 0 else frame_count
                        }
                        
                        processed_data.append(sample)
                        
                        # 每处理一定数量的帧显示进度并释放一些内存
                        if len(processed_data) % batch_size == 0:
                            print(f"处理中: {len(processed_data)} 帧")
                            
                    except MemoryError:
                        print("内存不足错误：跳过当前帧继续处理")
                        # 清理当前帧数据
                        if 'rgb_frame' in locals():
                            del rgb_frame
                        if 'preprocessed_frame' in locals():
                            del preprocessed_frame
                        # 尝试释放一些内存
                        import gc
                        gc.collect()
                
                frame_count += 1
            
            cap.release()
            
            print(f"视频处理完成，原始总帧数: {frame_count}, 采样后帧数: {len(processed_data)}")
            
            # 分批过滤异常数据以减少内存占用
            batch_size = 30
            filtered_data = []
            
            print(f"开始过滤异常数据，总帧数: {len(processed_data)}")
            
            # 清空原始processed_data列表以释放内存
            frames = [s['frame'] for s in processed_data]
            controls = [s['controls'] for s in processed_data]
            
            # 释放processed_data内存
            del processed_data
            import gc
            gc.collect()
            
            # 批量过滤数据
            for i in range(0, len(frames), batch_size):
                batch_frames = frames[i:i+batch_size]
                batch_controls = controls[i:i+batch_size] if i < len(controls) else []
                
                # 过滤当前批次
                batch_filtered_frames, batch_filtered_controls = self.filter_abnormal_frames(batch_frames, batch_controls)
                
                # 添加到结果列表
                for frame, control in zip(batch_filtered_frames, batch_filtered_controls):
                    filtered_data.append({
                        'frame': frame,
                        'controls': control
                    })
                
                # 释放当前批次内存
                del batch_frames
                del batch_controls
                gc.collect()
                
                # 显示进度
                if (i + batch_size) % (batch_size * 2) == 0:
                    print(f"已过滤 {min(i + batch_size, len(frames))} 帧")
            
            # 释放原始frames和controls内存
            del frames
            del controls
            gc.collect()
            
            print(f"过滤完成，剩余 {len(filtered_data)} 帧有效数据")
            
            # 保存处理后的数据
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"processed_{os.path.basename(video_path).split('.')[0]}_{timestamp}.npz"
            output_path = os.path.join(self.output_dir, filename)
            
            try:
                # 内存优化：分块保存数据
                print(f"开始保存数据，总计 {len(filtered_data)} 帧")
                
                # 对于大型数据集，使用更内存高效的保存方式
                if len(filtered_data) > 100:
                    # 创建临时字典存储数据
                    save_data = {'frames': [], 'steering': [], 'throttle': [], 'brake': []}
                    
                    # 分批处理数据
                    batch_size = 20
                    for i in range(0, len(filtered_data), batch_size):
                        batch_data = filtered_data[i:i+batch_size]
                        
                        # 添加批次数据
                        for d in batch_data:
                            save_data['frames'].append(d['frame'])
                            save_data['steering'].append(d['controls']['steering'])
                            save_data['throttle'].append(d['controls']['throttle'])
                            save_data['brake'].append(d['controls']['brake'])
                        
                        # 显示进度
                        print(f"处理保存批次: {i+batch_size}/{len(filtered_data)}")
                        
                        # 每3个批次释放一次内存
                        if (i // batch_size) % 3 == 0:
                            import gc
                            gc.collect()
                    
                    # 转换为numpy数组进行保存
                    save_frames = np.array(save_data['frames'])
                    save_steering = np.array(save_data['steering'])
                    save_throttle = np.array(save_data['throttle'])
                    save_brake = np.array(save_data['brake'])
                    
                    # 释放临时数据内存
                    del save_data
                    gc.collect()
                else:
                    # 对于小型数据集，可以直接转换
                    save_frames = np.array([s['frame'] for s in filtered_data])
                    save_steering = np.array([s['controls']['steering'] for s in filtered_data])
                    save_throttle = np.array([s['controls']['throttle'] for s in filtered_data])
                    save_brake = np.array([s['controls']['brake'] for s in filtered_data])
                
                # 不再保存路书信息
                print(f"正在保存数据到 {output_path}...")
                np.savez_compressed(output_path, 
                                   frames=save_frames,
                                   steering=save_steering,
                                   throttle=save_throttle,
                                   brake=save_brake)
                
                # 释放数组内存
                del save_frames
                del save_steering
                del save_throttle
                del save_brake
                gc.collect()
                
                print(f"处理完成，保存到: {output_path}")
                print(f"采样后样本数: {len(processed_data)}, 过滤后样本数: {len(filtered_data)}")
                
                return output_path
            except Exception as e:
                print(f"保存处理后数据时出错: {e}")
                return None
                
        except Exception as e:
            print(f"处理视频文件时出错: {e}")
            return None

# 示例用法
if __name__ == "__main__":
    # 使用类初始化时的默认路径，处理录制的数据
    input_dir = "./data/raw"
    
    # 创建预处理器实例
    preprocessor = DataPreprocessor(input_dir=input_dir)
    
    # 根据输入类型调用不同的处理方法
    if os.path.isdir(preprocessor.input_dir):
        # 如果是目录，处理目录中的所有文件
        preprocessor.process_directory()
    else:
        # 如果是单个文件，处理单个文件
        preprocessor.process_single_video(preprocessor.input_dir)