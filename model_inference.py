"""
模型推理模块 (ModelInference)

核心职责：加载训练好的模型，输入实时画面 + 路书特征，预测最优操作指令。
"""
import os
import time
import numpy as np
import torch
import cv2
import logging
from typing import Dict, Optional, Tuple, List
from datetime import datetime

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='../logs/model_inference.log'
)
logger = logging.getLogger('ModelInference')

# 导入模型类（从model_trainer模块）
from .model_trainer import DrivingModel

class ModelInference:
    """
    模型推理类，负责加载训练好的模型并进行实时预测
    """
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化模型推理模块
        
        Args:
            config: 配置字典
        """
        # 默认配置
        self.default_config = {
            'model_path': './models/best_model.pth',  # 模型文件路径，使用相对路径指向当前目录下的models文件夹
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',  # 运行设备
            'use_lstm': False,  # 是否使用LSTM模型
            'target_resolution': (960, 540),  # 输入分辨率，与训练和捕获一致
            'enable_preprocessing': True,  # 是否启用预处理
            'enable_smoothing': True,  # 是否启用输出平滑
            'smoothing_window': 2,  # 减少平滑窗口大小，使控制更灵敏
            'steering_sensitivity': 1.2,  # 提高转向灵敏度
            'throttle_sensitivity': 1.5,  # 提高油门灵敏度，使车辆更容易加速
            'brake_sensitivity': 1.0,  # 刹车灵敏度
            'min_throttle': 0.3,  # 提高最小油门值，确保车辆能持续前进
            'max_throttle': 1.0,  # 最大油门值
            'min_steering': -1.0,  # 最小转向值
            'max_steering': 1.0,  # 最大转向值
            'debug': True  # 启用调试模式，方便诊断问题
        }
        
        # 合并配置
        self.config = self.default_config.copy()
        if config:
            self.config.update(config)
        
        # 设置设备
        self.device = torch.device(self.config['device'])
        logger.info(f"使用设备: {self.device}")
        
        # 初始化模型
        self.model = None
        self.model_loaded = False
        
        # 初始化推理相关变量
        self.steering_history = []
        self.throttle_history = []
        self.brake_history = []
        self.lstm_hidden = None
        self.inference_times = []
        self.last_prediction = None
        self.last_prediction_time = 0
        
        # 加载模型
        self.load_model()
    
    def load_model(self) -> bool:
        """
        加载模型权重
        
        Returns:
            是否成功加载模型
        """
        try:
            if not os.path.exists(self.config['model_path']):
                logger.error(f"模型文件不存在: {self.config['model_path']}")
                print(f"[ERROR] 模型文件不存在: {self.config['model_path']}")
                return False
            
            logger.info(f"正在加载模型: {self.config['model_path']}")
            print(f"正在加载模型: {self.config['model_path']}")
            start_time = time.time()
            
            # 创建模型实例
            self.model = DrivingModel(use_lstm=self.config['use_lstm']).to(self.device)
            
            # 加载权重，添加weights_only=True参数以提高安全性
            checkpoint = torch.load(self.config['model_path'], map_location=self.device, weights_only=True)
            
            # 检查是否是完整的checkpoint还是仅模型权重
            if 'model_state_dict' in checkpoint:
                # 完整的checkpoint
                self.model.load_state_dict(checkpoint['model_state_dict'])
                
                # 恢复配置
                if 'config' in checkpoint:
                    # 更新与模型相关的配置
                    model_config = checkpoint['config']
                    self.config['use_lstm'] = model_config.get('use_lstm', self.config['use_lstm'])
                    logger.info(f"从checkpoint恢复配置: use_lstm={self.config['use_lstm']}")
                    print(f"从checkpoint恢复配置: use_lstm={self.config['use_lstm']}")
            else:
                # 仅模型权重
                self.model.load_state_dict(checkpoint)
            
            # 设置为评估模式
            self.model.eval()
            
            load_time = time.time() - start_time
            logger.info(f"模型加载完成，耗时: {load_time:.2f}秒")
            print(f"[INFO] 模型加载完成，耗时: {load_time:.2f}秒")
            
            self.model_loaded = True
            return True
            
        except Exception as e:
            logger.error(f"加载模型时出错: {e}")
            print(f"[ERROR] 加载模型时出错: {e}")
            self.model_loaded = False
            return False
    
    def _preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """
        预处理输入帧
        
        Args:
            frame: 输入图像帧
            
        Returns:
            预处理后的张量
        """
        try:
            # 调整大小
            resized = cv2.resize(frame, self.config['target_resolution'], interpolation=cv2.INTER_AREA)
            
            # 转换为RGB（如果需要）
            if len(resized.shape) == 3 and resized.shape[2] == 4:
                resized = resized[:, :, :3]  # 移除alpha通道
            elif len(resized.shape) == 2:
                resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)  # 灰度转RGB
            
            # 转置为CxHxW格式
            resized = resized.transpose(2, 0, 1)
            
            # 归一化
            normalized = resized / 255.0
            normalized = (normalized - np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)) / np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
            
            # 转换为张量
            tensor = torch.from_numpy(normalized).float()
            
            # 添加批次维度
            tensor = tensor.unsqueeze(0)
            
            return tensor.to(self.device)
            
        except Exception as e:
            logger.error(f"预处理帧时出错: {e}")
            raise
    
    def _prepare_roadbook_features(self, roadbook: Dict) -> torch.Tensor:
        """
        准备路书特征为模型输入格式
        
        Args:
            roadbook: 路书数据字典
            
        Returns:
            路书特征张量
        """
        try:
            # 确保roadbook是字典，避免类型错误
            if not isinstance(roadbook, dict):
                roadbook = {}
            
            # 将路书特征转换为数值向量，与训练时保持一致
            direction_map = {'left': -1.0, 'straight': 0.0, 'right': 1.0}
            slope_map = {'down': -1.0, 'flat': 0.0, 'up': 1.0}
            
            # 提取基础特征，使用默认值处理缺失情况
            direction = direction_map.get(roadbook.get('direction', 'straight'), 0.0)
            degree = roadbook.get('degree', 0) / 10.0  # 归一化到0-1
            speed = roadbook.get('speed', 0) / 200.0  # 假设最大速度200
            slope = slope_map.get(roadbook.get('slope', 'flat'), 0.0)
            
            # 特殊路况特征
            special = roadbook.get('special', None)
            has_special = 1.0 if special else 0.0
            
            # 组合特征向量
            feature_vec = np.array([direction, degree, speed, slope, has_special], dtype=np.float32)
            
            # 转换为张量并添加批次维度
            tensor = torch.from_numpy(feature_vec).float().unsqueeze(0)
            
            # 添加调试信息
            if self.config['debug']:
                print(f"[DEBUG] 路书特征: 方向={direction:.2f}, 度数={degree:.2f}, 速度={speed:.2f}, 坡度={slope:.2f}, 特殊路况={has_special:.2f}")
            
            return tensor.to(self.device)
            
        except Exception as e:
            logger.error(f"准备路书特征时出错: {e}")
            # 返回默认特征，确保模型能继续运行
            default_tensor = torch.zeros(1, 5).to(self.device)
            if self.config['debug']:
                print(f"[WARNING] 使用默认路书特征，错误: {e}")
            return default_tensor
    
    def _smooth_output(self, steering: float, throttle: float, brake: float) -> Tuple[float, float, float]:
        """
        平滑输出值
        
        Args:
            steering: 原始转向值
            throttle: 原始油门值
            brake: 原始刹车值
            
        Returns:
            平滑后的转向、油门、刹车值
        """
        # 添加到历史记录
        self.steering_history.append(steering)
        self.throttle_history.append(throttle)
        self.brake_history.append(brake)
        
        # 限制历史记录长度
        if len(self.steering_history) > self.config['smoothing_window']:
            self.steering_history.pop(0)
        if len(self.throttle_history) > self.config['smoothing_window']:
            self.throttle_history.pop(0)
        if len(self.brake_history) > self.config['smoothing_window']:
            self.brake_history.pop(0)
        
        # 计算平滑值
        smoothed_steering = sum(self.steering_history) / len(self.steering_history)
        smoothed_throttle = sum(self.throttle_history) / len(self.throttle_history)
        smoothed_brake = sum(self.brake_history) / len(self.brake_history)
        
        return smoothed_steering, smoothed_throttle, smoothed_brake
    
    def _apply_sensitivity(self, steering: float, throttle: float, brake: float) -> Tuple[float, float, float]:
        """
        应用灵敏度调整
        
        Args:
            steering: 转向值
            throttle: 油门值
            brake: 刹车值
            
        Returns:
            调整后的控制值
        """
        # 应用灵敏度
        steering = steering * self.config['steering_sensitivity']
        throttle = throttle * self.config['throttle_sensitivity']
        brake = brake * self.config['brake_sensitivity']
        
        # 限制范围
        steering = max(self.config['min_steering'], min(self.config['max_steering'], steering))
        throttle = max(self.config['min_throttle'], min(self.config['max_throttle'], throttle))
        brake = max(0.0, min(1.0, brake))
        
        return steering, throttle, brake
    
    def _clip_controls(self, steering: float, throttle: float, brake: float) -> Tuple[float, float, float]:
        """
        裁剪控制值到有效范围
        
        Args:
            steering: 转向值
            throttle: 油门值
            brake: 刹车值
            
        Returns:
            裁剪后的控制值
        """
        steering = np.clip(steering, -1.0, 1.0)
        throttle = np.clip(throttle, 0.0, 1.0)
        brake = np.clip(brake, 0.0, 1.0)
        
        return steering, throttle, brake
    
    def predict(self, frame: np.ndarray, roadbook: Dict) -> Tuple[Optional[Dict], float]:
        """
        进行预测
        
        Args:
            frame: 输入图像帧
            roadbook: 路书数据
            
        Returns:
            预测的控制指令和推理时间
        """
        # 检查模型是否已加载
        if not self.model_loaded or self.model is None:
            logger.error("模型未加载，无法进行预测")
            print("[ERROR] 模型未加载，无法进行预测")
            return None, 0.0
        
        try:
            start_time = time.time()
            
            # 预处理输入
            with torch.no_grad():
                # 预处理帧
                frame_tensor = self._preprocess_frame(frame)
                
                # 准备路书特征
                roadbook_tensor = self._prepare_roadbook_features(roadbook)
                
                # 前向传播
                if self.config['use_lstm']:
                    outputs, self.lstm_hidden = self.model(frame_tensor, roadbook_tensor, self.lstm_hidden)
                    # 分离隐藏状态以避免梯度计算
                    self.lstm_hidden = tuple(h.detach() for h in self.lstm_hidden)
                else:
                    outputs = self.model(frame_tensor, roadbook_tensor)
                
                # 获取预测结果
                steering = outputs[0, 0].item()
                throttle = outputs[0, 1].item()
                brake = outputs[0, 2].item()
            
            # 计算推理时间
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)
            
            # 限制推理时间历史记录长度
            if len(self.inference_times) > 100:
                self.inference_times.pop(0)
            
            # 应用灵敏度调整
            steering, throttle, brake = self._apply_sensitivity(steering, throttle, brake)
            
            # 裁剪控制值
            steering, throttle, brake = self._clip_controls(steering, throttle, brake)
            
            # 平滑输出（如果启用）
            if self.config['enable_smoothing']:
                steering, throttle, brake = self._smooth_output(steering, throttle, brake)
            
            # 构建结果
            controls = {
                'steering': steering,
                'throttle': throttle,
                'brake': brake,
                'inference_time_ms': inference_time * 1000,
                'timestamp': time.time()
            }
            
            # 更新最后预测
            self.last_prediction = controls
            self.last_prediction_time = time.time()
            
            # 增强调试信息
            if self.config['debug']:
                logger.debug(f"预测结果: 转向={steering:.3f}, 油门={throttle:.3f}, 刹车={brake:.3f}, 推理时间={inference_time*1000:.2f}ms")
                print(f"[DEBUG] 模型预测: 转向={steering:.3f}, 油门={throttle:.3f}, 刹车={brake:.3f}, 推理时间={inference_time*1000:.2f}ms")
            
            return controls, inference_time
            
        except Exception as e:
            logger.error(f"预测时出错: {e}")
            print(f"[ERROR] 模型预测出错: {e}")
            return None, 0.0
    
    def get_last_prediction(self) -> Optional[Dict]:
        """
        获取最后一次预测结果
        
        Returns:
            最后一次预测的控制指令
        """
        return self.last_prediction
    
    def get_average_inference_time(self) -> float:
        """
        获取平均推理时间
        
        Returns:
            平均推理时间（毫秒）
        """
        if not self.inference_times:
            return 0.0
        
        avg_time = sum(self.inference_times) / len(self.inference_times)
        return avg_time * 1000  # 转换为毫秒
    
    def get_fps(self) -> float:
        """
        获取当前FPS
        
        Returns:
            当前FPS值
        """
        if not self.inference_times:
            return 0.0
        
        avg_time = sum(self.inference_times) / len(self.inference_times)
        return 1.0 / avg_time if avg_time > 0 else 0.0
    
    def reset(self):
        """
        重置推理状态
        """
        # 清空历史记录
        self.steering_history = []
        self.throttle_history = []
        self.brake_history = []
        self.inference_times = []
        
        # 重置LSTM隐藏状态
        self.lstm_hidden = None
        
        # 重置最后预测
        self.last_prediction = None
        self.last_prediction_time = 0
        
        logger.info("推理状态已重置")
    
    def update_config(self, new_config: Dict):
        """
        更新配置
        
        Args:
            new_config: 新的配置字典
        """
        self.config.update(new_config)
        logger.info(f"配置已更新: {new_config}")
        
        # 如果更新了设备，重新加载模型
        if 'device' in new_config:
            self.device = torch.device(new_config['device'])
            logger.info(f"设备已切换到: {self.device}")
            if self.model_loaded:
                self.model.to(self.device)
                logger.info("模型已移动到新设备")
    
    def load_new_model(self, model_path: str) -> bool:
        """
        加载新模型
        
        Args:
            model_path: 新模型路径
            
        Returns:
            是否成功加载
        """
        self.config['model_path'] = model_path
        return self.load_model()
    
    def export_to_onnx(self, output_path: str) -> bool:
        """
        导出模型为ONNX格式（可选的优化功能）
        
        Args:
            output_path: 输出ONNX文件路径
            
        Returns:
            是否成功导出
        """
        if not self.model_loaded:
            logger.error("模型未加载，无法导出")
            return False
        
        try:
            # 创建示例输入
            dummy_frame = torch.randn(1, 3, self.config['target_resolution'][1], self.config['target_resolution'][0]).to(self.device)
            dummy_roadbook = torch.randn(1, 5).to(self.device)
            
            # 导出为ONNX
            torch.onnx.export(
                self.model,
                (dummy_frame, dummy_roadbook),
                output_path,
                export_params=True,
                opset_version=12,
                do_constant_folding=True,
                input_names=['frame', 'roadbook'],
                output_names=['controls'],
                dynamic_axes={
                    'frame': {0: 'batch_size'},
                    'roadbook': {0: 'batch_size'},
                    'controls': {0: 'batch_size'}
                }
            )
            
            logger.info(f"模型已成功导出为ONNX格式: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"导出ONNX模型时出错: {e}")
            return False

class ModelOptimizer:
    """
    模型优化器类，提供模型优化功能（可选）
    """
    @staticmethod
    def optimize_model(model_path: str, output_path: str, device: str = 'cuda') -> bool:
        """
        优化模型（简化版）
        
        Args:
            model_path: 原始模型路径
            output_path: 输出优化模型路径
            device: 目标设备
            
        Returns:
            是否成功优化
        """
        try:
            # 这是一个简化版本，实际应用中可以使用更多优化技术
            # 例如：量化、剪枝、ONNX Runtime等
            logger.info(f"开始优化模型: {model_path}")
            
            # 加载模型
            device_type = torch.device(device)
            model = DrivingModel()
            checkpoint = torch.load(model_path, map_location=device_type)
            
            # 提取模型权重
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            # 设置为评估模式
            model.eval()
            
            # 保存优化后的模型（仅权重）
            torch.save(model.state_dict(), output_path)
            
            logger.info(f"模型优化完成，保存到: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"优化模型时出错: {e}")
            return False

# 示例用法
def example_usage():
    # 创建推理实例
    inference_config = {
        'model_path': '../models/best_model.pth',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'enable_smoothing': True,
        'smoothing_window': 3
    }
    
    inference = ModelInference(inference_config)
    
    if not inference.model_loaded:
        print("无法加载模型")
        return
    
    # 模拟输入
    import numpy as np
    dummy_frame = np.random.randint(0, 255, (450, 800, 3), dtype=np.uint8)
    dummy_roadbook = {
        'direction': 'left',
        'degree': 4,
        'speed': 50,
        'slope': 'flat',
        'special': None
    }
    
    # 进行预测
    controls, inference_time = inference.predict(dummy_frame, dummy_roadbook)
    
    if controls:
        print(f"预测结果:")
        print(f"- 转向: {controls['steering']:.2f}")
        print(f"- 油门: {controls['throttle']:.2f}")
        print(f"- 刹车: {controls['brake']:.2f}")
        print(f"推理时间: {inference_time*1000:.2f}ms")

if __name__ == "__main__":
    example_usage()