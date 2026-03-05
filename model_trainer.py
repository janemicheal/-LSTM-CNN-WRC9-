"""
模型训练模块 (ModelTrainer)

核心职责：基于标注数据训练端到端自动驾驶模型。
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import matplotlib.pyplot as plt
from datetime import datetime
import json
import logging
import tqdm
import gc

# 设置日志
# 确保日志目录存在
import os
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename=os.path.join(log_dir, 'model_training.log')
)
logger = logging.getLogger('model_trainer')

class DrivingDataset(Dataset):
    """
    驾驶数据集类
    """
    def __init__(self, data_file, transform=None):
        """
        初始化数据集
        
        Args:
            data_file: 数据文件路径
            transform: 数据变换
        """
        try:
            with np.load(data_file, allow_pickle=True) as npz:
                self.frames = npz['frames']
                
                # 检查NPZ文件的键，适配不同格式
                keys = list(npz.keys())
                
                if 'roadbooks' in keys:
                    # 预处理后的数据格式
                    self.roadbooks = npz['roadbooks']
                else:
                    # 直接使用空路书列表
                    self.roadbooks = np.array([{} for _ in range(len(self.frames))])
                
                # 适配不同的控制指令字段名
                if 'steering' in keys:
                    self.steering = npz['steering'].astype(np.float32)
                    self.throttle = npz['throttle'].astype(np.float32)
                    self.brake = npz['brake'].astype(np.float32)
                elif 'steerings' in keys:
                    # SimpleRecorder生成的格式
                    self.steering = npz['steerings'].astype(np.float32)
                    self.throttle = npz['throttles'].astype(np.float32)
                    self.brake = npz['brakes'].astype(np.float32)
                else:
                    # 默认值
                    self.steering = np.zeros(len(self.frames), dtype=np.float32)
                    self.throttle = np.zeros(len(self.frames), dtype=np.float32)
                    self.brake = np.zeros(len(self.frames), dtype=np.float32)
                
                # 检查是否有赛道特征
                self.track_features = npz.get('track_features', np.array([{} for _ in range(len(self.frames))]))
        except Exception as e:
            logger.error(f"加载数据集失败: {e}")
            raise
        
        self.transform = transform
        self._prepare_roadbook_features()
    
    def _prepare_roadbook_features(self):
        """
        准备路书特征为数值向量
        """
        self.roadbook_features = []
        
        for roadbook in self.roadbooks:
            # 将路书特征转换为数值向量
            direction_map = {'left': -1.0, 'straight': 0.0, 'right': 1.0}
            slope_map = {'down': -1.0, 'flat': 0.0, 'up': 1.0}
            
            # 提取基础特征
            direction = direction_map.get(roadbook.get('direction', 'straight'), 0.0)
            degree = roadbook.get('degree', 0) / 10.0  # 归一化到0-1
            speed = roadbook.get('speed', 0) / 200.0  # 假设最大速度200
            slope = slope_map.get(roadbook.get('slope', 'flat'), 0.0)
            
            # 特殊路况特征
            special = roadbook.get('special', None)
            has_special = 1.0 if special else 0.0
            
            # 组合特征向量
            feature_vec = np.array([direction, degree, speed, slope, has_special], dtype=np.float32)
            self.roadbook_features.append(feature_vec)
        
        self.roadbook_features = np.array(self.roadbook_features)
    
    def __len__(self):
        return len(self.frames)
    
    def __getitem__(self, idx):
        frame = self.frames[idx]
        roadbook_feature = self.roadbook_features[idx]
        steering = self.steering[idx]
        throttle = self.throttle[idx]
        brake = self.brake[idx]
        
        # 应用数据变换
        if self.transform:
            frame = self.transform(frame)
        
        # 确保frame是正确的形状 [C, H, W]
        if isinstance(frame, np.ndarray):
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                frame = frame.transpose(2, 0, 1)  # HWC -> CHW
        
        return {
            'frame': torch.tensor(frame, dtype=torch.float32),
            'roadbook_feature': torch.tensor(roadbook_feature, dtype=torch.float32),
            'steering': torch.tensor(steering, dtype=torch.float32),
            'throttle': torch.tensor(throttle, dtype=torch.float32),
            'brake': torch.tensor(brake, dtype=torch.float32)
        }

class DrivingModel(nn.Module):
    """
    端到端自动驾驶模型
    """
    def __init__(self, use_lstm=False):
        """
        初始化模型
        
        Args:
            use_lstm: 是否使用LSTM处理时序特征
        """
        super(DrivingModel, self).__init__()
        self.use_lstm = use_lstm
        
        # 使用预训练的MobileNetV2作为CNN特征提取器
        self.cnn_backbone = models.mobilenet_v2(pretrained=True)
        
        # 替换最后的分类层为特征提取层
        self.cnn_features = nn.Sequential(*list(self.cnn_backbone.children())[:-1])
        
        # 获取CNN输出特征维度
        cnn_output_size = 1280  # MobileNetV2最后特征图的通道数
        
        # 路书特征处理
        roadbook_input_size = 5  # 我们的路书特征向量维度
        self.roadbook_fc = nn.Sequential(
            nn.Linear(roadbook_input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU()
        )
        
        # 如果使用LSTM
        if use_lstm:
            combined_input_size = cnn_output_size + 128
            self.lstm = nn.LSTM(combined_input_size, 256, batch_first=True, num_layers=2, dropout=0.3)
            lstm_output_size = 256
        else:
            combined_input_size = cnn_output_size + 128
            lstm_output_size = combined_input_size
        
        # 组合特征并输出控制指令
        self.control_fc = nn.Sequential(
            nn.Linear(lstm_output_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3)  # 输出: [steering, throttle, brake]
        )
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """
        初始化模型权重
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self, x, roadbook_features, hidden=None):
        """
        前向传播
        
        Args:
            x: 输入图像 [batch_size, seq_len, C, H, W] 或 [batch_size, C, H, W]
            roadbook_features: 路书特征 [batch_size, seq_len, 5] 或 [batch_size, 5]
            hidden: LSTM隐藏状态
            
        Returns:
            预测的控制指令 [batch_size, seq_len, 3] 或 [batch_size, 3]
        """
        # 检查输入是否有时间维度
        is_sequence = len(x.shape) == 5
        
        if is_sequence:
            batch_size, seq_len = x.shape[0], x.shape[1]
            # 重塑输入以批量处理
            x_reshaped = x.view(-1, x.shape[2], x.shape[3], x.shape[4])
            roadbook_reshaped = roadbook_features.view(-1, roadbook_features.shape[2])
        else:
            x_reshaped = x
            roadbook_reshaped = roadbook_features
        
        # CNN特征提取
        cnn_features = self.cnn_features(x_reshaped)
        cnn_features = F.adaptive_avg_pool2d(cnn_features, (1, 1))
        cnn_features = cnn_features.view(cnn_features.size(0), -1)
        
        # 路书特征处理
        roadbook_embed = self.roadbook_fc(roadbook_reshaped)
        
        # 组合特征
        combined = torch.cat([cnn_features, roadbook_embed], dim=1)
        
        # 如果是序列输入且使用LSTM
        if is_sequence and self.use_lstm:
            combined = combined.view(batch_size, seq_len, -1)
            lstm_out, hidden = self.lstm(combined, hidden)
            control_output = self.control_fc(lstm_out)
        else:
            # 直接通过全连接层
            control_output = self.control_fc(combined)
        
        # 应用激活函数确保输出范围
        steering = torch.tanh(control_output[..., 0])  # -1.0 到 1.0
        throttle = torch.sigmoid(control_output[..., 1])  # 0.0 到 1.0
        brake = torch.sigmoid(control_output[..., 2])  # 0.0 到 1.0
        
        # 组合输出
        outputs = torch.stack([steering, throttle, brake], dim=-1)
        
        if is_sequence and self.use_lstm:
            return outputs, hidden
        else:
            return outputs

class ModelTrainer:
    """
    模型训练器类
    """
    def __init__(self, config=None):
        """
        初始化模型训练器
        
        Args:
            config: 训练配置字典
        """
        # 默认配置 - 优化GPU内存使用
        self.config = {
            'batch_size': 8,  # 减小batch_size以减少内存占用
            'epochs': 20,  # 减少训练轮数
            'learning_rate': 1e-4,
            'weight_decay': 1e-5,
            'use_lstm': False,
            'checkpoint_dir': './models',
            'log_dir': './logs',
            'train_data': './data/processed/train_data.npz',
            'val_data': './data/processed/val_data.npz',
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'gradient_accumulation_steps': 4,  # 梯度累积步数
            'mixed_precision': True,  # 启用混合精度训练
            'clear_cache_interval': 10  # 每10个批次清理一次缓存
        }
        
        # 使用绝对路径，确保目录正确
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.config['checkpoint_dir'] = os.path.join(self.base_dir, self.config['checkpoint_dir'])
        self.config['log_dir'] = os.path.join(self.base_dir, self.config['log_dir'])
        self.config['train_data'] = os.path.join(self.base_dir, self.config['train_data'])
        self.config['val_data'] = os.path.join(self.base_dir, self.config['val_data'])
        
        # 确保目录存在
        os.makedirs(self.config['checkpoint_dir'], exist_ok=True)
        os.makedirs(self.config['log_dir'], exist_ok=True)
        
        # 更新配置
        if config:
            self.config.update(config)
        
        self.device = torch.device(self.config['device'])
        self.model = DrivingModel(use_lstm=self.config['use_lstm']).to(self.device)
        
        # 确保目录存在
        os.makedirs(self.config['checkpoint_dir'], exist_ok=True)
        os.makedirs(self.config['log_dir'], exist_ok=True)
        
        # 准备数据变换
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 训练历史
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_steering_loss': [],
            'val_steering_loss': [],
            'best_val_loss': float('inf')
        }
        
        # 设置混合精度训练
        if self.config['mixed_precision'] and self.device.type == 'cuda':
            from torch.cuda.amp import autocast, GradScaler
            self.autocast = autocast
            self.scaler = GradScaler()
        else:
            self.autocast = lambda: torch.no_grad()
            self.scaler = None
    
    def _create_data_loaders(self):
        """
        创建数据加载器 - 优化内存使用
        """
        # 训练数据集
        train_dataset = DrivingDataset(self.config['train_data'], transform=self.transform)
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=2,  # 减少工作进程数以减少CPU内存使用
            pin_memory=True,
            persistent_workers=False  # 释放工作进程内存
        )
        
        # 验证数据集
        val_dataset = DrivingDataset(self.config['val_data'], transform=self.transform)
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            persistent_workers=False
        )
        
        logger.info(f"数据集加载完成 - 训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}")
    
    def _setup_optimizer(self):
        """
        设置优化器和学习率调度器
        """
        # 参数分组 - 对预训练的CNN使用较小的学习率
        cnn_params = list(self.model.cnn_backbone.parameters())
        other_params = [p for n, p in self.model.named_parameters() if not any(part in n for part in ['cnn_backbone'])]
        
        # 创建优化器
        self.optimizer = optim.Adam([
            {'params': cnn_params, 'lr': self.config['learning_rate'] * 0.1},
            {'params': other_params, 'lr': self.config['learning_rate']}
        ], weight_decay=self.config['weight_decay'])
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
    
    def _compute_loss(self, predictions, targets):
        """
        计算损失
        
        Args:
            predictions: 模型预测 [batch_size, 3]
            targets: 目标值 [batch_size, 3]
            
        Returns:
            总损失和各组件损失
        """
        steering_pred, throttle_pred, brake_pred = predictions[:, 0], predictions[:, 1], predictions[:, 2]
        steering_target, throttle_target, brake_target = targets[:, 0], targets[:, 1], targets[:, 2]
        
        # 使用MSE损失
        steering_loss = F.mse_loss(steering_pred, steering_target)
        throttle_loss = F.mse_loss(throttle_pred, throttle_target)
        brake_loss = F.mse_loss(brake_pred, brake_target)
        
        # 加权损失 - 转向更重要
        total_loss = 2.0 * steering_loss + 1.0 * throttle_loss + 1.0 * brake_loss
        
        return total_loss, steering_loss, throttle_loss, brake_loss
    
    def train_epoch(self, epoch):
        """
        训练一个epoch - 优化GPU内存使用
        
        Args:
            epoch: 当前epoch数
            
        Returns:
            平均训练损失
        """
        import gc
        
        self.model.train()
        total_loss = 0.0
        total_steering_loss = 0.0
        
        progress_bar = tqdm.tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.config["epochs"]}', unit='batch')
        
        # 梯度累积计数器
        accumulation_steps = self.config.get('gradient_accumulation_steps', 1)
        
        for i, batch in enumerate(progress_bar):
            # 准备输入 - 使用inplace操作减少内存
            frames = batch['frame'].to(self.device, non_blocking=True)
            roadbook_features = batch['roadbook_feature'].to(self.device, non_blocking=True)
            
            # 准备目标
            steering_target = batch['steering'].to(self.device, non_blocking=True)
            throttle_target = batch['throttle'].to(self.device, non_blocking=True)
            brake_target = batch['brake'].to(self.device, non_blocking=True)
            targets = torch.stack([steering_target, throttle_target, brake_target], dim=1)
            
            # 混合精度训练
            with self.autocast():
                # 前向传播
                predictions = self.model(frames, roadbook_features)
                
                # 计算损失
                loss, steering_loss, throttle_loss, brake_loss = self._compute_loss(predictions, targets)
                # 缩放损失以配合梯度累积
                loss = loss / accumulation_steps
            
            # 反向传播
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # 只在累积步数达到时更新权重
            if (i + 1) % accumulation_steps == 0:
                if self.scaler:
                    # 梯度裁剪
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    # 更新权重
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                
                # 重置梯度
                self.optimizer.zero_grad(set_to_none=True)  # 使用set_to_none更高效
            
            # 更新统计信息
            batch_size = frames.size(0)
            total_loss += loss.item() * accumulation_steps * batch_size  # 还原损失值
            total_steering_loss += steering_loss.item() * batch_size
            
            # 更新进度条
            progress_bar.set_postfix(loss=loss.item() * accumulation_steps, steering_loss=steering_loss.item())
            
            # 定期清理缓存和内存
            if (i + 1) % self.config['clear_cache_interval'] == 0:
                # 释放中间变量
                del frames, roadbook_features, targets, predictions, loss, steering_loss
                torch.cuda.empty_cache()
                gc.collect()
        
        # 计算平均损失
        avg_loss = total_loss / len(self.train_loader.dataset)
        avg_steering_loss = total_steering_loss / len(self.train_loader.dataset)
        
        logger.info(f"Epoch {epoch+1} 训练损失: {avg_loss:.6f}, 转向损失: {avg_steering_loss:.6f}")
        
        # 保存历史
        self.history['train_loss'].append(avg_loss)
        self.history['train_steering_loss'].append(avg_steering_loss)
        
        return avg_loss
    
    def validate(self):
        """
        验证模型 - 优化GPU内存使用
        
        Returns:
            平均验证损失
        """
        import gc
        
        self.model.eval()
        total_loss = 0.0
        total_steering_loss = 0.0
        
        with torch.no_grad():
            for i, batch in enumerate(tqdm.tqdm(self.val_loader, desc='验证', unit='batch')):
                # 准备输入 - 使用inplace操作
                frames = batch['frame'].to(self.device, non_blocking=True)
                roadbook_features = batch['roadbook_feature'].to(self.device, non_blocking=True)
                
                # 准备目标
                steering_target = batch['steering'].to(self.device, non_blocking=True)
                throttle_target = batch['throttle'].to(self.device, non_blocking=True)
                brake_target = batch['brake'].to(self.device, non_blocking=True)
                targets = torch.stack([steering_target, throttle_target, brake_target], dim=1)
                
                # 混合精度推理
                with self.autocast():
                    # 前向传播
                    predictions = self.model(frames, roadbook_features)
                    
                    # 计算损失
                    loss, steering_loss, throttle_loss, brake_loss = self._compute_loss(predictions, targets)
                
                # 更新统计信息
                batch_size = frames.size(0)
                total_loss += loss.item() * batch_size
                total_steering_loss += steering_loss.item() * batch_size
                
                # 定期清理内存
                if (i + 1) % self.config['clear_cache_interval'] == 0:
                    del frames, roadbook_features, targets, predictions, loss, steering_loss
                    torch.cuda.empty_cache()
                    gc.collect()
        
        # 计算平均损失
        avg_loss = total_loss / len(self.val_loader.dataset)
        avg_steering_loss = total_steering_loss / len(self.val_loader.dataset)
        
        logger.info(f"验证损失: {avg_loss:.6f}, 转向损失: {avg_steering_loss:.6f}")
        
        # 保存历史
        self.history['val_loss'].append(avg_loss)
        self.history['val_steering_loss'].append(avg_steering_loss)
        
        return avg_loss
    
    def save_checkpoint(self, epoch, is_best=False):
        """
        保存模型检查点
        
        Args:
            epoch: 当前epoch
            is_best: 是否为最佳模型
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'config': self.config
        }
        
        # 保存常规检查点
        checkpoint_path = os.path.join(self.config['checkpoint_dir'], f'checkpoint_epoch_{epoch+1}.pth')
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"检查点已保存到: {checkpoint_path}")
        
        # 保存最佳模型
        if is_best:
            best_path = os.path.join(self.config['checkpoint_dir'], 'best_model.pth')
            torch.save(checkpoint, best_path)
            logger.info(f"最佳模型已保存到: {best_path}")
            
            # 同时保存推理模型（仅模型权重）
            inference_path = os.path.join(self.config['checkpoint_dir'], 'inference_model.pth')
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'config': {k: v for k, v in self.config.items() if k != 'device'}
            }, inference_path)
    
    def plot_history(self):
        """
        绘制训练历史
        """
        plt.figure(figsize=(12, 5))
        
        # 总损失图
        plt.subplot(1, 2, 1)
        plt.plot(self.history['train_loss'], label='训练损失')
        plt.plot(self.history['val_loss'], label='验证损失')
        plt.title('总损失')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # 转向损失图
        plt.subplot(1, 2, 2)
        plt.plot(self.history['train_steering_loss'], label='训练转向损失')
        plt.plot(self.history['val_steering_loss'], label='验证转向损失')
        plt.title('转向损失')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        
        # 保存图像
        plot_path = os.path.join(self.config['log_dir'], 'training_history.png')
        plt.savefig(plot_path)
        logger.info(f"训练历史图已保存到: {plot_path}")
    
    def save_history(self):
        """
        保存训练历史
        """
        history_path = os.path.join(self.config['log_dir'], 'training_history.json')
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, indent=2, ensure_ascii=False)
        logger.info(f"训练历史已保存到: {history_path}")
    
    def _validate_npz_file(self, file_path):
        """
        验证NPZ文件是否完整
        
        Args:
            file_path: NPZ文件路径
            
        Returns:
            bool: 文件是否完整
        """
        try:
            print(f"验证文件: {file_path}")
            logger.info(f"验证文件: {file_path}")
            
            # 尝试加载NPZ文件中的所有数据
            with np.load(file_path, allow_pickle=True) as npz:
                # 获取所有键
                keys = list(npz.keys())
                print(f"   文件包含键: {keys}")
                
                # 尝试访问每个键的数据
                for key in keys:
                    data = npz[key]
                    print(f"   成功访问 {key}，形状: {data.shape}")
            
            print(f"   ✅ 文件验证通过")
            logger.info(f"文件验证通过: {file_path}")
            return True
        except Exception as e:
            print(f"   ❌ 文件验证失败: {e}")
            logger.error(f"文件验证失败: {file_path}, 错误: {e}")
            return False
    
    def _check_and_split_data(self):
        """
        检查并分割数据
        """
        # 检查训练集和验证集是否存在，并且验证它们
        train_exists = os.path.exists(self.config['train_data'])
        val_exists = os.path.exists(self.config['val_data'])
        
        # 验证文件是否完整
        if train_exists and val_exists:
            print("⚠️  训练集和验证集文件存在，正在验证文件完整性...")
            logger.info("训练集和验证集文件存在，正在验证文件完整性...")
            
            train_valid = self._validate_npz_file(self.config['train_data'])
            val_valid = self._validate_npz_file(self.config['val_data'])
            
            if train_valid and val_valid:
                # 文件完整，直接返回
                print("✅ 训练集和验证集文件完整")
                logger.info("训练集和验证集文件完整")
                return True
            else:
                # 文件不完整，需要重新生成
                print("❌ 训练集或验证集文件损坏，正在重新生成...")
                logger.info("训练集或验证集文件损坏，正在重新生成...")
        
        # 重新生成数据
        print("⚠️  训练集或验证集文件不存在或损坏，正在尝试自动分割数据...")
        logger.info("训练集或验证集文件不存在或损坏，正在尝试自动分割数据...")
        
        # 获取预处理后的文件列表
        processed_dir = os.path.dirname(self.config['train_data'])
        processed_files = [f for f in os.listdir(processed_dir) if f.endswith('.npz') and f not in ['train_data.npz', 'val_data.npz']]
        
        if not processed_files:
            print("❌ 没有找到预处理后的数据文件")
            print("请先运行数据预处理：python src/data_preprocessor.py")
            logger.error("没有找到预处理后的数据文件")
            return False
        
        print(f"找到 {len(processed_files)} 个预处理后的数据文件")
        logger.info(f"找到 {len(processed_files)} 个预处理后的数据文件")
        
        # 加载所有数据
        all_frames = []
        all_roadbooks = []
        all_steering = []
        all_throttle = []
        all_brake = []
        
        for file in processed_files:
            file_path = os.path.join(processed_dir, file)
            print(f"加载文件: {file}")
            logger.info(f"加载文件: {file}")
            
            try:
                with np.load(file_path, allow_pickle=True) as npz:
                    # 加载数据
                    frames = npz['frames']
                    roadbooks = npz.get('roadbooks', np.array([{} for _ in range(len(frames))]))
                    steering = npz['steering']
                    throttle = npz['throttle']
                    brake = npz['brake']
                    
                    # 添加到总数据中
                    all_frames.append(frames)
                    all_roadbooks.append(roadbooks)
                    all_steering.append(steering)
                    all_throttle.append(throttle)
                    all_brake.append(brake)
                    
                    print(f"   ✅ 成功加载，帧数量: {len(frames)}")
                    logger.info(f"成功加载文件: {file}，帧数量: {len(frames)}")
                    
            except Exception as e:
                print(f"❌ 加载文件 {file} 失败: {e}")
                logger.error(f"加载文件 {file} 失败: {e}")
                continue
        
        # 合并所有数据
        if not all_frames:
            print("❌ 没有成功加载任何数据文件")
            print("请检查预处理后的数据文件，或者重新运行数据预处理")
            logger.error("没有成功加载任何数据文件")
            return False
        
        all_frames = np.concatenate(all_frames, axis=0)
        all_roadbooks = np.concatenate(all_roadbooks, axis=0)
        all_steering = np.concatenate(all_steering, axis=0)
        all_throttle = np.concatenate(all_throttle, axis=0)
        all_brake = np.concatenate(all_brake, axis=0)
        
        print(f"总数据量: {len(all_frames)} 帧")
        logger.info(f"总数据量: {len(all_frames)} 帧")
        
        # 打乱数据
        indices = np.arange(len(all_frames))
        np.random.shuffle(indices)
        
        # 分割训练集和验证集
        train_ratio = 0.8
        split_idx = int(len(all_frames) * train_ratio)
        
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]
        
        print(f"训练集: {len(train_indices)} 帧")
        print(f"验证集: {len(val_indices)} 帧")
        logger.info(f"训练集: {len(train_indices)} 帧，验证集: {len(val_indices)} 帧")
        
        # 保存训练集
        train_data = {
            'frames': all_frames[train_indices],
            'roadbooks': all_roadbooks[train_indices],
            'steering': all_steering[train_indices],
            'throttle': all_throttle[train_indices],
            'brake': all_brake[train_indices]
        }
        
        print("保存训练集...")
        np.savez_compressed(self.config['train_data'], **train_data)
        print(f"✅ 训练集已保存到: {self.config['train_data']}")
        logger.info(f"训练集已保存到: {self.config['train_data']}")
        
        # 保存验证集
        val_data = {
            'frames': all_frames[val_indices],
            'roadbooks': all_roadbooks[val_indices],
            'steering': all_steering[val_indices],
            'throttle': all_throttle[val_indices],
            'brake': all_brake[val_indices]
        }
        
        print("保存验证集...")
        np.savez_compressed(self.config['val_data'], **val_data)
        print(f"✅ 验证集已保存到: {self.config['val_data']}")
        logger.info(f"验证集已保存到: {self.config['val_data']}")
        
        # 验证生成的文件
        print("验证生成的文件...")
        train_valid = self._validate_npz_file(self.config['train_data'])
        val_valid = self._validate_npz_file(self.config['val_data'])
        
        if train_valid and val_valid:
            print("✅ 所有文件生成并验证成功")
            logger.info("所有文件生成并验证成功")
            return True
        else:
            print("❌ 生成的文件仍有问题，请检查系统资源或重新运行")
            logger.error("生成的文件仍有问题")
            return False
    
    def train(self):
        """
        训练模型
        """
        logger.info("开始训练模型...")
        logger.info(f"使用配置: {json.dumps(self.config, indent=2)}")
        
        # 检查并分割数据
        if not self._check_and_split_data():
            return None
        
        # 准备数据
        try:
            self._create_data_loaders()
        except Exception as e:
            print(f"❌ 创建数据加载器失败: {e}")
            logger.error(f"创建数据加载器失败: {e}")
            return None
        
        # 设置优化器
        self._setup_optimizer()
        
        try:
            for epoch in range(self.config['epochs']):
                # 训练
                train_loss = self.train_epoch(epoch)
                
                # 验证
                val_loss = self.validate()
                
                # 更新学习率
                self.scheduler.step(val_loss)
                
                # 检查是否为最佳模型
                is_best = val_loss < self.history['best_val_loss']
                if is_best:
                    self.history['best_val_loss'] = val_loss
                    logger.info(f"新的最佳验证损失: {val_loss:.6f}")
                
                # 保存检查点（每3个epoch或最佳模型）以减少磁盘写入
                if (epoch + 1) % 3 == 0 or is_best:
                    self.save_checkpoint(epoch, is_best)
                
                # 每个epoch结束后清理缓存
                torch.cuda.empty_cache()
                gc.collect()
                
                # 保存历史
                self.save_history()
            
            # 训练完成后绘制历史图
            self.plot_history()
            
            logger.info("训练完成！")
            return self.history['best_val_loss']
            
        except torch.cuda.OutOfMemoryError:
            logger.error("CUDA内存不足错误！尝试清理内存并继续...")
            # 清理内存
            torch.cuda.empty_cache()
            gc.collect()
            # 保存当前状态
            self.save_checkpoint(epoch, False)
            logger.info("已保存当前模型状态，可以调整batch_size后继续训练")
            return None
        except KeyboardInterrupt:
            logger.warning("训练被用户中断")
            # 保存当前状态
            self.save_checkpoint(epoch, False)
            self.plot_history()
            return None
        except Exception as e:
            logger.error(f"训练过程中出错: {e}")
            # 清理内存后再抛出异常
            torch.cuda.empty_cache()
            gc.collect()
            raise

# 示例用法
if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.train()