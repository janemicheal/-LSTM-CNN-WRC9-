"""
数据标注模块 (DataAnnotator)

核心职责：对录制数据进行人工/半自动标注（补充模型训练所需标签）。
"""
import os
import numpy as np
import cv2
import json
from datetime import datetime
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import threading
import queue

class DataAnnotator:
    """数据标注器类"""
    
    def __init__(self, input_dir="../data/processed", output_dir="../data/annotated"):
        """
        初始化数据标注器
        
        Args:
            input_dir: 预处理数据目录或视频文件路径
            output_dir: 标注后数据保存目录
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.current_data = None
        self.current_index = 0
        self.annotations = []
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 训练集/验证集划分比例
        self.train_ratio = 0.8
        
        # GUI相关变量
        self.root = None
        self.image_panel = None
        self.current_frame_tk = None
        
        # 用于处理图像加载的队列
        self.image_queue = queue.Queue()
        self.stop_event = threading.Event()
    
    def load_processed_data(self, file_path):
        """
        加载预处理后的数据或视频文件
        
        Args:
            file_path: 数据文件路径或视频文件路径
            
        Returns:
            数据字典
        """
        try:
            # 检查是否为视频文件
            ext = os.path.splitext(file_path)[1].lower()
            if ext in ['.mp4', '.avi', '.mov', '.mkv']:
                print(f"处理视频文件: {file_path}")
                return self._process_video_file(file_path)
            
            # 否则视为NPZ数据文件
            with np.load(file_path, allow_pickle=True) as npz:
                # 检查必要的键
                required_keys = ['frames', 'steering', 'throttle', 'brake']
                for key in required_keys:
                    if key not in npz:
                        print(f"警告: 数据文件缺少 {key} 键")
                        return None
                
                # 初始化结果字典
                result = {
                    'frames': npz['frames'],
                    'steering': npz['steering'],
                    'throttle': npz['throttle'],
                    'brake': npz['brake']
                }
                
                # 如果没有roadbooks键，创建空的roadbooks数组
                if 'roadbooks' not in npz:
                    print("信息: 数据文件中没有roadbooks键，创建空的roadbooks数据")
                    # 创建空的roadbooks数组，每个元素是一个空字典
                    num_frames = len(npz['frames'])
                    result['roadbooks'] = np.array([{} for _ in range(num_frames)], dtype=object)
                else:
                    result['roadbooks'] = npz['roadbooks']
                
            return result
        except Exception as e:
            print(f"加载数据文件错误: {e}")
            return None
    
    def _process_video_file(self, video_path):
        """
        处理视频文件，提取帧数据
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            格式化的数据字典
        """
        try:
            cap = cv2.VideoCapture(video_path)
            
            # 获取视频信息
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            print(f"视频信息: {total_frames} 帧, {fps:.2f} FPS")
            
            # 提取帧（抽样以避免数据过多）
            frames = []
            roadbooks = []  # 空路书数据，需要后续手动标注
            steering = []  # 空方向盘数据
            throttle = []  # 空油门数据
            brake = []  # 空刹车数据
            speeds = []  # 新增：保存从视频中提取的速度数据
            
            # 设置抽样间隔（每N帧取1帧）
            sample_interval = max(1, int(fps / 10))  # 尝试保持约10FPS的标注速度
            
            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 按间隔抽样
                if frame_count % sample_interval == 0:
                    # 保持原始图像大小
                    frames.append(frame)
                    
                    # 从右下角区域提取速度信息（假设速度显示在右下角）
                    # 这里使用一个默认值，实际项目中需要根据视频特点调整提取逻辑
                    speed = self._extract_speed_from_frame(frame)
                    speeds.append(speed)
                    
                    # 为每个帧创建空的控制数据
                    roadbooks.append({})
                    steering.append(0.0)  # 默认方向盘居中
                    throttle.append(0.5)  # 默认半油门
                    brake.append(0.0)     # 默认不刹车
                
                frame_count += 1
            
            cap.release()
            
            print(f"成功提取 {len(frames)} 帧数据")
            
            # 返回格式化的数据
            return {
                'frames': np.array(frames),
                'roadbooks': np.array(roadbooks, dtype=object),
                'steering': np.array(steering),
                'throttle': np.array(throttle),
                'brake': np.array(brake),
                'speeds': np.array(speeds)  # 新增：返回速度数据
            }
            
        except Exception as e:
            print(f"处理视频文件错误: {e}")
            messagebox.showerror("错误", f"处理视频文件失败: {str(e)}")
            return None
            
    def _extract_speed_from_frame(self, frame):
        """
        从视频帧的右下角区域提取速度信息
        
        Args:
            frame: 视频帧图像
            
        Returns:
            提取的速度值（浮点数）
        """
        # 这里提供一个简单的实现，实际项目中需要根据视频特点调整
        # 例如，可以使用OCR技术识别速度数字，或者根据像素变化估算
        
        # 默认返回一个基准速度值
        return 50.0  # km/h
    
    def save_annotations(self):
        """
        保存标注数据
        """
        if not self.annotations:
            messagebox.showwarning("警告", "没有标注数据需要保存")
            return False
        
        # 生成唯一的文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"annotated_{timestamp}.npz"
        output_path = os.path.join(self.output_dir, filename)
        
        try:
            # 准备保存的数据
            frames = np.array([ann['frame'] for ann in self.annotations])
            roadbooks = np.array([ann['roadbook'] for ann in self.annotations], dtype=object)
            steering = np.array([ann['steering'] for ann in self.annotations])
            throttle = np.array([ann['throttle'] for ann in self.annotations])
            brake = np.array([ann['brake'] for ann in self.annotations])
            track_features = np.array([ann.get('track_features', {}) for ann in self.annotations], dtype=object)
            obstacles = np.array([ann.get('obstacles', {}) for ann in self.annotations], dtype=object)
            speeds = np.array([ann.get('speed', 0.0) for ann in self.annotations])  # 新增：速度数据
            
            # 保存数据
            np.savez_compressed(output_path, 
                               frames=frames,
                               roadbooks=roadbooks,
                               steering=steering,
                               throttle=throttle,
                               brake=brake,
                               track_features=track_features,
                               obstacles=obstacles,
                               speeds=speeds)  # 新增：保存速度数据
            
            # 生成训练集/验证集划分
            self._split_train_val(frames, roadbooks, steering, throttle, brake, track_features, obstacles, timestamp, speeds)
            
            messagebox.showinfo("成功", f"标注数据已保存到: {output_path}")
            return True
        except Exception as e:
            messagebox.showerror("错误", f"保存标注数据时出错: {e}")
            return False
    
    def _split_train_val(self, frames, roadbooks, steering, throttle, brake, track_features, obstacles, timestamp, speeds=None):
        """
        划分训练集和验证集
        """
        total_samples = len(frames)
        train_size = int(total_samples * self.train_ratio)
        
        # 随机打乱数据
        indices = np.random.permutation(total_samples)
        
        # 划分索引
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        # 保存训练集
        train_filename = os.path.join(self.output_dir, "train_data.npz")
        train_kwargs = {
            'frames': frames[train_indices],
            'roadbooks': roadbooks[train_indices],
            'steering': steering[train_indices],
            'throttle': throttle[train_indices],
            'brake': brake[train_indices],
            'track_features': track_features[train_indices],
            'obstacles': obstacles[train_indices]
        }
        # 如果有速度数据，添加到保存参数中
        if speeds is not None:
            train_kwargs['speeds'] = speeds[train_indices]
        
        np.savez_compressed(train_filename, **train_kwargs)
        
        # 保存验证集
        val_filename = os.path.join(self.output_dir, "val_data.npz")
        val_kwargs = {
            'frames': frames[val_indices],
            'roadbooks': roadbooks[val_indices],
            'steering': steering[val_indices],
            'throttle': throttle[val_indices],
            'brake': brake[val_indices],
            'track_features': track_features[val_indices],
            'obstacles': obstacles[val_indices]
        }
        # 如果有速度数据，添加到保存参数中
        if speeds is not None:
            val_kwargs['speeds'] = speeds[val_indices]
        
        np.savez_compressed(val_filename, **val_kwargs)
        
        print(f"训练集大小: {len(train_indices)}, 验证集大小: {len(val_indices)}")
        
        # 保存数据集信息
        info = {
            'train_size': len(train_indices),
            'val_size': len(val_indices),
            'total_size': total_samples,
            'train_ratio': self.train_ratio,
            'creation_time': timestamp
        }
        
        with open(os.path.join(self.output_dir, "dataset_info.json"), 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=2, ensure_ascii=False)
    
    def start_gui(self):
        """
        启动GUI标注界面
        """
        self.root = tk.Tk()
        self.root.title("WRC9自动驾驶数据标注工具")
        self.root.geometry("1200x800")
        self.root.minsize(1000, 600)
        
        # 创建菜单栏
        menubar = tk.Menu(self.root)
        
        # 文件菜单
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="打开数据文件", command=self._open_data_file)
        file_menu.add_command(label="保存标注", command=self.save_annotations)
        file_menu.add_separator()
        file_menu.add_command(label="退出", command=self._quit_app)
        menubar.add_cascade(label="文件", menu=file_menu)
        
        # 标注菜单
        annot_menu = tk.Menu(menubar, tearoff=0)
        annot_menu.add_command(label="设置训练/验证比例", command=self._set_split_ratio)
        menubar.add_cascade(label="标注", menu=annot_menu)
        
        self.root.config(menu=menubar)
        
        # 创建主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 创建右侧控制面板，设置固定宽度
        control_frame = ttk.LabelFrame(main_frame, text="标注控制", padding="10")
        control_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False, padx=5, pady=5)
        control_frame.config(width=350)
        control_frame.pack_propagate(False)  # 防止内容改变面板大小
        
        # 创建左侧图像显示区域，使用滚动窗口
        image_frame = ttk.LabelFrame(main_frame, text="图像预览", padding="10")
        image_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 添加滚动条
        canvas = tk.Canvas(image_frame)
        scroll_x = ttk.Scrollbar(image_frame, orient=tk.HORIZONTAL, command=canvas.xview)
        scroll_y = ttk.Scrollbar(image_frame, orient=tk.VERTICAL, command=canvas.yview)
        
        # 创建用于放置图像的框架
        image_container = ttk.Frame(canvas)
        image_container.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        
        canvas.create_window((0, 0), window=image_container, anchor="nw")
        canvas.configure(xscrollcommand=scroll_x.set, yscrollcommand=scroll_y.set)
        
        # 布局滚动条和画布
        canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
        scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 将图像面板放在容器中并显示
        self.image_panel = ttk.Label(image_container)
        self.image_panel.pack(fill=tk.BOTH, expand=True)  # 确保图像面板可见并能扩展
        
        # 路书信息显示
        roadbook_frame = ttk.LabelFrame(control_frame, text="路书信息", padding="10")
        roadbook_frame.pack(fill=tk.X, pady=5)
        
        self.roadbook_text = tk.Text(roadbook_frame, height=5, width=40)
        self.roadbook_text.pack(fill=tk.BOTH, expand=True)
        self.roadbook_text.config(state=tk.DISABLED)
        
        # 操作控制调整
        control_adjust_frame = ttk.LabelFrame(control_frame, text="操作调整", padding="10")
        control_adjust_frame.pack(fill=tk.X, pady=5)
        
        # 转向控制
        ttk.Label(control_adjust_frame, text="转向角度 (-1.0 to 1.0)").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.steering_var = tk.DoubleVar(value=0.0)
        steering_scale = ttk.Scale(control_adjust_frame, from_=-1.0, to=1.0, orient=tk.HORIZONTAL,
                                  variable=self.steering_var, length=200)
        steering_scale.grid(row=0, column=1, padx=5, pady=2)
        self.steering_value = ttk.Label(control_adjust_frame, text="0.0")
        self.steering_value.grid(row=0, column=2, sticky=tk.W, padx=5)
        
        # 油门控制
        ttk.Label(control_adjust_frame, text="油门值 (0.0 to 1.0)").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.throttle_var = tk.DoubleVar(value=0.0)
        throttle_scale = ttk.Scale(control_adjust_frame, from_=0.0, to=1.0, orient=tk.HORIZONTAL,
                                  variable=self.throttle_var, length=200)
        throttle_scale.grid(row=1, column=1, padx=5, pady=2)
        self.throttle_value = ttk.Label(control_adjust_frame, text="0.0")
        self.throttle_value.grid(row=1, column=2, sticky=tk.W, padx=5)
        
        # 刹车控制
        ttk.Label(control_adjust_frame, text="刹车值 (0.0 to 1.0)").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.brake_var = tk.DoubleVar(value=0.0)
        brake_scale = ttk.Scale(control_adjust_frame, from_=0.0, to=1.0, orient=tk.HORIZONTAL,
                               variable=self.brake_var, length=200)
        brake_scale.grid(row=2, column=1, padx=5, pady=2)
        self.brake_value = ttk.Label(control_adjust_frame, text="0.0")
        self.brake_value.grid(row=2, column=2, sticky=tk.W, padx=5)
        
        # 新增：速度输入面板
        ttk.Label(control_adjust_frame, text="当前速度 (km/h)").grid(row=3, column=0, sticky=tk.W, pady=2)
        self.speed_var = tk.DoubleVar(value=50.0)
        speed_entry = ttk.Entry(control_adjust_frame, textvariable=self.speed_var, width=10)
        speed_entry.grid(row=3, column=1, padx=5, pady=2, sticky=tk.W)
        ttk.Label(control_adjust_frame, text="km/h").grid(row=3, column=1, padx=(150, 5), pady=2, sticky=tk.W)
        
        # 更新显示值
        def update_values(*args):
            self.steering_value.config(text=f"{self.steering_var.get():.2f}")
            self.throttle_value.config(text=f"{self.throttle_var.get():.2f}")
            self.brake_value.config(text=f"{self.brake_var.get():.2f}")
            # 当速度改变时，重新计算油门和刹车
            try:
                speed = self.speed_var.get()
                # 简单的速度到油门/刹车的映射逻辑
                if speed > 60:
                    self.throttle_var.set(0.6)
                    self.brake_var.set(0.0)
                elif speed < 40:
                    self.throttle_var.set(0.8)
                    self.brake_var.set(0.0)
                else:
                    self.throttle_var.set(0.5)
                    self.brake_var.set(0.0)
            except:
                pass
        
        self.steering_var.trace_add("write", update_values)
        self.throttle_var.trace_add("write", update_values)
        self.brake_var.trace_add("write", update_values)
        self.speed_var.trace_add("write", update_values)
        
        # 手动路书标注控件
        roadbook_frame = ttk.LabelFrame(control_frame, text="路书标注", padding="10")
        roadbook_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # 方向选择 - 使用统一的pack布局确保完整显示
        direction_label = ttk.Label(roadbook_frame, text="方向选择:")
        direction_label.pack(anchor=tk.W, pady=(5, 2))
        
        self.road_direction = tk.StringVar(value="straight")
        directions_frame = ttk.Frame(roadbook_frame)
        directions_frame.pack(anchor=tk.W, pady=(0, 10))
        
        # 确保所有方向选项完整显示
        ttk.Radiobutton(directions_frame, text="直行", variable=self.road_direction, value="straight", command=lambda: self._update_curve_value("6")).pack(side=tk.LEFT, padx=10)
        ttk.Radiobutton(directions_frame, text="左转", variable=self.road_direction, value="left", command=lambda: self._update_curve_value(str(self.curve_intensity.get()))).pack(side=tk.LEFT, padx=10)
        ttk.Radiobutton(directions_frame, text="右转", variable=self.road_direction, value="right", command=lambda: self._update_curve_value(str(self.curve_intensity.get()))).pack(side=tk.LEFT, padx=10)
        
        # 转弯幅度部分 - 使用pack布局确保完整显示
        curve_label = ttk.Label(roadbook_frame, text="转弯幅度 (6-1, 数字越小弯度越大):")
        curve_label.pack(anchor=tk.W, pady=(5, 2))
        
        self.curve_intensity = tk.IntVar(value=6)
        
        # 创建输入框容器并使用pack布局
        curve_scale_frame = ttk.Frame(roadbook_frame)
        curve_scale_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 添加输入验证器，确保输入值在1-6之间
        def validate_curve_input(new_value):
            if new_value == "":
                return True
            try:
                value = int(new_value)
                return 1 <= value <= 6
            except ValueError:
                return False
        
        # 注册验证器
        validate_cmd = roadbook_frame.register(validate_curve_input)
        
        # 替换滑块为输入框
        self.curve_entry = ttk.Entry(curve_scale_frame, 
                                    textvariable=self.curve_intensity,
                                    validate="key",
                                    validatecommand=(validate_cmd, "%P"),
                                    width=5,
                                    font=('SimHei', 12))
        self.curve_entry.pack(side=tk.LEFT, padx=5)
        self.curve_entry.bind("<FocusOut>", lambda event: self._validate_curve_input())
        
        # 保持值标签用于显示
        self.curve_value_label = ttk.Label(curve_scale_frame, 
                                          text="6", 
                                          font=('SimHei', 12),
                                          width=3,
                                          padding=(5, 0))
        self.curve_value_label.pack(side=tk.LEFT, padx=10)
        
        # 赛道特征标签（保留部分必要的特征）
        track_frame = ttk.LabelFrame(control_frame, text="赛道特征", padding="10")
        track_frame.pack(fill=tk.X, pady=5)
        
        # 初始化标签变量
        self.track_features = {
            'has_obstacle': tk.BooleanVar(value=False),
            'is_wet': tk.BooleanVar(value=False)
        }
        
        # 杂物标记
        self.obstacle_types = {
            'has_vegetation': tk.BooleanVar(value=False),
            'has_sign': tk.BooleanVar(value=False),
            'has_barrier': tk.BooleanVar(value=False),
            'has_other_car': tk.BooleanVar(value=False),
            'has_other_obstacle': tk.BooleanVar(value=False)
        }
        
        # 显示赛道特征和障碍物
        ttk.Checkbutton(track_frame, text="障碍物", variable=self.track_features['has_obstacle']).grid(row=0, column=0, sticky=tk.W, pady=2)
        ttk.Checkbutton(track_frame, text="湿滑路面", variable=self.track_features['is_wet']).grid(row=0, column=1, sticky=tk.W, pady=2)
        
        # 杂物标记
        obstacle_frame = ttk.LabelFrame(control_frame, text="画面杂物", padding="10")
        obstacle_frame.pack(fill=tk.X, pady=5)
        
        ttk.Checkbutton(obstacle_frame, text="植被", variable=self.obstacle_types['has_vegetation']).grid(row=0, column=0, sticky=tk.W, pady=2)
        ttk.Checkbutton(obstacle_frame, text="路牌/标志", variable=self.obstacle_types['has_sign']).grid(row=0, column=1, sticky=tk.W, pady=2)
        ttk.Checkbutton(obstacle_frame, text="护栏", variable=self.obstacle_types['has_barrier']).grid(row=1, column=0, sticky=tk.W, pady=2)
        ttk.Checkbutton(obstacle_frame, text="其他车辆", variable=self.obstacle_types['has_other_car']).grid(row=1, column=1, sticky=tk.W, pady=2)
        ttk.Checkbutton(obstacle_frame, text="其他杂物", variable=self.obstacle_types['has_other_obstacle']).grid(row=2, column=0, sticky=tk.W, pady=2)
        
        # 导航按钮
        nav_frame = ttk.Frame(control_frame)
        nav_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(nav_frame, text="上一帧", command=self._prev_frame).grid(row=0, column=0, padx=5, pady=5)
        ttk.Button(nav_frame, text="保存当前标注", command=self._save_current_annotation).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(nav_frame, text="下一帧", command=self._next_frame).grid(row=0, column=2, padx=5, pady=5)
        
        # 状态栏
        self.status_var = tk.StringVar(value="就绪: 请打开数据文件")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # 启动图像加载线程
        self.image_thread = threading.Thread(target=self._image_loading_thread)
        self.image_thread.daemon = True
        self.image_thread.start()
        
        # 启动GUI主循环
        self.root.mainloop()
    

    
    def _update_image_display(self, tk_image):
        """更新图像显示"""
        self.current_frame_tk = tk_image  # 保持引用以防止被垃圾回收
        
        # 清除image_panel上的现有图像配置
        self.image_panel.config(image='')
        
        # 设置新图像
        self.image_panel.config(image=tk_image)
        
        # 确保image_panel正确显示并更新滚动区域
        if self.root:
            self.root.update_idletasks()  # 确保UI更新

    def _image_loading_thread(self):
        """图像预加载线程"""
        while not self.stop_event.is_set():
            try:
                # 从队列获取加载任务
                frame = self.image_queue.get(timeout=1.0)
                
                # 调试信息
                print(f"处理图像: 形状={frame.shape}, 数据类型={frame.dtype}, 最小值={frame.min()}, 最大值={frame.max()}")
                
                # 处理图像 - 修复颜色和格式转换问题
                # 检查图像数据范围
                if frame.dtype == np.uint8:
                    # 如果已经是0-255范围的uint8类型，直接处理
                    rgb_image = frame.copy()  # 创建副本避免修改原始数据
                else:
                    # 如果是浮点数或其他类型，需要归一化到0-255范围
                    rgb_image = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
                
                # 检查图像通道顺序并转换为RGB
                # OpenCV通常使用BGR顺序，需要转换为RGB以便PIL正确显示
                if len(rgb_image.shape) > 2 and rgb_image.shape[2] == 3:
                    # 将BGR转换为RGB（OpenCV读取的默认格式是BGR）
                    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
                elif len(rgb_image.shape) > 2 and rgb_image.shape[2] == 4:
                    # 如果是BGRA格式，也需要转换
                    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGRA2RGBA)
                
                # 调试信息
                print(f"转换后图像: 形状={rgb_image.shape}, 数据类型={rgb_image.dtype}")
                
                # 转换为PIL图像
                pil_image = Image.fromarray(rgb_image)
                
                # 保持图像原始尺寸，不进行缩放
                # 获取原始尺寸
                orig_width, orig_height = pil_image.size
                print(f"保持图像原始尺寸: {orig_width}x{orig_height}")
                
                # 转换为Tkinter可用的格式
                tk_image = ImageTk.PhotoImage(image=pil_image)
                
                # 更新GUI
                if self.root:
                    self.root.after(0, self._update_image_display, tk_image)
                
                self.image_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"图像加载线程错误: {e}")
                # 发生错误时，尝试清空队列，避免一直处理错误的图像
                try:
                    while not self.image_queue.empty():
                        self.image_queue.get(False)
                        self.image_queue.task_done()
                except:
                    pass
    
    def _open_data_file(self):
        """
        打开数据文件或视频文件
        """
        file_path = filedialog.askopenfilename(
            title="选择数据文件或视频文件",
            filetypes=[
                ("预处理数据文件", "*.npz"),
                ("视频文件", "*.mp4 *.avi *.mov *.mkv"),
                ("所有文件", "*.*")
            ],
            initialdir=self.input_dir
        )
        
        if file_path:
            self.current_data = self.load_processed_data(file_path)
            if self.current_data:
                self.current_index = 0
                self.annotations = []
                self.status_var.set(f"已加载数据: {len(self.current_data['frames'])} 帧")
                self._load_frame(0)
            else:
                messagebox.showerror("错误", "加载数据文件失败")
    
    def _load_frame(self, index):
        """
        加载指定索引的帧
        """
        if not self.current_data or index < 0 or index >= len(self.current_data['frames']):
            return
        
        # 更新状态
        self.current_index = index
        
        # 获取帧数据
        frame = self.current_data['frames'][index]
        
        # 安全地获取roadbook数据，如果不存在则使用空字典
        roadbook = {}
        if 'roadbooks' in self.current_data and index < len(self.current_data['roadbooks']):
            roadbook = self.current_data['roadbooks'][index]
            # 确保roadbook是字典类型
            if not isinstance(roadbook, dict):
                roadbook = {}
        
        # 获取控制数据
        steering = self.current_data['steering'][index] if index < len(self.current_data['steering']) else 0.0
        throttle = self.current_data['throttle'][index] if index < len(self.current_data['throttle']) else 0.0
        brake = self.current_data['brake'][index] if index < len(self.current_data['brake']) else 0.0
        
        # 新增：基于速度差值自动计算油门和刹车值
        if 'speeds' in self.current_data:
            speed_current = self.current_data['speeds'][index]
            # 尝试获取下一帧的速度（如果存在）
            if index + 1 < len(self.current_data['speeds']):
                speed_next = self.current_data['speeds'][index + 1]
                speed_diff = speed_next - speed_current
                
                # 根据速度差值调整油门和刹车
                # 加速时增加油门
                if speed_diff > 1.0:
                    throttle = min(1.0, 0.5 + speed_diff * 0.05)  # 根据加速度调整油门
                    brake = 0.0
                # 减速时增加刹车
                elif speed_diff < -1.0:
                    brake = min(1.0, 0.1 + abs(speed_diff) * 0.03)  # 根据减速度调整刹车
                    throttle = 0.0
                # 匀速时保持适中油门
                else:
                    throttle = 0.5
                    brake = 0.0
        
        # 更新状态栏信息
        status_text = f"帧 {index + 1}/{len(self.current_data['frames'])}"
        if 'speeds' in self.current_data:
            speed_current = self.current_data['speeds'][index]
            status_text += f" | 速度: {speed_current:.1f} km/h"
        self.status_var.set(status_text)
        
        # 更新路书信息显示
        roadbook_str = self._format_roadbook_info(roadbook)
        self.roadbook_text.config(state=tk.NORMAL)
        self.roadbook_text.delete(1.0, tk.END)
        self.roadbook_text.insert(tk.END, roadbook_str)
        self.roadbook_text.config(state=tk.DISABLED)
        
        # 更新控制滑块
        self.steering_var.set(steering)
        self.throttle_var.set(throttle)
        self.brake_var.set(brake)
        
        # 重置所有标签
        for var in self.track_features.values():
            var.set(False)
        # 移除路面类型相关的重置，因为我们不再使用路面类型标注
        for var in self.obstacle_types.values():
            var.set(False)
        
        # 将图像加载任务放入队列
        self.image_queue.put(frame)
    
    def _prev_frame(self):
        """显示上一帧"""
        if self.current_index > 0:
            self._load_frame(self.current_index - 1)
    
    def _next_frame(self):
        """显示下一帧"""
        if self.current_data and self.current_index < len(self.current_data['frames']) - 1:
            self._load_frame(self.current_index + 1)
    
    def _update_curve_value(self, value=None):
        """
        更新转弯幅度输入框的值显示，确保实时更新
        """
        try:
            # 获取当前值
            int_value = self.curve_intensity.get()
            
            # 更新值显示标签
            if hasattr(self, 'curve_value_label'):
                self.curve_value_label.config(
                    text=str(int_value),
                    font=('SimHei', 12)
                )
            
            # 如果选择直行，设置为6
            if hasattr(self, 'road_direction'):
                if self.road_direction.get() == "straight":
                    # 当选择直行时，确保显示为6
                    self.curve_intensity.set(6)
                    if hasattr(self, 'curve_value_label'):
                        self.curve_value_label.config(text="6")
        except Exception as e:
            print(f"更新转弯幅度出错: {e}")
            
    def _validate_curve_input(self):
        """
        验证输入框的值，确保在1-6范围内
        """
        try:
            value = self.curve_intensity.get()
            # 如果值为空或不在范围内，设置为默认值6
            if not (1 <= value <= 6):
                self.curve_intensity.set(6)
            # 更新显示
            self._update_curve_value()
        except:
            self.curve_intensity.set(6)
            self._update_curve_value()
    
    def _format_roadbook_info(self, roadbook):
        """
        格式化路书信息，使显示更加清晰
        
        Args:
            roadbook: 路书数据字典
            
        Returns:
            格式化后的路书信息字符串
        """
        if not roadbook:
            return "无可用路书信息"
            
        # 解析不同类型的路书数据
        try:
            formatted_lines = []
            
            # 基本方向信息（新格式）
            if 'direction' in roadbook:
                direction = roadbook['direction']
                direction_text = {
                    'left': '左转',
                    'right': '右转',
                    'straight': '直行'
                }.get(direction, direction)
                formatted_lines.append(f"方向: {direction_text}")
                
                # 显示转弯幅度
                if direction in ['left', 'right'] and 'intensity' in roadbook:
                    intensity = roadbook['intensity']
                    # 计算弯度描述
                    curve_desc = ""
                    if intensity <= 2:
                        curve_desc = "（急弯）"
                    elif intensity <= 4:
                        curve_desc = "（中弯）"
                    else:
                        curve_desc = "（缓弯）"
                    formatted_lines.append(f"转弯幅度: {intensity} {curve_desc}")
            
            # 如果没有以上信息，显示原始数据
            if not formatted_lines and isinstance(roadbook, dict):
                return json.dumps(roadbook, ensure_ascii=False, indent=2)
                
            return "\n".join(formatted_lines)
            
        except Exception as e:
            # 出错时返回原始数据
            return json.dumps(roadbook, ensure_ascii=False, indent=2)
    
    def _save_current_annotation(self):
        """
        保存当前帧的标注
        """
        if not self.current_data or self.current_index < 0 or self.current_index >= len(self.current_data['frames']):
            messagebox.showwarning("警告", "没有可保存的当前帧数据")
            return
        
        # 收集标注数据
        frame = self.current_data['frames'][self.current_index]
        steering = self.steering_var.get()
        throttle = self.throttle_var.get()
        brake = self.brake_var.get()
        
        # 创建新的手动路书标注数据
        roadbook = {
            'direction': self.road_direction.get(),
            'intensity': self.curve_intensity.get() if self.road_direction.get() in ['left', 'right'] else 0
        }
        
        # 收集赛道特征
        track_features = {}
        for key, var in self.track_features.items():
            track_features[key] = var.get()
        
        # 收集杂物信息
        obstacles = {}
        for key, var in self.obstacle_types.items():
            obstacles[key] = var.get()
        
        # 创建标注样本
        annotation = {
            'frame': frame,
            'roadbook': roadbook,
            'steering': steering,
            'throttle': throttle,
            'brake': brake,
            'track_features': track_features,
            'obstacles': obstacles,
            'original_index': self.current_index
        }
        
        # 如果有速度数据，也保存下来
        if 'speeds' in self.current_data and self.current_index < len(self.current_data['speeds']):
            annotation['speed'] = self.current_data['speeds'][self.current_index]
        
        # 检查是否已存在该帧的标注
        existing_index = next((i for i, ann in enumerate(self.annotations) 
                             if ann['original_index'] == self.current_index), None)
        
        if existing_index is not None:
            self.annotations[existing_index] = annotation
        else:
            self.annotations.append(annotation)
        
        # 更新状态栏，显示更详细的信息
        status_text = f"帧: {self.current_index + 1}/{len(self.current_data['frames'])} | 已保存标注: 共 {len(self.annotations)} 条"
        if 'speeds' in self.current_data and self.current_index < len(self.current_data['speeds']):
            status_text += f" | 速度: {self.current_data['speeds'][self.current_index]:.1f} km/h"
        self.status_var.set(status_text)
    
    def _set_split_ratio(self):
        """设置训练/验证集划分比例"""
        dialog = tk.Toplevel(self.root)
        dialog.title("设置训练/验证集比例")
        dialog.geometry("300x150")
        dialog.transient(self.root)
        dialog.grab_set()
        
        ttk.Label(dialog, text="训练集比例 (0.1 - 0.9)").pack(pady=10)
        
        ratio_var = tk.DoubleVar(value=self.train_ratio)
        ratio_scale = ttk.Scale(dialog, from_=0.1, to=0.9, orient=tk.HORIZONTAL,
                               variable=ratio_var, length=200)
        ratio_scale.pack(pady=5)
        
        ratio_value = ttk.Label(dialog, text=f"{self.train_ratio:.2f}")
        ratio_value.pack(pady=5)
        
        def update_ratio_label(*args):
            ratio_value.config(text=f"{ratio_var.get():.2f}")
        
        ratio_var.trace_add("write", update_ratio_label)
        
        def apply_ratio():
            self.train_ratio = ratio_var.get()
            dialog.destroy()
        
        button_frame = ttk.Frame(dialog)
        button_frame.pack(pady=10)
        
        ttk.Button(button_frame, text="应用", command=apply_ratio).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="取消", command=dialog.destroy).pack(side=tk.RIGHT, padx=5)
    
    def _quit_app(self):
        """退出应用"""
        # 提示保存
        if self.annotations:
            if messagebox.askyesnocancel("保存", "是否保存当前标注数据？"):
                self.save_annotations()
            elif None:  # 用户点击取消
                return
        
        # 停止图像加载线程
        self.stop_event.set()
        
        # 销毁GUI
        self.root.destroy()

# 示例用法
if __name__ == "__main__":
    annotator = DataAnnotator()
    annotator.start_gui()