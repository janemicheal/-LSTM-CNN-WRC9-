"""
路书识别模块 (RoadbookRecognizer)

核心职责：实时识别屏幕上路书文本并转为结构化特征，为自动驾驶模型提供关键的导航信息。
"""
import os
import re
import cv2
import numpy as np
from PIL import Image
import logging
from typing import Dict, Optional, Tuple, List
from datetime import datetime

# 条件导入pytesseract模块
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    logging.warning("未找到pytesseract模块，请使用 'pip install pytesseract' 安装")
    logging.warning("注意：还需要安装Tesseract OCR引擎，请参考官方文档")

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='../logs/roadbook_recognizer.log'
)
logger = logging.getLogger('RoadbookRecognizer')

class RoadbookRecognizer:
    """
    路书识别器类，用于从图像中识别路书文本并解析为结构化数据
    """
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化路书识别器
        
        Args:
            config: 配置字典
        """
        # 默认配置
        self.default_config = {
            'tesseract_cmd': None,         # Tesseract可执行文件路径
            'roadbook_region': (430, 530, 30, 110),  # 路书区域 [左, 上, 宽, 高] - 适配960*540分辨率下路书位置
            'apply_preprocessing': True,   # 是否应用图像预处理
            'language': 'chi_sim+eng',     # Tesseract语言设置
            'oem': 3,                      # OCR引擎模式 (3=默认LSTM引擎)
            'psm': 7,                      # 页面分割模式 (7=单行文本)
            'confidence_threshold': 0.6,   # 置信度阈值
            'use_template_matching': True, # 是否使用模板匹配
            'templates_dir': '../config/roadbook_templates', # 模板目录
            'enable_cache': True,          # 是否启用缓存
            'cache_size': 100,             # 缓存大小
            'debug': False                 # 是否启用调试模式
        }
        
        # 合并配置
        self.config = self.default_config.copy()
        if config:
            self.config.update(config)
        
        # 检查pytesseract是否可用
        if not TESSERACT_AVAILABLE:
            logging.warning("pytesseract不可用，将禁用OCR相关功能")
            
        # 配置Tesseract路径
        if TESSERACT_AVAILABLE and self.config['tesseract_cmd']:
            pytesseract.pytesseract.tesseract_cmd = self.config['tesseract_cmd']
        
        # 初始化缓存
        self.cache = {}
        self.cache_history = []
        
        # 加载模板（如果启用）
        self.templates = {}
        if self.config['use_template_matching']:
            self._load_templates()
        
        # 初始化结果变量
        self.last_roadbook = None
        self.last_confidence = 0.0
        self.recognition_history = []
    
    def _load_templates(self):
        """
        加载路书识别模板
        """
        if not os.path.exists(self.config['templates_dir']):
            logger.warning(f"模板目录不存在: {self.config['templates_dir']}")
            return
        
        try:
            # 这里可以加载预定义的路书模板图像
            # 例如：方向箭头、数字模板等
            template_files = [f for f in os.listdir(self.config['templates_dir']) 
                            if f.endswith(('.png', '.jpg', '.jpeg'))]
            
            for template_file in template_files:
                template_path = os.path.join(self.config['templates_dir'], template_file)
                template_name = os.path.splitext(template_file)[0]
                
                # 加载模板并转换为灰度
                template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
                if template is not None:
                    self.templates[template_name] = template
                    logger.info(f"加载模板: {template_name}")
                else:
                    logger.warning(f"无法加载模板: {template_path}")
                    
            logger.info(f"总共加载了 {len(self.templates)} 个模板")
            
        except Exception as e:
            logger.error(f"加载模板时出错: {e}")
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        预处理图像以提高OCR识别率
        
        Args:
            image: 输入图像
            
        Returns:
            预处理后的图像
        """
        try:
            # 转换为灰度图
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image.copy()
            
            # 应用自适应阈值
            thresh = cv2.adaptiveThreshold(
                gray, 255, 
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY_INV, 
                11, 2
            )
            
            # 降噪 - 中值滤波
            denoised = cv2.medianBlur(thresh, 3)
            
            # 进行形态学操作 - 膨胀
            kernel = np.ones((2, 2), np.uint8)
            processed = cv2.morphologyEx(denoised, cv2.MORPH_DILATE, kernel)
            
            # 反转颜色使文本为白色，背景为黑色
            processed = 255 - processed
            
            return processed
            
        except Exception as e:
            logger.error(f"预处理图像时出错: {e}")
            return image
    
    def _extract_roadbook_region(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        从完整帧中提取路书区域
        
        Args:
            frame: 完整游戏画面帧
            
        Returns:
            路书区域图像，如果提取失败则返回None
        """
        try:
            left, top, width, height = self.config['roadbook_region']
            
            # 确保区域在有效范围内
            h, w = frame.shape[:2]
            if left < 0 or top < 0 or left + width > w or top + height > h:
                logger.warning(f"路书区域超出图像范围: ({left}, {top}, {width}, {height}), 图像大小: {w}x{h}")
                # 调整区域到有效范围
                left = max(0, min(left, w - 1))
                top = max(0, min(top, h - 1))
                width = max(1, min(width, w - left))
                height = max(1, min(height, h - top))
            
            # 提取区域
            roadbook_region = frame[top:top+height, left:left+width]
            
            # 调试模式下保存提取的区域
            if self.config['debug']:
                debug_dir = '../logs/debug'
                os.makedirs(debug_dir, exist_ok=True)
                debug_path = os.path.join(debug_dir, f"roadbook_region_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
                cv2.imwrite(debug_path, cv2.cvtColor(roadbook_region, cv2.COLOR_RGB2BGR))
            
            return roadbook_region
            
        except Exception as e:
            logger.error(f"提取路书区域时出错: {e}")
            return None
    
    def _template_match(self, roadbook_region: np.ndarray) -> Dict:
        """
        使用模板匹配识别路书元素
        
        Args:
            roadbook_region: 路书区域图像
            
        Returns:
            匹配结果字典
        """
        results = {
            'direction': None,
            'degree': 0,
            'speed': 0,
            'has_direction': False,
            'has_degree': False,
            'has_speed': False,
            'confidence': 0.5  # 模板匹配的基础置信度
        }
        
        # 这里实现简单的模板匹配逻辑
        # 实际应用中可能需要更复杂的匹配策略
        if not self.templates:
            return results
        
        try:
            # 转换为灰度图
            gray = cv2.cvtColor(roadbook_region, cv2.COLOR_RGB2GRAY) if len(roadbook_region.shape) == 3 else roadbook_region
            
            # 对每个模板进行匹配
            for template_name, template in self.templates.items():
                # 进行模板匹配
                result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(result)
                
                # 如果匹配分数高于阈值，记录匹配结果
                if max_val > 0.7:
                    logger.debug(f"模板匹配成功: {template_name}, 分数: {max_val:.2f}")
                    
                    # 根据模板名称推断路书元素
                    if 'left' in template_name.lower():
                        results['direction'] = 'left'
                        results['has_direction'] = True
                    elif 'right' in template_name.lower():
                        results['direction'] = 'right'
                        results['has_direction'] = True
                    elif 'straight' in template_name.lower():
                        results['direction'] = 'straight'
                        results['has_direction'] = True
                    
                    # 更新置信度
                    results['confidence'] += max_val * 0.1
            
        except Exception as e:
            logger.error(f"模板匹配时出错: {e}")
        
        # 确保置信度不超过1.0
        results['confidence'] = min(1.0, results['confidence'])
        
        return results
    
    def _ocr_text(self, image: np.ndarray) -> Tuple[str, float]:
        """
        使用OCR识别文本
        
        Args:
            image: 输入图像
            
        Returns:
            识别的文本和平均置信度
        """
        # 检查pytesseract是否可用
        if not TESSERACT_AVAILABLE:
            logger.warning("pytesseract不可用，无法执行OCR识别")
            return "", 0.0
            
        try:
            # 配置Tesseract参数
            custom_config = f'--oem {self.config["oem"]} --psm {self.config["psm"]}'
            
            # 获取详细数据，包括每个字符的置信度
            details = pytesseract.image_to_data(
                image,
                output_type=pytesseract.Output.DICT,
                lang=self.config['language'],
                config=custom_config
            )
            
            # 提取文本和计算平均置信度
            texts = []
            confidences = []
            
            for i, text in enumerate(details['text']):
                # 过滤空字符串和低置信度文本
                clean_text = text.strip()
                if clean_text and details['conf'][i] > 0:
                    texts.append(clean_text)
                    confidences.append(float(details['conf'][i]) / 100.0)  # 转换为0-1范围
            
            # 组合文本
            recognized_text = ' '.join(texts)
            
            # 计算平均置信度
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            return recognized_text, avg_confidence
            
        except Exception as e:
            logger.error(f"OCR识别时出错: {e}")
            return "", 0.0
    
    def _parse_roadbook_text(self, text: str, template_results: Dict = None) -> Tuple[Dict, float]:
        """
        解析路书文本为结构化数据
        
        Args:
            text: 识别的路书文本
            template_results: 模板匹配结果
            
        Returns:
            结构化路书数据和解析置信度
        """
        # 初始化结果
        roadbook = {
            'direction': 'straight',  # 方向: left, straight, right
            'degree': 0,             # 弯道度数
            'speed': 0,              # 建议速度
            'slope': 'flat',         # 坡度: up, flat, down
            'special': None,         # 特殊路况
            'raw_text': text         # 原始文本
        }
        
        # 如果有模板匹配结果，先使用它
        if template_results and template_results['direction']:
            roadbook['direction'] = template_results['direction']
        
        # 清理文本
        clean_text = text.strip().lower()
        confidence = 0.5  # 基础置信度
        
        # 定义正则表达式模式
        patterns = {
            # 方向模式
            'direction': [
                (r'左', 'left'),
                (r'右', 'right'),
                (r'直', 'straight')
            ],
            # 度数模式
            'degree': [
                (r'[左直右]?\s*(\d{1,2})\s*', lambda x: int(x.group(1)))
            ],
            # 速度模式
            'speed': [
                (r'(\d{2,3})\s*$', lambda x: int(x.group(1))),
                (r'(\d{2,3})\s*km|公里', lambda x: int(x.group(1))),
                (r'速度[:：]?\s*(\d{2,3})', lambda x: int(x.group(1)))
            ],
            # 坡度模式
            'slope': [
                (r'坡上|上坡|上升', 'up'),
                (r'坡下|下坡|下降', 'down'),
                (r'平', 'flat')
            ],
            # 特殊路况
            'special': [
                (r'跳|跳跃', 'jump'),
                (r'水|涉水', 'water'),
                (r'坑|凹', 'hole'),
                (r'桥', 'bridge'),
                (r'泥|泥泞', 'mud')
            ]
        }
        
        # 应用模式匹配
        for field, field_patterns in patterns.items():
            for pattern, value_map in field_patterns:
                matches = re.finditer(pattern, clean_text)
                for match in matches:
                    # 对于复杂的值映射（如函数）
                    if callable(value_map):
                        extracted_value = value_map(match)
                    else:
                        extracted_value = value_map
                    
                    # 更新路书数据
                    if field != 'direction' or roadbook['direction'] == 'straight':
                        # 只有当方向还未确定时才更新方向
                        roadbook[field] = extracted_value
                    
                    # 增加置信度
                    confidence += 0.1
                    
                    # 对于度数，特殊处理
                    if field == 'degree':
                        # 根据度数大小进一步确认方向
                        if roadbook['degree'] > 0:
                            if '左' in match.group(0):
                                roadbook['direction'] = 'left'
                            elif '右' in match.group(0):
                                roadbook['direction'] = 'right'
        
        # 特殊处理：如果没有明确方向但有度数，可能是直道但有微调
        if roadbook['direction'] == 'straight' and roadbook['degree'] > 0:
            # 检查是否有左右相关的字符
            if '左' in clean_text:
                roadbook['direction'] = 'left'
                confidence += 0.1
            elif '右' in clean_text:
                roadbook['direction'] = 'right'
                confidence += 0.1
        
        # 确保置信度在合理范围内
        confidence = min(1.0, confidence)
        
        # 进一步验证结果的合理性
        if roadbook['degree'] > 10:
            # 通常弯道度数不会超过10
            logger.warning(f"检测到异常弯道度数: {roadbook['degree']}")
            confidence -= 0.2
        
        if roadbook['speed'] > 200:
            # 通常建议速度不会超过200
            logger.warning(f"检测到异常速度值: {roadbook['speed']}")
            confidence -= 0.2
        
        return roadbook, max(0.0, confidence)  # 确保置信度不为负
    
    def _update_cache(self, key: str, value: Dict):
        """
        更新识别缓存
        
        Args:
            key: 缓存键
            value: 缓存值
        """
        if not self.config['enable_cache']:
            return
        
        # 添加到缓存
        self.cache[key] = value
        self.cache_history.append(key)
        
        # 如果缓存超过大小限制，移除最旧的项目
        if len(self.cache) > self.config['cache_size']:
            oldest_key = self.cache_history.pop(0)
            if oldest_key in self.cache:
                del self.cache[oldest_key]
    
    def _get_cache_key(self, frame: np.ndarray) -> str:
        """
        从帧生成缓存键
        
        Args:
            frame: 输入帧
            
        Returns:
            缓存键
        """
        # 使用帧的直方图作为缓存键
        # 这是一种快速判断帧是否相似的方法
        hist = cv2.calcHist([frame], [0], None, [16], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        return hash(hist.tobytes())
    
    def recognize_roadbook(self, frame: np.ndarray) -> Tuple[Optional[Dict], float]:
        """
        识别路书
        
        Args:
            frame: 完整游戏画面帧
            
        Returns:
            结构化路书数据和识别置信度
        """
        try:
            # 检查缓存
            if self.config['enable_cache']:
                cache_key = self._get_cache_key(frame)
                if cache_key in self.cache:
                    cached_result, cached_conf = self.cache[cache_key]
                    logger.debug(f"使用缓存结果，置信度: {cached_conf:.2f}")
                    return cached_result, cached_conf
            
            # 提取路书区域
            roadbook_region = self._extract_roadbook_region(frame)
            if roadbook_region is None:
                return None, 0.0
            
            # 应用预处理
            if self.config['apply_preprocessing']:
                processed = self._preprocess_image(roadbook_region)
            else:
                processed = roadbook_region
            
            # 模板匹配（如果启用）
            template_results = {} if not self.config['use_template_matching'] else self._template_match(roadbook_region)
            
            # OCR识别
            recognized_text, ocr_confidence = self._ocr_text(processed)
            
            # 解析路书
            roadbook, parse_confidence = self._parse_roadbook_text(recognized_text, template_results)
            
            # 综合置信度
            confidence = (ocr_confidence * 0.4 + parse_confidence * 0.6)
            
            # 更新缓存
            self._update_cache(cache_key, (roadbook, confidence))
            
            # 更新历史记录
            self.recognition_history.append((roadbook, confidence))
            if len(self.recognition_history) > 100:  # 限制历史记录长度
                self.recognition_history.pop(0)
            
            # 更新最后识别结果
            self.last_roadbook = roadbook
            self.last_confidence = confidence
            
            return roadbook, confidence
        except Exception as e:
            logger.error(f"路书识别时出错: {e}")
            # 返回默认值而不是None，避免下游处理问题
            default_roadbook = {
                'direction': 'straight',
                'degree': 0,
                'speed': 0,
                'slope': 'flat',
                'special': None,
                'raw_text': ''
            }
            return default_roadbook, 0.0
            
            # 如果启用了模板匹配，也考虑它的置信度
            if self.config['use_template_matching']:
                template_conf = template_results.get('confidence', 0.5)
                confidence = (confidence * 0.8 + template_conf * 0.2)
            
            # 检查是否高于置信度阈值
            if confidence < self.config['confidence_threshold']:
                logger.warning(f"识别置信度低于阈值: {confidence:.2f} < {self.config['confidence_threshold']}")
                logger.warning(f"识别文本: '{recognized_text}'")
                
                # 使用历史值平滑
                if self.recognition_history:
                    # 使用最近3个识别结果的平均值
                    recent_results = self.recognition_history[-3:]
                    avg_roadbook = self._average_roadbook_results(recent_results)
                    return avg_roadbook, min(confidence + 0.1, 0.8)  # 稍微提高置信度
                
                return None, confidence
            
            # 更新缓存
            if self.config['enable_cache']:
                self._update_cache(cache_key, (roadbook, confidence))
            
            # 保存历史
            self.recognition_history.append((roadbook, confidence))
            if len(self.recognition_history) > 100:  # 保留最近100个结果
                self.recognition_history.pop(0)
            
            # 更新最后结果
            self.last_roadbook = roadbook
            self.last_confidence = confidence
            
            # 调试输出
            if self.config['debug']:
                logger.debug(f"识别路书: {roadbook}, 置信度: {confidence:.2f}")
            
            return roadbook, confidence
            
        except Exception as e:
            logger.error(f"识别路书时出错: {e}")
            return None, 0.0
    
    def _average_roadbook_results(self, results: List[Tuple[Dict, float]]) -> Dict:
        """
        对多个路书结果进行平均
        
        Args:
            results: 路书结果列表
            
        Returns:
            平均后的路书数据
        """
        if not results:
            return {
                'direction': 'straight',
                'degree': 0,
                'speed': 0,
                'slope': 'flat',
                'special': None,
                'raw_text': 'averaged'
            }
        
        # 计算加权平均
        avg_roadbook = {
            'direction': {},
            'degree_sum': 0,
            'degree_count': 0,
            'speed_sum': 0,
            'speed_count': 0,
            'slope': {},
            'special': {}
        }
        total_conf = sum(conf for _, conf in results)
        
        for roadbook, conf in results:
            # 方向投票（基于置信度加权）
            dir_weight = conf / total_conf
            direction = roadbook.get('direction', 'straight')
            avg_roadbook['direction'][direction] = avg_roadbook['direction'].get(direction, 0) + dir_weight
            
            # 度数加权平均
            if roadbook.get('degree', 0) > 0:
                avg_roadbook['degree_sum'] += roadbook['degree'] * conf
                avg_roadbook['degree_count'] += conf
            
            # 速度加权平均
            if roadbook.get('speed', 0) > 0:
                avg_roadbook['speed_sum'] += roadbook['speed'] * conf
                avg_roadbook['speed_count'] += conf
            
            # 坡度投票
            slope = roadbook.get('slope', 'flat')
            avg_roadbook['slope'][slope] = avg_roadbook['slope'].get(slope, 0) + dir_weight
            
            # 特殊路况
            special = roadbook.get('special', None)
            if special:
                avg_roadbook['special'][special] = avg_roadbook['special'].get(special, 0) + dir_weight
        
        # 构建最终结果
        result = {
            'direction': max(avg_roadbook['direction'].items(), key=lambda x: x[1])[0] if avg_roadbook['direction'] else 'straight',
            'degree': round(avg_roadbook['degree_sum'] / avg_roadbook['degree_count']) if avg_roadbook['degree_count'] > 0 else 0,
            'speed': round(avg_roadbook['speed_sum'] / avg_roadbook['speed_count']) if avg_roadbook['speed_count'] > 0 else 0,
            'slope': max(avg_roadbook['slope'].items(), key=lambda x: x[1])[0] if avg_roadbook['slope'] else 'flat',
            'special': max(avg_roadbook['special'].items(), key=lambda x: x[1])[0] if avg_roadbook['special'] else None,
            'raw_text': 'averaged_result'
        }
        
        return result
    
    def get_last_roadbook(self) -> Optional[Dict]:
        """
        获取最后一次识别的路书
        
        Returns:
            最后一次识别的路书数据
        """
        return self.last_roadbook
    
    def get_last_confidence(self) -> float:
        """
        获取最后一次识别的置信度
        
        Returns:
            最后一次识别的置信度
        """
        return self.last_confidence
    
    def get_average_confidence(self, window: int = 10) -> float:
        """
        获取最近几次识别的平均置信度
        
        Args:
            window: 窗口大小
            
        Returns:
            平均置信度
        """
        recent_results = self.recognition_history[-window:]
        if not recent_results:
            return 0.0
        
        return sum(conf for _, conf in recent_results) / len(recent_results)
    
    def save_roadbook_history(self, output_file: str) -> bool:
        """
        保存路书识别历史
        
        Args:
            output_file: 输出文件路径
            
        Returns:
            是否成功保存
        """
        try:
            import json
            
            # 准备历史数据
            history_data = []
            for roadbook, confidence in self.recognition_history:
                if roadbook:
                    history_item = roadbook.copy()
                    history_item['confidence'] = confidence
                    history_item['timestamp'] = datetime.now().isoformat()
                    history_data.append(history_item)
            
            # 保存到文件
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(history_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"路书历史已保存到: {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"保存路书历史时出错: {e}")
            return False
    
    def set_roadbook_region(self, region: Tuple[int, int, int, int]):
        """
        设置路书区域
        
        Args:
            region: 路书区域 [左, 上, 宽, 高]
        """
        self.config['roadbook_region'] = region
        logger.info(f"更新路书区域: {region}")
    
    def calibrate_roadbook_region(self, frame: np.ndarray) -> bool:
        """
        校准路书区域（简单版本）
        
        Args:
            frame: 包含路书的游戏画面帧
            
        Returns:
            是否成功校准
        """
        # 这个方法可以通过用户交互实现更精确的校准
        # 这里提供一个基础版本，实际应用中可能需要GUI支持
        logger.warning("自动校准路书区域功能尚未完全实现")
        return False

# 示例用法
def example_usage():
    # 导入必要的库
    from screen_capture import ScreenCapture
    import time
    
    # 创建屏幕捕获器
    capture = ScreenCapture()
    capture.start()
    
    # 创建路书识别器
    recognizer_config = {
        'tesseract_cmd': r'C:\Program Files\Tesseract-OCR\tesseract.exe',  # 根据实际安装路径修改
        'debug': True
    }
    recognizer = RoadbookRecognizer(recognizer_config)
    
    try:
        # 捕获并识别路书
        for i in range(10):
            frame = capture.get_latest_frame()
            if frame is not None:
                roadbook, confidence = recognizer.recognize_roadbook(frame)
                print(f"识别结果 (置信度: {confidence:.2f}): {roadbook}")
            time.sleep(1)
    finally:
        # 清理资源
        capture.stop()

if __name__ == "__main__":
    example_usage()