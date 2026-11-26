"""
Utility functions for MHPMR
"""

import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import Union, Tuple, List
import logging

logger = logging.getLogger(__name__)


def setup_logger(name: str, log_file: str = None, level=logging.INFO) -> logging.Logger:
    """Setup logger for the module"""
    logger_obj = logging.getLogger(name)
    logger_obj.setLevel(level)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger_obj.addHandler(console_handler)
    
    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger_obj.addHandler(file_handler)
    
    return logger_obj


def get_device_strategy(device: str = 'cuda:0'):
    """Get TensorFlow device strategy"""
    if device.startswith('cuda'):
        gpu_id = int(device.split(':')[1]) if ':' in device else 0
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                tf.config.set_visible_devices(gpus[gpu_id], 'GPU')
                return tf.distribute.OneDeviceStrategy(f'/gpu:{gpu_id}')
            except Exception as e:
                logger.warning(f"Failed to set GPU {gpu_id}: {e}")
    
    return tf.distribute.OneDeviceStrategy('/cpu:0')


def normalize_image(image: np.ndarray, mean: List[float] = None, std: List[float] = None) -> np.ndarray:
    """Normalize image using ImageNet statistics"""
    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std = [0.229, 0.224, 0.225]
    
    # Convert to float
    image = image.astype(np.float32) / 255.0
    
    # Normalize
    for i in range(3):
        image[:, :, i] = (image[:, :, i] - mean[i]) / std[i]
    
    return image


def denormalize_image(image: np.ndarray, mean: List[float] = None, std: List[float] = None) -> np.ndarray:
    """Denormalize image"""
    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std = [0.229, 0.224, 0.225]
    
    image = image.copy()
    for i in range(3):
        image[:, :, i] = image[:, :, i] * std[i] + mean[i]
    
    return np.clip(image * 255, 0, 255).astype(np.uint8)


def resize_image(image: np.ndarray, size: Union[int, Tuple[int, int]]) -> Tuple[np.ndarray, Tuple[float, float]]:
    """Resize image while maintaining aspect ratio"""
    import cv2
    
    if isinstance(size, int):
        size = (size, size)
    
    h, w = image.shape[:2]
    target_h, target_w = size
    
    # Calculate scale
    scale_h = target_h / h
    scale_w = target_w / w
    scale = min(scale_h, scale_w)
    
    # Resize
    new_h = int(h * scale)
    new_w = int(w * scale)
    resized = cv2.resize(image, (new_w, new_h))
    
    # Pad if needed
    if new_h < target_h or new_w < target_w:
        top = (target_h - new_h) // 2
        bottom = target_h - new_h - top
        left = (target_w - new_w) // 2
        right = target_w - new_w - left
        
        resized = cv2.copyMakeBorder(
            resized, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )
    
    return resized, (scale, (left if 'left' in locals() else 0, top if 'top' in locals() else 0))


def convert_to_tensor(data: Union[np.ndarray, List]) -> tf.Tensor:
    """Convert numpy array or list to TensorFlow tensor"""
    if isinstance(data, np.ndarray):
        return tf.convert_to_tensor(data)
    elif isinstance(data, list):
        return tf.convert_to_tensor(data)
    else:
        return data


def batch_data(data: List, batch_size: int):
    """Create batches from data"""
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]


def create_directory(path: Union[str, Path], exist_ok: bool = True) -> Path:
    """Create directory if it doesn't exist"""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=exist_ok)
    return path


def load_model_weights(model: tf.keras.Model, weights_path: str):
    """Load model weights from file"""
    if not Path(weights_path).exists():
        logger.warning(f"Weights file not found: {weights_path}")
        return
    
    try:
        model.load_weights(weights_path)
        logger.info(f"Loaded weights from {weights_path}")
    except Exception as e:
        logger.error(f"Failed to load weights: {e}")


def save_model_weights(model: tf.keras.Model, weights_path: str):
    """Save model weights to file"""
    create_directory(Path(weights_path).parent)
    model.save_weights(weights_path)
    logger.info(f"Saved weights to {weights_path}")


class AverageMeter:
    """Compute and store the average and current value"""
    
    def __init__(self, name: str = ""):
        self.name = name
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __str__(self):
        return f"{self.name}: {self.avg:.4f} (current: {self.val:.4f})"
