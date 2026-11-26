"""
Configuration management for MHPMR
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import yaml
import os


@dataclass
class ModelConfig:
    """Configuration for model architecture"""
    
    # Vision Transformer backbone
    vit_model: str = "base"  # 'tiny', 'small', 'base', 'large'
    vit_pretrained: bool = True
    vit_freeze_backbone: bool = False
    
    # Input/Output dimensions
    input_size: int = 384
    num_joints: int = 17  # COCO joints
    num_vertices: int = 6890  # SMPL vertices
    num_shape_params: int = 10  # SMPL shape parameters
    num_pose_params: int = 72  # SMPL pose parameters (24 * 3)
    
    # Vertex head
    vertex_head_channels: List[int] = field(default_factory=lambda: [256, 128])
    
    # GNN Refiner
    gnn_num_layers: int = 3
    gnn_hidden_dim: int = 128
    gnn_dropout: float = 0.1
    
    # Occlusion handling
    use_occlusion_handling: bool = True
    occlusion_threshold: float = 0.5
    
    # Training
    use_mixed_precision: bool = True
    

@dataclass
class TrainingConfig:
    """Configuration for training"""
    
    # Batch size and learning
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    num_epochs: int = 100
    warmup_epochs: int = 5
    
    # Loss weights
    loss_weights: Dict[str, float] = field(default_factory=lambda: {
        "keypoint_2d": 1.0,
        "keypoint_3d": 1.0,
        "vertex": 0.5,
        "shape": 0.1,
        "pose_prior": 0.1,
        "occlusion": 0.1,
    })
    
    # Data augmentation
    augmentation: bool = True
    augmentation_prob: float = 0.5
    
    # Validation and checkpointing
    validation_split: float = 0.1
    checkpoint_interval: int = 5  # epochs
    early_stopping_patience: int = 20
    

@dataclass
class InferenceConfig:
    """Configuration for inference"""
    
    device: str = "cuda:0"
    batch_size: int = 1
    return_uncertainty: bool = True
    return_all_predictions: bool = False
    confidence_threshold: float = 0.5
    

@dataclass
class DataConfig:
    """Configuration for data loading"""
    
    # Dataset paths
    dataset_root: str = "./data/dataset"
    cache_dir: str = "./data/cache"
    
    # Data processing
    num_workers: int = 4
    prefetch_buffer_size: int = 32
    
    # Normalization
    image_mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    image_std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])
    

@dataclass
class VisualizationConfig:
    """Configuration for visualization"""
    
    mesh_color: tuple = (0.8, 0.8, 0.8)
    skeleton_color: tuple = (1.0, 0.0, 0.0)
    joint_radius: int = 3
    edge_width: int = 2
    

class Config:
    """Main configuration class"""
    
    def __init__(self):
        self.model = ModelConfig()
        self.training = TrainingConfig()
        self.inference = InferenceConfig()
        self.data = DataConfig()
        self.visualization = VisualizationConfig()
    
    @staticmethod
    def from_yaml(config_path: str) -> 'Config':
        """Load configuration from YAML file"""
        config = Config()
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            yaml_dict = yaml.safe_load(f)
        
        # Update config from YAML
        for key, value in yaml_dict.items():
            if hasattr(config, key):
                setattr(getattr(config, key), **value)
        
        return config
    
    def to_yaml(self, config_path: str) -> None:
        """Save configuration to YAML file"""
        config_dict = {
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'inference': self.inference.__dict__,
            'data': self.data.__dict__,
            'visualization': self.visualization.__dict__,
        }
        
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)


# Global default config
DEFAULT_CONFIG = Config()
