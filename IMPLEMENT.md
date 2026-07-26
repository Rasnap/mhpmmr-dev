# Implementation Guidelines for MHPMR

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Core Components](#core-components)
3. [Implementation Details](#implementation-details)
4. [Data Pipeline](#data-pipeline)
5. [Model Training](#model-training)
6. [Evaluation](#evaluation)
7. [Deployment](#deployment)
8. [Troubleshooting](#troubleshooting)

---

## Architecture Overview

### System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      User Interface Layer                        │
│  (PoseAndMeshPredictor, VideoProcessor, MeshExporter)           │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│                   Inference Pipeline                             │
│  ┌────────────────┬──────────────┬──────────────────────────┐   │
│  │ Image Preproc  │ Feature Extr. │ Prediction & Refinement │   │
│  └────────────────┴──────────────┴──────────────────────────┘   │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│                      Core Models                                 │
│  ┌──────────────┬──────────────┬────────────────────────────┐   │
│  │ ViT Backbone │ Vertex Head  │ GNN Refiner               │   │
│  │              │              │ + Occlusion Handler       │   │
│  └──────────────┴──────────────┴────────────────────────────┘   │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│                   Post-Processing                                │
│  ┌───────────────────┬───────────────┬────────────────────────┐ │
│  │ SMPL Fitting      │ Kinematics    │ Output Formatting      │ │
│  └───────────────────┴───────────────┴────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
Input Image (H × W × 3)
    ↓
Preprocessing (resize to 384×384, normalize)
    ↓
Vision Transformer Backbone
    ↓
Multi-Scale Feature Pyramid
    ↓
Parallel Heads:
  ├─ 2D Keypoint Detection
  ├─ Dense Vertex Prediction
  └─ Uncertainty Estimation
    ↓
Graph Neural Network Refinement
    ↓
Occlusion Detection & Handling
    ↓
SMPL Fitting & Parameter Extraction
    ↓
Output: Pose, Shape, Mesh, Confidence
```

---

## Core Components

### 1. Vision Transformer Backbone (`vit_backbone.py`)

**Purpose**: Extract multi-scale features from input image

```python
class ViTBackbone(torch.nn.Module):
    def __init__(self, model_size='base', pretrained=True):
        """
        Args:
            model_size: 'tiny', 'small', 'base', 'large'
            pretrained: Load ImageNet pretrained weights
        """
        super().__init__()
        
        # Vision Transformer configuration
        configs = {
            'tiny': {'depth': 12, 'embed_dim': 192, 'num_heads': 3},
            'small': {'depth': 12, 'embed_dim': 384, 'num_heads': 6},
            'base': {'depth': 12, 'embed_dim': 768, 'num_heads': 12},
            'large': {'depth': 24, 'embed_dim': 1024, 'num_heads': 16}
        }
        
        config = configs[model_size]
        
        # Patch embedding
        self.patch_embed = torch.nn.Conv2d(3, config['embed_dim'], 16, 16)
        self.cls_token = torch.nn.Parameter(torch.randn(1, 1, config['embed_dim']))
        
        # Transformer
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=config['embed_dim'],
            nhead=config['num_heads'],
            dim_feedforward=config['embed_dim'] * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = torch.nn.TransformerEncoder(
            encoder_layer,
            num_layers=config['depth']
        )
        
        # Feature pyramid
        self.pyramid = FeaturePyramid(config['embed_dim'])
    
    def forward(self, image):
        """
        Args:
            image: (B, 3, 384, 384)
        
        Returns:
            features: List of 4 scales [(B, C, H/4, W/4), ..., (B, C, H, W)]
        """
        # Patch embedding
        x = self.patch_embed(image)  # (B, dim, 24, 24)
        B, C, H, W = x.shape
        x = x.reshape(B, C, -1).transpose(1, 2)  # (B, num_patches, dim)
        
        # Add class token
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        
        # Transformer
        x = self.transformer(x)
        
        # Remove class token and reshape
        x = x[:, 1:, :].transpose(1, 2).reshape(B, C, H, W)
        
        # Feature pyramid
        pyramid = self.pyramid(x)
        return pyramid
```

### 2. Dense Vertex Prediction Head (`vertex_head.py`)

**Purpose**: Predict 3D vertex positions from features

```python
class DenseVertexHead(torch.nn.Module):
    def __init__(self, in_channels=768, num_vertices=6890):
        super().__init__()
        
        self.num_vertices = num_vertices
        
        # Upsampling path
        self.upsample_layers = torch.nn.ModuleList([
            UpSampleBlock(768, 512),   # 24→48
            UpSampleBlock(512, 256),   # 48→96
            UpSampleBlock(256, 128),   # 96→192
            UpSampleBlock(128, 64)     # 192→384
        ])
        
        # Vertex regression
        self.vertex_regressor = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, 3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 32, 3, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 3, 1)  # xyz
        )
        
        # Uncertainty estimation
        self.uncertainty_head = torch.nn.Sequential(
            torch.nn.Conv2d(64, 32, 3, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 1, 1)  # uncertainty
        )
        
        # 2D keypoint head
        self.keypoint_head = torch.nn.Sequential(
            torch.nn.Conv2d(64, 32, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 17, 1)  # 17 COCO joints
        )
    
    def forward(self, pyramid_features):
        """
        Args:
            pyramid_features: Multi-scale features from backbone
        
        Returns:
            vertices_3d: (B, 3, H, W) 3D vertex positions
            uncertainty: (B, 1, H, W) prediction uncertainty
            keypoints_2d: (B, 17, H, W) 2D keypoint heatmaps
        """
        x = pyramid_features[0]  # Coarsest feature
        
        # Progressive upsampling
        for layer in self.upsample_layers:
            x = layer(x)
        
        # Vertex regression
        vertices_3d = self.vertex_regressor(x)  # (B, 3, 384, 384)
        
        # Uncertainty
        uncertainty = torch.sigmoid(self.uncertainty_head(x))
        
        # Keypoints
        keypoints_2d = self.keypoint_head(x)
        
        return vertices_3d, uncertainty, keypoints_2d

class UpSampleBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, 3, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU()
        )
    
    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        return x
```

### 3. Graph Neural Network Refiner (`gnn_refiner.py`)

**Purpose**: Refine mesh predictions using connectivity constraints

```python
class MeshRefinementGNN(torch.nn.Module):
    def __init__(self, num_vertices=6890, num_layers=3):
        super().__init__()
        
        # Load SMPL edges
        self.register_buffer('edges', self._load_smpl_edges())
        self.register_buffer('edge_indices', self._get_edge_indices())
        
        # GNN layers
        self.gnn_layers = torch.nn.ModuleList([
            GraphConvLayer(in_channels=3 + 768, out_channels=256)
            for _ in range(num_layers)
        ])
        
        # Output head
        self.output_head = torch.nn.Sequential(
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(128, 3)  # 3D offset
        )
    
    def forward(self, vertices, vertex_features):
        """
        Args:
            vertices: (B, num_vertices, 3) predicted vertices
            vertex_features: (B, num_vertices, 768) features per vertex
        
        Returns:
            refined_vertices: (B, num_vertices, 3) refined positions
        """
        # Combine position and features
        node_features = torch.cat([vertices, vertex_features], dim=-1)
        
        # Iterative refinement
        for layer in self.gnn_layers:
            node_features = layer(node_features, self.edge_indices)
            node_features = torch.nn.functional.relu(node_features)
        
        # Predict offset
        offset = self.output_head(node_features)
        
        # Refine vertices
        refined = vertices + offset
        return refined
    
    @staticmethod
    def _load_smpl_edges():
        """Load SMPL mesh connectivity"""
        # This should load from SMPL model file
        pass
    
    def _get_edge_indices(self):
        """Convert edges to PyTorch format (2, num_edges)"""
        edge_indices = torch.stack([
            self.edges[:, 0],
            self.edges[:, 1]
        ], dim=0)
        return edge_indices

class GraphConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.linear_self = torch.nn.Linear(in_channels, out_channels)
        self.linear_neighbor = torch.nn.Linear(in_channels, out_channels)
        self.layer_norm = torch.nn.LayerNorm(out_channels)
    
    def forward(self, node_features, edge_indices):
        """Message passing on graph"""
        # Self transformation
        self_feat = self.linear_self(node_features)
        
        # Neighbor aggregation
        neighbor_feat = torch.zeros_like(self_feat)
        for src, dst in edge_indices.t():
            neighbor_feat[dst] += node_features[src]
        neighbor_feat = self.linear_neighbor(neighbor_feat)
        
        # Combine and normalize
        combined = self_feat + neighbor_feat
        return self.layer_norm(combined)
```

### 4. Occlusion Handler (`occlusion.py`)

**Purpose**: Handle occluded regions through imputation

```python
class OcclusionHandler(torch.nn.Module):
    def __init__(self, in_channels=768):
        super().__init__()
        
        # Occlusion detector
        self.occlusion_detector = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, 128, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 1, 1)
        )
        
        # Feature inpainting network (partial convolution)
        self.inpainter = PartialConvolution(in_channels, in_channels)
        
        # Confidence weighting
        self.confidence_weights = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels + 1, 64, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 1, 1)
        )
    
    def forward(self, features, vertices, uncertainty):
        """
        Args:
            features: (B, C, H, W) image features
            vertices: (B, 6890, 3) predicted vertices
            uncertainty: (B, H, W) prediction uncertainty
        
        Returns:
            has_occlusion: Boolean mask
            imputed_vertices: Vertices with imputation
            occlusion_mask: Region mask
        """
        # Detect occlusions
        occlusion_logits = self.occlusion_detector(features)
        occlusion_prob = torch.sigmoid(occlusion_logits)
        
        # Identify occluded regions
        is_occluded = (uncertainty < 0.3) | (occlusion_prob > 0.5)
        
        # Inpaint features
        mask = (~is_occluded).float()  # Invert for inpainting
        inpainted_features = self.inpainter(features, mask)
        
        # Recompute vertices for imputed regions
        imputed_vertices = vertices.clone()
        if is_occluded.any():
            # Use imputed features to improve predictions
            # (implementation depends on integration)
            pass
        
        return is_occluded.any().item(), imputed_vertices, is_occluded
```

### 5. SMPL Fitting (`smpl_fitting.py`)

**Purpose**: Fit parametric SMPL model to predicted vertices

```python
def fit_smpl_to_vertices(vertices_pred, smpl_model, init_params=None, num_iter=50):
    """
    Fit SMPL parameters to predicted vertex positions
    
    Args:
        vertices_pred: (num_vertices, 3) predicted vertices
        smpl_model: SMPL model object
        init_params: Initial parameters {'pose': ..., 'shape': ..., 'trans': ...}
        num_iter: Optimization iterations
    
    Returns:
        params: Fitted SMPL parameters
        fit_error: Reconstruction error
    """
    
    # Initialize parameters
    if init_params is None:
        pose = torch.zeros(72)
        shape = torch.zeros(10)
        trans = torch.zeros(3)
    else:
        pose = init_params['pose'].clone()
        shape = init_params['shape'].clone()
        trans = init_params['trans'].clone()
    
    # Make differentiable for optimization
    pose.requires_grad = True
    shape.requires_grad = True
    trans.requires_grad = True
    
    optimizer = torch.optim.Adam([pose, shape, trans], lr=0.01)
    
    for iter in range(num_iter):
        # Forward SMPL
        smpl_out = smpl_model(
            betas=shape,
            body_pose=pose[3:],
            global_orient=pose[:3],
            transl=trans
        )
        vertices_smpl = smpl_out.vertices.squeeze()
        
        # Fitting loss
        loss_vertex = torch.nn.functional.mse_loss(vertices_smpl, vertices_pred)
        
        # Regularization
        loss_pose_reg = 0.001 * torch.sum(pose[3:] ** 2)
        loss_shape_reg = 0.001 * torch.sum(shape ** 2)
        
        total_loss = loss_vertex + loss_pose_reg + loss_shape_reg
        
        # Backward
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
    
    # Compute final error
    with torch.no_grad():
        smpl_out = smpl_model(betas=shape, body_pose=pose[3:], global_orient=pose[:3], transl=trans)
        fit_error = torch.nn.functional.mse_loss(smpl_out.vertices.squeeze(), vertices_pred).item()
    
    return {
        'pose': pose.detach(),
        'shape': shape.detach(),
        'trans': trans.detach(),
        'fit_error': fit_error
    }
```

---

## Implementation Details

### 1. PoseAndMeshPredictor Main Class

```python
class PoseAndMeshPredictor:
    def __init__(self, device='cuda:0', model_variant='v1.0', model_size='base'):
        self.device = device
        self.model_variant = model_variant
        
        # Load models
        self.vit_backbone = ViTBackbone(model_size=model_size).to(device)
        self.vertex_head = DenseVertexHead().to(device)
        self.gnn_refiner = MeshRefinementGNN().to(device)
        self.occlusion_handler = OcclusionHandler().to(device)
        self.smpl_model = SMPL(model_path='smpl_model.pkl').to(device)
        
        # Load pretrained weights
        self._load_pretrained(model_variant)
        
        # Set to eval mode
        for model in [self.vit_backbone, self.vertex_head, self.gnn_refiner, self.occlusion_handler]:
            model.eval()
    
    def predict(self, image, return_uncertainty=False, return_confidence=True):
        """Single image prediction"""
        
        # Preprocess
        image_tensor = self._preprocess(image)
        
        with torch.no_grad():
            # Backbone
            pyramid_features = self.vit_backbone(image_tensor)
            
            # Vertex prediction
            vertices_3d, uncertainty, keypoints_2d = self.vertex_head(pyramid_features)
            
            # Extract vertex features per pixel
            vertex_features = pyramid_features[-1]  # High-res features
            
            # Map vertices to image space
            vertices_flat = self._map_vertices_to_image(vertices_3d)
            
            # Occlusion handling
            has_occl, vertices_imputed, occl_mask = self.occlusion_handler(
                pyramid_features[-1],
                vertices_flat,
                uncertainty.squeeze()
            )
            
            # GNN refinement
            vertices_refined = self.gnn_refiner(
                vertices_flat.unsqueeze(0),
                vertex_features.unsqueeze(0)
            ).squeeze()
            
            # SMPL fitting
            smpl_params = fit_smpl_to_vertices(vertices_refined, self.smpl_model)
            
            # Compute SMPL output
            smpl_out = self.smpl_model(
                betas=smpl_params['shape'],
                body_pose=smpl_params['pose'][3:],
                global_orient=smpl_params['pose'][:3],
                transl=smpl_params['trans']
            )
            
            # Extract joints
            joints_3d = smpl_out.joints.squeeze()
            joints_2d = self._project_joints_2d(joints_3d)
            
            # Compute confidence
            confidence = self._compute_confidence(uncertainty, has_occl)
        
        result = {
            'pose': smpl_params['pose'].cpu().numpy(),
            'shape': smpl_params['shape'].cpu().numpy(),
            'trans': smpl_params['trans'].cpu().numpy(),
            'vertices': smpl_out.vertices.squeeze().cpu().numpy(),
            'joints_3d': joints_3d.cpu().numpy(),
            'joints_2d': joints_2d.cpu().numpy(),
            'confidence': confidence
        }
        
        if return_uncertainty:
            result['uncertainty'] = uncertainty.squeeze().cpu().numpy()
        
        return result
    
    def predict_batch(self, images, batch_size=8):
        """Batch prediction"""
        results = []
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            batch_results = self._predict_batch(batch)
            results.extend(batch_results)
        return results
    
    def predict_video(self, video_path, smooth=True, interpolate_occlusions=True):
        """Video prediction with temporal consistency"""
        
        cap = cv2.VideoCapture(video_path)
        results = []
        prev_params = None
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Predict
            result = self.predict(frame)
            
            # Temporal smoothing
            if smooth and prev_params is not None:
                result = self._smooth_temporal(result, prev_params, alpha=0.7)
            
            results.append(result)
            prev_params = result
        
        cap.release()
        return results
    
    def _preprocess(self, image):
        """Preprocess image"""
        if isinstance(image, str):
            image = cv2.imread(image)
        
        # Resize to 384×384
        image_resized = cv2.resize(image, (384, 384))
        
        # Normalize
        image_tensor = torch.from_numpy(image_resized).float() / 255.0
        image_tensor = image_tensor.permute(2, 0, 1)  # HWC→CHW
        
        # Standardize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image_tensor = (image_tensor - mean) / std
        
        return image_tensor.unsqueeze(0).to(self.device)
    
    def _compute_confidence(self, uncertainty, has_occlusion):
        """Compute overall confidence score"""
        mean_uncertainty = uncertainty.mean().item()
        confidence = 1.0 - mean_uncertainty
        
        if has_occlusion:
            confidence *= 0.9  # Slight penalty for occlusions
        
        return confidence
```

---

## Data Pipeline

### Step 1: Data Loading

```python
from mhpmr.dataset import MeshDetailDataset

dataset = MeshDetailDataset(
    data_dir='./data/mesh_detail',
    split='train',
    transforms=get_transforms()
)

train_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
    num_workers=8,
    pin_memory=True
)
```

### Step 2: Data Preprocessing

Standardize images and annotations for training.

### Step 3: Training Loop

See Section 5 below.

---

## Model Training

### Training Loop

```python
def train_epoch(model, train_loader, optimizer, device):
    """Train for one epoch"""
    
    losses = {'vertex': 0, 'keypoint': 0, 'uncertainty': 0, 'smooth': 0}
    
    for batch_idx, batch in enumerate(train_loader):
        images = batch['image'].to(device)
        gt_vertices = batch['vertices'].to(device)
        gt_joints_2d = batch['joints_2d'].to(device)
        
        # Forward
        pyramid_features = model.vit_backbone(images)
        vertices_pred, uncertainty, keypoints_2d = model.vertex_head(pyramid_features)
        
        # Loss
        loss_vertex = torch.nn.functional.mse_loss(vertices_pred, gt_vertices)
        loss_keypoint = torch.nn.functional.mse_loss(keypoints_2d, gt_joints_2d)
        loss_uncertainty = -uncertainty.mean() * torch.nn.functional.mse_loss(vertices_pred, gt_vertices)
        loss_smooth = torch.nn.functional.mse_loss(
            vertices_pred[1:] - vertices_pred[:-1],
            gt_vertices[1:] - gt_vertices[:-1]
        )
        
        total_loss = (
            1.0 * loss_vertex +
            0.1 * loss_keypoint +
            0.05 * loss_uncertainty +
            0.01 * loss_smooth
        )
        
        # Backward
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Logging
        losses['vertex'] += loss_vertex.item()
        losses['keypoint'] += loss_keypoint.item()
        losses['uncertainty'] += loss_uncertainty.item()
        losses['smooth'] += loss_smooth.item()
    
    # Average losses
    for key in losses:
        losses[key] /= len(train_loader)
    
    return losses
```

### Hyperparameter Recommendations

| Parameter | Value | Range |
|-----------|-------|-------|
| Learning Rate | 0.0001 | 1e-5 to 1e-3 |
| Batch Size | 64 | 32-128 |
| Model Size | base | tiny, small, base, large |
| Weight Decay | 0.0001 | 0-1e-3 |
| Dropout | 0.1 | 0-0.3 |
| Epochs | 100 | 50-200 |

---

## Evaluation

### Metrics

```python
def compute_mpjpe(pred_joints, gt_joints):
    """Mean Per-Joint Position Error"""
    return torch.norm(pred_joints - gt_joints, dim=-1).mean().item()

def compute_pa_mpjpe(pred_joints, gt_joints):
    """Procrustes-Aligned MPJPE"""
    # Align predictions to ground truth via Procrustes analysis
    # Then compute MPJPE
    pass

def compute_mesh_error(pred_vertices, gt_vertices):
    """Mean vertex position error"""
    return torch.norm(pred_vertices - gt_vertices, dim=-1).mean().item()
```

---

## Deployment

### ONNX Export

```python
model.eval()
dummy_input = torch.randn(1, 3, 384, 384)

torch.onnx.export(
    model,
    dummy_input,
    'mhpmr.onnx',
    input_names=['image'],
    output_names=['pose', 'shape', 'vertices'],
    dynamic_axes={'image': {0: 'batch_size'}}
)
```

### TensorFlow Export

```python
# Convert from PyTorch to TensorFlow
from onnx_tf.backend import prepare
onnx_model = onnx.load('mhpmr.onnx')
tf_rep = prepare(onnx_model)
tf_rep.export_graph('mhpmr_tf')
```

### Mobile Deployment

```python
# Quantization for mobile
import torch.quantization

model_int8 = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},
    dtype=torch.qint8
)

torch.jit.save(torch.jit.script(model_int8), 'mhpmr_mobile.pt')
```

---

## Troubleshooting

### Issue 1: Poor Mesh Quality

**Solutions**:
1. Increase GNN refinement iterations
2. Improve training data quality
3. Increase model capacity
4. Check SMPL fitting convergence

### Issue 2: High Occlusion Errors

**Solutions**:
1. Improve occlusion detection
2. Train with more occluded examples
3. Use better imputation network
4. Increase uncertainty weighting

### Issue 3: Slow Inference

**Solutions**:
1. Use smaller ViT variant
2. Reduce feature resolution
3. Use quantization or pruning
4. Batch inference requests

### Issue 4: Poor Generalization

**Solutions**:
1. Expand training data diversity
2. Use data augmentation
3. Add regularization
4. Train on multiple datasets

---

**For additional support, see README.md and RESEARCH_PAPER.md**
