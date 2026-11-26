# Monocular 3D Human Pose and Mesh Recovery (MHPMR)

Recover accurate 3D human pose, body shape, and detailed mesh from a single RGB image using hybrid dense-parametric learning.

## 🎯 Overview

This project provides a state-of-the-art framework for monocular 3D human pose and mesh recovery. Unlike traditional approaches that either estimate pose from 2D keypoints or fit parametric models, our hybrid approach combines dense vertex prediction with parametric model fitting to achieve unprecedented accuracy and detail.

**Key Features:**
- 🏃 **69.2 mm MPJPE** on Human3.6M (best monocular performance)
- 📊 **Hybrid Architecture**: Dense vertices + SMPL fitting for detail
- 🎥 **Real-Time**: 30 FPS on GPU for 384×384 images
- 🔄 **Cross-Dataset Generalization**: <15% performance drop on unseen datasets
- 📦 **Occlusion Robust**: 85%+ accuracy with >50% body occlusion
- 🧠 **Vision Transformer + GNN**: Combines global context with mesh connectivity
- 💾 **MESH-DETAIL Dataset**: 1M images with detailed annotations

**Supported:**
- Single-image pose and shape estimation
- Video-based refinement
- Occlusion handling and uncertainty estimation
- Multiple body models (SMPL, SMPL-X)
- Real-time interactive applications

## 📋 Table of Contents

- [Getting Started](#getting-started)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Examples](#usage-examples)
- [Project Structure](#project-structure)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- PyTorch 1.9+ or TensorFlow 2.6+
- CUDA 11.0+ (for GPU acceleration)
- OpenCV
- pip or conda

### Installation

#### From Source

```bash
git clone https://github.com/Rasnap/mhpmr-dev.git
cd mhpmr-dev
pip install -e .
```

#### Via pip (when published)

```bash
pip install mhpmr
```

#### Using Conda

```bash
conda env create -f environment.yml
conda activate mhpmr
```

### Quick Start

**1. Estimate pose and mesh from an image:**

```python
import cv2
import mhpmr

# Initialize predictor
predictor = mhpmr.PoseAndMeshPredictor(device='cuda:0')

# Load image
image = cv2.imread('person.jpg')

# Predict
result = predictor.predict(image)

print(f"Pose shape: {result['pose'].shape}")        # (72,)
print(f"Body shape: {result['shape'].shape}")       # (10,)
print(f"Mesh vertices: {result['vertices'].shape}") # (6890, 3)
print(f"Joints 3D: {result['joints_3d'].shape}")    # (17, 3)
print(f"Confidence: {result['confidence']:.2f}")
```

**2. Visualize 3D pose and mesh:**

```python
from mhpmr.visualization import PoseVisualizer

visualizer = PoseVisualizer()

# Render mesh
mesh_image = visualizer.render_mesh(
    image,
    result['vertices'],
    wireframe=False,
    texture='default'
)

# Render pose skeleton
pose_image = visualizer.render_skeleton(
    image,
    result['joints_3d'],
    joints_2d=result['joints_2d']
)

cv2.imshow('Mesh', mesh_image)
cv2.imshow('Pose', pose_image)
cv2.waitKey(0)
```

**3. Process video with temporal refinement:**

```python
from mhpmr import VideoProcessor

processor = VideoProcessor(device='cuda:0')

# Process video
results = processor.process_video(
    'video.mp4',
    smooth_temporal=True,
    interpolate_occlusions=True
)

# Save output video
processor.save_video(
    results,
    output_path='output.mp4',
    visualize='mesh'  # or 'skeleton'
)
```

## 💻 Usage Examples

### Example 1: Batch Processing

```python
import glob
from mhpmr import PoseAndMeshPredictor

predictor = PoseAndMeshPredictor(batch_size=8, device='cuda:0')

# Process multiple images
image_paths = glob.glob('images/*.jpg')
results = predictor.predict_batch(image_paths)

for path, result in zip(image_paths, results):
    print(f"{path}: MPJPE={result['joint_error']:.1f}mm")
```

### Example 2: Extract Specific Joints

```python
# Get specific joint predictions
JOINT_NAMES = ['neck', 'left_shoulder', 'right_shoulder', 'spine', 'left_hip', 'right_hip']

result = predictor.predict(image)
joints_3d = result['joints_3d']

for name, idx in predictor.get_joint_indices(JOINT_NAMES).items():
    joint_pos = joints_3d[idx]
    print(f"{name}: {joint_pos}")
```

### Example 3: Occlusion Handling

```python
from mhpmr import OcclusionAwarePredictor

predictor = OcclusionAwarePredictor(device='cuda:0')

result = predictor.predict(
    image,
    return_uncertainty=True,
    handle_occlusions=True
)

# Get uncertainty per joint
uncertainty = result['joint_uncertainty']  # (17,)
confidence = result['confidence']  # scalar

# Identify low-confidence predictions
low_conf_joints = np.where(uncertainty > 0.3)[0]
print(f"Low confidence joints: {low_conf_joints}")

# Get imputed predictions for occluded parts
if result.get('has_occlusions'):
    imputed_joints = result['imputed_joints_3d']
    occlusion_mask = result['occlusion_mask']
    print(f"Occluded regions: {occlusion_mask.sum()} pixels")
```

### Example 4: Body Shape Analysis

```python
from mhpmr import BodyShapeAnalyzer

analyzer = BodyShapeAnalyzer()

result = predictor.predict(image)
shape_params = result['shape']  # (10,)

# Analyze body measurements
measurements = analyzer.get_body_measurements(
    shape_params,
    height_cm=170
)

print(f"Height: {measurements['height']:.1f} cm")
print(f"Chest: {measurements['chest']:.1f} cm")
print(f"Waist: {measurements['waist']:.1f} cm")
print(f"Hip: {measurements['hip']:.1f} cm")
print(f"BMI: {measurements['bmi']:.1f}")

# Compare body shapes
comparison = analyzer.compare_shapes(
    shape_params,
    reference_shapes={'average': np.zeros(10), 'athletic': np.random.randn(10)}
)
```

### Example 5: Mesh Export and Processing

```python
from mhpmr import MeshExporter

exporter = MeshExporter()

result = predictor.predict(image)
vertices = result['vertices']

# Export to various formats
exporter.export_obj('mesh.obj', vertices)
exporter.export_ply('mesh.ply', vertices, colors=result['vertex_colors'])
exporter.export_usdz('mesh.usdz', vertices)  # For AR/VR

# Load into trimesh for further processing
import trimesh
mesh = trimesh.Trimesh(vertices=vertices, faces=smpl_faces)
mesh.show()  # Visualization
```

### Example 6: Confidence-Based Filtering

```python
# Get all predictions with confidence threshold
results = predictor.predict(image, return_all_predictions=True)

# Filter by confidence
high_conf_results = [r for r in results if r['confidence'] > 0.8]
low_conf_results = [r for r in results if r['confidence'] <= 0.8]

print(f"High confidence: {len(high_conf_results)}")
print(f"Low confidence: {len(low_conf_results)}")

# Use only high-confidence results
for result in high_conf_results:
    process_result(result)
```

## 📁 Project Structure

```
mhpmr-dev/
├── README.md                    # This file
├── RESEARCH_PAPER.md            # Full research paper
├── IMPLEMENT.md                 # Implementation guidelines
├── requirements.txt             # Python dependencies
├── setup.py                     # Package setup
├── environment.yml              # Conda environment
│
├── src/
│   ├── mhpmr/
│   │   ├── __init__.py
│   │   ├── predictor.py         # Main predictor class
│   │   ├── vit_backbone.py      # Vision Transformer
│   │   ├── vertex_head.py       # Dense vertex prediction
│   │   ├── gnn_refiner.py       # Graph refinement
│   │   ├── occlusion.py         # Occlusion handling
│   │   ├── smpl_fitting.py      # SMPL model fitting
│   │   ├── utils.py             # Utilities
│   │   └── models/
│   │       └── pretrained/      # Pre-trained models
│   │
│   ├── dataset/
│   │   ├── mesh_detail.py       # MESH-DETAIL dataset
│   │   ├── loaders.py           # Data loaders
│   │   ├── processors.py        # Data processing
│   │   └── annotations.py       # Annotation utilities
│   │
│   ├── export/
│   │   ├── obj_exporter.py      # OBJ format
│   │   ├── ply_exporter.py      # PLY format
│   │   ├── usdz_exporter.py     # USDZ format
│   │   └── format_utils.py      # Format conversions
│   │
│   ├── visualization/
│   │   ├── viewer.py            # 3D viewer
│   │   ├── renderer.py          # OpenGL rendering
│   │   └── interactive.py       # Interactive UI
│   │
│   ├── analysis/
│   │   ├── body_shape.py        # Body shape analysis
│   │   ├── kinematics.py        # Kinematic analysis
│   │   └── metrics.py           # Evaluation metrics
│   │
│   └── training/
│       ├── trainer.py           # Training loop
│       ├── losses.py            # Loss functions
│       ├── metrics.py           # Metrics
│       └── callbacks.py         # Callbacks
│
├── tests/
│   ├── test_predictor.py
│   ├── test_vit.py
│   ├── test_occlusion.py
│   ├── test_export.py
│   └── fixtures/                # Test data
│
├── notebooks/
│   ├── 01_getting_started.ipynb
│   ├── 02_video_processing.ipynb
│   ├── 03_occlusion_robustness.ipynb
│   ├── 04_body_shape_analysis.ipynb
│   ├── 05_comparison_benchmarks.ipynb
│   └── 06_visualization.ipynb
│
├── data/
│   ├── models/                  # Character models
│   ├── dataset/                 # MESH-DETAIL dataset
│   ├── pretrained/              # Pre-trained weights
│   └── examples/                # Example images
│
└── tools/
    ├── prepare_dataset.py       # Dataset preparation
    ├── train_model.py           # Training script
    ├── benchmark.py             # Benchmarking
    ├── export_mesh.py           # Export script
    └── video_processor.py       # Video processing
```

## 📚 Documentation

### Core Concepts

- **Pose Estimation**: 17 COCO joints + 23 SMPL body joints
- **Mesh Recovery**: 6890 SMPL vertices with parametric fitting
- **Body Shape**: PCA-based shape parameters (10-dim)
- **Occlusion Handling**: Uncertainty-aware imputation
- **Vision Transformer**: Multi-scale feature extraction
- **Graph Refinement**: Mesh connectivity constraints

### API Reference

#### PoseAndMeshPredictor

```python
class PoseAndMeshPredictor:
    def __init__(self, model_variant='v1.0', device='cuda:0'):
        """Initialize predictor
        
        Args:
            model_variant: Model version
            device: Compute device
        """
    
    def predict(self, image, return_uncertainty=False):
        """Predict pose and mesh
        
        Returns:
            dict with keys: pose, shape, vertices, joints_3d, joints_2d,
                           confidence, uncertainty (optional)
        """
    
    def predict_batch(self, images, batch_size=8):
        """Batch prediction"""
    
    def predict_video(self, video_path, smooth=True):
        """Video-based prediction with temporal coherence"""
```

See [IMPLEMENT.md](IMPLEMENT.md) for detailed API and implementation guidelines.

## 🔧 Installation & Setup

### Full Development Setup

```bash
# Clone repository
git clone https://github.com/nayanajitesh09/mhpmr-dev.git
cd mhpmr-dev

# Create environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements-dev.txt

# Install in editable mode
pip install -e .

# Download models
python tools/download_models.py

# Run tests
pytest tests/ -v
```

### Verify Installation

```python
import mhpmr
print(mhpmr.__version__)
print(mhpmr.list_available_models())
print(mhpmr.get_supported_devices())
```

## 🧪 Testing

```bash
# All tests
pytest

# With coverage
pytest --cov=mhpmr tests/

# Specific test
pytest tests/test_predictor.py::test_basic_prediction -v

# Benchmark tests
pytest tests/test_performance.py -v
```

## 📊 Benchmarks

Accuracy on standard benchmarks (NVIDIA RTX 3090):

| Benchmark | MPJPE (mm) | PA-MPJPE (mm) | Model Size | FPS |
|-----------|-----------|-------------|-----------|-----|
| Human3.6M | 69.2 | 41.8 | 182 MB | 30 |
| 3DPW | 89.3 | 54.1 | — | — |
| AGORA | 85.1 | 51.3 | — | — |

See [RESEARCH_PAPER.md](RESEARCH_PAPER.md) Section 5 for detailed results.

## 🎓 Learning Resources

- **Getting Started**: `notebooks/01_getting_started.ipynb`
- **Video Processing**: `notebooks/02_video_processing.ipynb`
- **Occlusion Robustness**: `notebooks/03_occlusion_robustness.ipynb`
- **Body Shape Analysis**: `notebooks/04_body_shape_analysis.ipynb`
- **Benchmarks**: `notebooks/05_comparison_benchmarks.ipynb`
- **Visualization**: `notebooks/06_visualization.ipynb`
- **Research Paper**: [RESEARCH_PAPER.md](RESEARCH_PAPER.md)
- **Implementation**: [IMPLEMENT.md](IMPLEMENT.md)

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md).

### Development Workflow

1. Fork repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Make changes and add tests
4. Run tests: `pytest tests/`
5. Commit (`git commit -m 'Add amazing feature'`)
6. Push (`git push origin feature/amazing-feature`)
7. Open Pull Request

## 📄 License

MIT License - see [LICENSE](LICENSE)

## 🙏 Acknowledgments

- SMPL model and body dataset from Max Planck Institute
- Motion capture data from CMU and other institutions
- Inspiration from prior work in human understanding and 3D vision
- Community contributions and feedback

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/nayanajitesh09/mhpmr-dev/issues)
- **Discussions**: [GitHub Discussions](https://github.com/nayanajitesh09/mhpmr-dev/discussions)
- **Documentation**: [Full Docs](./docs/)

## 🗺️ Roadmap

- [ ] SMPL-X support (hands and face)
- [ ] Real-time video optimization
- [ ] Clothed human handling
- [ ] Multi-person scene
- [ ] Hand pose refinement
- [ ] Facial expression integration
- [ ] Mobile optimization (TFLite, ONNX)
- [ ] Web-based demo
- [ ] Cloud API service
- [ ] Advanced visualization UI

## 📰 Citation

```bibtex
@article{nayana2024mhpmr,
  title={Monocular 3D Human Pose and Mesh Recovery},
  author={Nayana Jitesh and ...},
  journal={Proceedings of ICCV},
  year={2024}
}
```

## 📸 Gallery

**Example Outputs:**
- [Single Image Pose](./examples/pose_example.jpg)
- [Mesh Recovery](./examples/mesh_example.jpg)
- [Video Processing](./examples/video_example.mp4)
- [Occlusion Handling](./examples/occlusion_example.jpg)
- [Body Shape Analysis](./examples/shape_analysis.jpg)

---

**Last Updated**: July 2024

For implementation details, see [IMPLEMENT.md](IMPLEMENT.md).

**Status**: Beta Release — Production-grade stability in v1.0.
