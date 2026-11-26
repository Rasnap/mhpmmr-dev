"""
MHPMR - Monocular 3D Human Pose and Mesh Recovery

A framework for recovering accurate 3D human pose, body shape, and detailed mesh
from a single RGB image using hybrid dense-parametric learning.
"""

__version__ = "0.1.0"
__author__ = "Nayana Jitesh"
__license__ = "MIT"

from .predictor import PoseAndMeshPredictor
from .video import VideoProcessor
from .export import MeshExporter

__all__ = [
    "PoseAndMeshPredictor",
    "VideoProcessor",
    "MeshExporter",
]
