"""
Test configuration and fixtures
"""

import pytest
import numpy as np
import tensorflow as tf


@pytest.fixture
def sample_image():
    """Generate sample test image"""
    return np.random.randint(0, 256, (384, 384, 3), dtype=np.uint8)


@pytest.fixture
def sample_batch():
    """Generate sample batch of images"""
    return np.random.randint(0, 256, (4, 384, 384, 3), dtype=np.uint8)


@pytest.fixture
def sample_joints_3d():
    """Generate sample 3D joint positions"""
    return np.random.randn(17, 3) * 100  # Random 3D positions


@pytest.fixture
def sample_vertices():
    """Generate sample mesh vertices"""
    return np.random.randn(6890, 3)


@pytest.fixture
def sample_shape():
    """Generate sample shape parameters"""
    return np.random.randn(10)


@pytest.fixture
def sample_pose():
    """Generate sample pose parameters"""
    return np.random.randn(72)


@pytest.fixture
def tf_dataset():
    """Create sample TensorFlow dataset"""
    images = np.random.randint(0, 256, (10, 384, 384, 3), dtype=np.uint8)
    dataset = tf.data.Dataset.from_tensor_slices(images).batch(2)
    return dataset
