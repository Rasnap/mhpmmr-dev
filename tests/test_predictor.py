"""
Tests for core predictor module
"""

import pytest
import numpy as np
from mhpmr.predictor import PoseAndMeshPredictor


class TestPoseAndMeshPredictor:
    """Test PoseAndMeshPredictor class"""
    
    def test_initialization(self):
        """Test predictor initialization"""
        predictor = PoseAndMeshPredictor(device='cpu')
        assert predictor is not None
        assert predictor.device == 'cpu'
    
    def test_available_models(self):
        """Test listing available models"""
        models = PoseAndMeshPredictor.list_available_models()
        assert isinstance(models, list)
        assert len(models) > 0
    
    def test_supported_devices(self):
        """Test listing supported devices"""
        devices = PoseAndMeshPredictor.get_supported_devices()
        assert isinstance(devices, list)
        assert 'cpu' in devices
    
    def test_predict_not_implemented(self, sample_image):
        """Test that predict raises NotImplementedError"""
        predictor = PoseAndMeshPredictor(device='cpu')
        with pytest.raises(NotImplementedError):
            predictor.predict(sample_image)
    
    def test_predict_batch_not_implemented(self, sample_batch):
        """Test that predict_batch raises NotImplementedError"""
        predictor = PoseAndMeshPredictor(device='cpu')
        with pytest.raises(NotImplementedError):
            predictor.predict_batch([sample_batch])
