"""
Tests for utilities module
"""

import pytest
import numpy as np
from pathlib import Path
from mhpmr.utils import (
    normalize_image, 
    denormalize_image,
    AverageMeter,
    create_directory
)


class TestImageProcessing:
    """Test image processing utilities"""
    
    def test_normalize_image(self):
        """Test image normalization"""
        image = np.ones((384, 384, 3), dtype=np.uint8) * 128
        normalized = normalize_image(image)
        
        assert normalized.dtype == np.float32
        assert normalized.shape == (384, 384, 3)
        assert normalized.min() >= -2.0
        assert normalized.max() <= 2.0
    
    def test_denormalize_image(self):
        """Test image denormalization"""
        normalized = np.random.randn(384, 384, 3).astype(np.float32) * 0.5
        denormalized = denormalize_image(normalized)
        
        assert denormalized.dtype == np.uint8
        assert denormalized.min() >= 0
        assert denormalized.max() <= 255


class TestAverageMeter:
    """Test AverageMeter utility"""
    
    def test_initialization(self):
        """Test AverageMeter initialization"""
        meter = AverageMeter("loss")
        assert meter.name == "loss"
        assert meter.avg == 0
    
    def test_update(self):
        """Test meter update"""
        meter = AverageMeter("loss")
        meter.update(10.0)
        meter.update(20.0)
        
        assert meter.avg == 15.0
        assert meter.sum == 30.0
        assert meter.count == 2
    
    def test_string_representation(self):
        """Test string representation"""
        meter = AverageMeter("loss")
        meter.update(5.0)
        
        str_repr = str(meter)
        assert "loss" in str_repr
        assert "5.0000" in str_repr


class TestDirectoryUtils:
    """Test directory utilities"""
    
    def test_create_directory(self, tmp_path):
        """Test directory creation"""
        test_dir = tmp_path / "test" / "nested" / "dir"
        created = create_directory(test_dir)
        
        assert created.exists()
        assert created.is_dir()
