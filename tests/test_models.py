"""
Tests for model components
"""

import pytest
import tensorflow as tf
from mhpmr.models import ViTBackbone, VertexHead, GNNRefiner


class TestViTBackbone:
    """Test Vision Transformer backbone"""
    
    @pytest.mark.parametrize("model_size", ["tiny", "small", "base", "large"])
    def test_initialization(self, model_size):
        """Test ViT backbone initialization"""
        backbone = ViTBackbone(model_size=model_size)
        assert backbone is not None
        assert backbone.model_size == model_size
    
    def test_config(self):
        """Test ViT configuration"""
        backbone = ViTBackbone(model_size="base")
        config = backbone.get_config()
        assert config['model_size'] == 'base'


class TestVertexHead:
    """Test vertex prediction head"""
    
    def test_initialization(self):
        """Test VertexHead initialization"""
        head = VertexHead(num_vertices=6890)
        assert head is not None
        assert head.num_vertices == 6890


class TestGNNRefiner:
    """Test GNN refiner"""
    
    def test_initialization(self):
        """Test GNNRefiner initialization"""
        refiner = GNNRefiner(num_vertices=6890, num_layers=3)
        assert refiner is not None
        assert refiner.num_vertices == 6890
        assert refiner.num_layers == 3
