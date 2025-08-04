"""Unit tests for Ising model implementation."""

import pytest
import numpy as np
import torch
from unittest.mock import Mock, patch

from spin_glass_rl.core.ising_model import IsingModel, IsingModelConfig
from spin_glass_rl.utils.exceptions import ModelError, ValidationError


class TestIsingModelConfig:
    """Test IsingModelConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = IsingModelConfig(n_spins=10)
        
        assert config.n_spins == 10
        assert config.coupling_strength == 1.0
        assert config.external_field_strength == 0.5
        assert config.use_sparse == True
        assert config.device == "cpu"
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = IsingModelConfig(
            n_spins=50,
            coupling_strength=2.0,
            external_field_strength=1.0,
            use_sparse=False,
            device="cuda"
        )
        
        assert config.n_spins == 50
        assert config.coupling_strength == 2.0
        assert config.external_field_strength == 1.0
        assert config.use_sparse == False
        assert config.device == "cuda"


class TestIsingModel:
    """Test IsingModel class."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return IsingModelConfig(n_spins=10, device="cpu")
    
    @pytest.fixture
    def model(self, config):
        """Create test Ising model."""
        return IsingModel(config)
    
    def test_initialization(self, config):
        """Test model initialization."""
        model = IsingModel(config)
        
        assert model.n_spins == 10
        assert model.device.type == "cpu"
        assert model.spins.shape == (10,)
        assert torch.all(torch.abs(model.spins) == 1)  # Spins should be ±1
        assert model.external_fields.shape == (10,)
        assert torch.all(model.external_fields == 0)
    
    def test_set_coupling(self, model):
        """Test setting individual coupling."""
        model.set_coupling(0, 1, 2.5)
        
        if model.config.use_sparse:
            coupling_value = model.couplings.to_dense()[0, 1].item()
        else:
            coupling_value = model.couplings[0, 1].item()
        
        assert coupling_value == 2.5
        
        # Check symmetry
        if model.config.use_sparse:
            coupling_value = model.couplings.to_dense()[1, 0].item()
        else:
            coupling_value = model.couplings[1, 0].item()
        
        assert coupling_value == 2.5
    
    def test_set_external_field(self, model):
        """Test setting external field."""
        model.set_external_field(0, 1.5)
        assert model.external_fields[0].item() == 1.5
        
        # Test setting all fields
        fields = torch.randn(model.n_spins)
        model.set_external_fields(fields)
        assert torch.allclose(model.external_fields, fields)
    
    def test_flip_spin(self, model):
        """Test spin flipping."""
        original_spin = model.spins[0].item()
        original_energy = model.compute_energy()
        
        delta_energy = model.flip_spin(0)
        
        # Check spin was flipped
        assert model.spins[0].item() == -original_spin
        
        # Check energy change
        new_energy = model.compute_energy()
        assert abs(new_energy - original_energy - delta_energy) < 1e-6
    
    def test_compute_energy(self, model):
        """Test energy computation."""
        # Set known configuration
        model.spins = torch.tensor([1, -1, 1, -1, 1, -1, 1, -1, 1, -1], dtype=torch.float)
        model.set_coupling(0, 1, -1.0)  # Ferromagnetic
        model.set_external_field(0, 0.5)
        
        energy = model.compute_energy()
        
        # Should be a finite number
        assert torch.isfinite(torch.tensor(energy))
        
        # Energy should be cached
        energy2 = model.compute_energy()
        assert energy == energy2
    
    def test_get_local_field(self, model):
        """Test local field computation."""
        model.set_coupling(0, 1, 1.0)
        model.set_external_field(0, 0.5)
        
        local_field = model.get_local_field(0)
        
        # Should include both coupling and external field contributions
        assert isinstance(local_field, float)
        assert torch.isfinite(torch.tensor(local_field))
    
    def test_magnetization(self, model):
        """Test magnetization calculation."""
        # Set all spins to +1
        model.spins = torch.ones(model.n_spins)
        assert model.get_magnetization() == 1.0
        
        # Set all spins to -1
        model.spins = -torch.ones(model.n_spins)
        assert model.get_magnetization() == -1.0
        
        # Set half spins up, half down
        model.spins = torch.tensor([1, -1, 1, -1, 1, -1, 1, -1, 1, -1], dtype=torch.float)
        assert model.get_magnetization() == 0.0
    
    def test_reset_to_random(self, model):
        """Test random reset."""
        original_spins = model.spins.clone()
        model.reset_to_random()
        
        # Spins should still be ±1
        assert torch.all(torch.abs(model.spins) == 1)
        
        # Should be different (with high probability)
        assert not torch.allclose(model.spins, original_spins)
    
    def test_copy(self, model):
        """Test model copying."""
        # Modify original
        model.set_coupling(0, 1, 2.0)
        model.set_external_field(0, 1.0)
        
        # Create copy
        model_copy = model.copy()
        
        # Should be independent
        assert model_copy.n_spins == model.n_spins
        assert torch.allclose(model_copy.spins, model.spins)
        assert torch.allclose(model_copy.external_fields, model.external_fields)
        
        # Modify copy shouldn't affect original
        model_copy.set_external_field(0, 5.0)
        assert model.external_fields[0].item() != 5.0
    
    def test_serialization(self, model):
        """Test to_dict and from_dict."""
        # Modify model
        model.set_coupling(0, 1, 1.5)
        model.set_external_field(0, 0.8)
        
        # Serialize
        data = model.to_dict()
        
        # Check structure
        assert "config" in data
        assert "spins" in data
        assert "couplings" in data
        assert "external_fields" in data
        
        # Deserialize
        model2 = IsingModel.from_dict(data)
        
        # Check equality
        assert model2.n_spins == model.n_spins
        assert torch.allclose(model2.spins, model.spins)
        assert torch.allclose(model2.external_fields, model.external_fields)
    
    def test_dense_vs_sparse(self):
        """Test dense vs sparse coupling matrices."""
        config_dense = IsingModelConfig(n_spins=5, use_sparse=False)
        config_sparse = IsingModelConfig(n_spins=5, use_sparse=True)
        
        model_dense = IsingModel(config_dense)
        model_sparse = IsingModel(config_sparse)
        
        # Set same couplings and fields
        coupling_matrix = torch.randn(5, 5)
        coupling_matrix = (coupling_matrix + coupling_matrix.t()) / 2
        torch.fill_diagonal_(coupling_matrix, 0)
        
        model_dense.set_couplings_from_matrix(coupling_matrix)
        model_sparse.set_couplings_from_matrix(coupling_matrix)
        
        fields = torch.randn(5)
        model_dense.set_external_fields(fields)
        model_sparse.set_external_fields(fields)
        
        # Set same spin configuration
        spins = torch.randint(0, 2, (5,)) * 2 - 1
        model_dense.set_spins(spins)
        model_sparse.set_spins(spins)
        
        # Energies should be the same
        energy_dense = model_dense.compute_energy()
        energy_sparse = model_sparse.compute_energy()
        
        assert abs(energy_dense - energy_sparse) < 1e-6
    
    def test_cache_invalidation(self, model):
        """Test energy cache invalidation."""
        # Compute energy (should cache)
        energy1 = model.compute_energy()
        
        # Flip spin (should invalidate cache)
        model.flip_spin(0)
        energy2 = model.compute_energy()
        
        # Energies should be different
        assert energy1 != energy2
        
        # Set external field (should invalidate cache)
        energy3 = model.compute_energy()
        model.set_external_field(0, 1.0)
        energy4 = model.compute_energy()
        
        assert energy3 != energy4
    
    def test_invalid_indices(self, model):
        """Test handling of invalid spin indices."""
        with pytest.raises((IndexError, RuntimeError)):
            model.flip_spin(model.n_spins)  # Out of bounds
        
        with pytest.raises((IndexError, RuntimeError)):
            model.set_coupling(-1, 0, 1.0)  # Negative index
    
    def test_device_consistency(self):
        """Test device consistency."""
        if torch.cuda.is_available():
            config = IsingModelConfig(n_spins=10, device="cuda")
            model = IsingModel(config)
            
            assert model.spins.device.type == "cuda"
            assert model.external_fields.device.type == "cuda"
            assert model.couplings.device.type == "cuda"
    
    def test_repr(self, model):
        """Test string representation."""
        repr_str = repr(model)
        
        assert "IsingModel" in repr_str
        assert "n_spins=10" in repr_str
        assert "energy=" in repr_str
        assert "magnetization=" in repr_str


class TestIsingModelEdgeCases:
    """Test edge cases for IsingModel."""
    
    def test_single_spin(self):
        """Test model with single spin."""
        config = IsingModelConfig(n_spins=1)
        model = IsingModel(config)
        
        assert model.n_spins == 1
        assert model.spins.shape == (1,)
        
        energy = model.compute_energy()
        assert torch.isfinite(torch.tensor(energy))
    
    def test_large_model(self):
        """Test larger model (marked as slow)."""
        config = IsingModelConfig(n_spins=1000)
        model = IsingModel(config)
        
        assert model.n_spins == 1000
        
        # Should be able to compute energy efficiently
        energy = model.compute_energy()
        assert torch.isfinite(torch.tensor(energy))
    
    def test_extreme_values(self):
        """Test with extreme coupling and field values."""
        config = IsingModelConfig(n_spins=5)
        model = IsingModel(config)
        
        # Very large coupling
        model.set_coupling(0, 1, 1000.0)
        energy1 = model.compute_energy()
        assert torch.isfinite(torch.tensor(energy1))
        
        # Very small coupling
        model.set_coupling(0, 1, 1e-10)
        energy2 = model.compute_energy()
        assert torch.isfinite(torch.tensor(energy2))
        
        # Very large field
        model.set_external_field(0, 1000.0)
        energy3 = model.compute_energy()
        assert torch.isfinite(torch.tensor(energy3))
    
    def test_zero_couplings(self):
        """Test model with no couplings."""
        config = IsingModelConfig(n_spins=5)
        model = IsingModel(config)
        
        # Energy should just be external field contribution
        model.set_external_field(0, 1.0)
        energy = model.compute_energy()
        
        # With random spins, energy should be reasonable
        assert torch.isfinite(torch.tensor(energy))
    
    def test_all_same_spins(self):
        """Test with all spins in same direction."""
        config = IsingModelConfig(n_spins=5)
        model = IsingModel(config)
        
        # All spins up
        model.spins = torch.ones(5)
        model.set_coupling(0, 1, -1.0)  # Ferromagnetic
        
        energy = model.compute_energy()
        assert torch.isfinite(torch.tensor(energy))
        assert energy < 0  # Should be favorable for ferromagnetic coupling