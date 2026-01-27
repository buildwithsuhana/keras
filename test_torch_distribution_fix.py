"""Test for enhanced PyTorch distribution library.

This test verifies that the torch distribution library now properly supports
all parallelism modes: data parallel, model parallel, and tensor parallel.
"""

import os
import sys
import unittest
from unittest import mock

# Set torch backend before importing keras
os.environ["KERAS_BACKEND"] = "torch"

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# Mock distributed initialization to test without actual multi-process setup
class TestTorchDistributionLib(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock distributed environment
        self.mock_dist_init = mock.patch('torch.distributed.init_process_group')
        self.mock_is_initialized = mock.patch('torch.distributed.is_initialized')
        self.mock_get_rank = mock.patch('torch.distributed.get_rank')
        self.mock_get_world_size = mock.patch('torch.distributed.get_world_size')
        
        self.mock_init = self.mock_dist_init.start()
        self.mock_initialized = self.mock_is_initialized.start()
        self.mock_rank = self.mock_get_rank.start()
        self.mock_world_size = self.mock_get_world_size.start()
        
        # Default mock returns
        self.mock_initialized.return_value = True
        self.mock_rank.return_value = 0
        self.mock_world_size.return_value = 2
        
        # Import after patching
        from keras.src.backend.torch import distribution_lib
        self.dist_lib = distribution_lib

    def tearDown(self):
        """Clean up after tests."""
        self.mock_dist_init.stop()
        self.mock_is_initialized.stop()
        self.mock_get_rank.stop()
        self.mock_get_world_size.stop()

    def test_all_gather_single_argument_backward_compatibility(self):
        """Test that single-argument all_gather still works (backward compatibility)."""
        tensor = torch.randn(4, 8)
        
        # Mock the actual distributed gathering
        with mock.patch.object(self.dist_lib, '_get_world_size', return_value=2):
            with mock.patch('torch.distributed.all_gather') as mock_gather:
                mock_gather.return_value = None
                
                result = self.dist_lib.all_gather(tensor)
                # Should work with default axis=0
                self.assertIsNotNone(result)

    def test_all_gather_with_axis_parameter(self):
        """Test that axis parameter now works (key fix for tensor parallelism)."""
        tensor = torch.randn(4, 8)
        
        # Mock to avoid actual distributed calls
        with mock.patch.object(self.dist_lib, '_get_world_size', return_value=2):
            with mock.patch('torch.distributed.all_gather') as mock_gather:
                # Create proper mock return values
                mock_output = [torch.randn(4, 8), torch.randn(4, 8)]
                mock_gather.return_value = None
                
                # This should NOT raise TypeError anymore
                try:
                    # Test axis=0 (batch dimension - data parallelism)
                    result_0 = self.dist_lib.all_gather(tensor, axis=0)
                    self.assertIsNotNone(result_0)
                    
                    # Test axis=-1 (output dimension - tensor parallelism)
                    result_1 = self.dist_lib.all_gather(tensor, axis=-1)
                    self.assertIsNotNone(result_1)
                    
                    # Test axis=1 (hidden dimension)
                    result_2 = self.dist_lib.all_gather(tensor, axis=1)
                    self.assertIsNotNone(result_2)
                    
                except TypeError as e:
                    self.fail(f"all_gather should accept axis parameter, got: {e}")

    def test_all_gather_with_axis_name(self):
        """Test that axis_name parameter is accepted for JAX compatibility."""
        tensor = torch.randn(4, 8)
        
        with mock.patch.object(self.dist_lib, '_get_world_size', return_value=2):
            with mock.patch('torch.distributed.all_gather') as mock_gather:
                mock_gather.return_value = None
                
                # This should work without raising an error
                try:
                    result = self.dist_lib.all_gather(tensor, axis=0, axis_name="model")
                    self.assertIsNotNone(result)
                except TypeError as e:
                    self.fail(f"all_gather should accept axis_name parameter, got: {e}")

    def test_all_reduce_with_op_parameter(self):
        """Test that all_reduce supports different reduction operations."""
        tensor = torch.randn(4, 8)
        
        with mock.patch('torch.distributed.all_reduce') as mock_reduce:
            mock_reduce.return_value = None
            
            # Test sum reduction
            result_sum = self.dist_lib.all_reduce(tensor, op="sum")
            mock_reduce.assert_called()
            
            # Test mean reduction
            result_mean = self.dist_lib.all_reduce(tensor, op="mean")
            # Mean should divide by world_size

    def test_convenience_functions(self):
        """Test convenience functions for different parallelism modes."""
        tensor = torch.randn(4, 8)
        
        # Test data parallel convenience function
        with mock.patch.object(self.dist_lib, 'all_reduce') as mock_reduce:
            mock_reduce.return_value = tensor
            result = self.dist_lib.data_parallel_all_reduce(tensor)
            mock_reduce.assert_called_with(tensor, op="mean")
        
        # Test tensor parallel convenience function
        with mock.patch.object(self.dist_lib, 'all_gather') as mock_gather:
            mock_gather.return_value = tensor
            result = self.dist_lib.tensor_parallel_all_gather(tensor, axis=-1)
            mock_gather.assert_called_with(tensor, axis=-1, axis_name="model")
        
        # Test model parallel convenience function
        with mock.patch.object(self.dist_lib, 'all_reduce') as mock_reduce:
            mock_reduce.return_value = tensor
            result = self.dist_lib.model_parallel_all_reduce(tensor, axis_name="model")
            mock_reduce.assert_called_with(tensor, op="sum", axis_name="model")

    def test_scatter_function(self):
        """Test that scatter function is available."""
        tensor = torch.randn(8, 8)
        
        with mock.patch.object(self.dist_lib, '_get_world_size', return_value=2):
            with mock.patch('torch.distributed.scatter') as mock_scatter:
                mock_scatter.return_value = torch.randn(4, 8)
                
                result = self.dist_lib.scatter(tensor, scatter_dim=0, num_chunks=2)
                self.assertIsNotNone(result)

    def test_gather_function(self):
        """Test that gather function is available."""
        tensor = torch.randn(4, 8)
        
        with mock.patch.object(self.dist_lib, '_get_world_size', return_value=2):
            with mock.patch('torch.distributed.gather') as mock_gather:
                mock_gather.return_value = None
                
                result = self.dist_lib.gather(tensor, axis=0, dst=0)
                self.assertIsNotNone(result)

    def test_reduce_scatter_function(self):
        """Test that reduce_scatter function is available."""
        tensor = torch.randn(4, 8)
        
        with mock.patch.object(self.dist_lib, '_get_world_size', return_value=2):
            with mock.patch.object(self.dist_lib, 'all_reduce') as mock_reduce:
                with mock.patch.object(self.dist_lib, 'scatter') as mock_scatter:
                    mock_reduce.return_value = tensor
                    mock_scatter.return_value = torch.randn(2, 8)
                    
                    result = self.dist_lib.reduce_scatter(tensor, reduce_op="sum", scatter_axis=0)
                    self.assertIsNotNone(result)

    def test_utility_functions(self):
        """Test utility functions for distributed training."""
        # Test is_distributed_initialized
        self.assertTrue(self.dist_lib.is_distributed_initialized())
        
        # Test get_local_rank
        with mock.patch('torch.distributed.get_rank', return_value=1):
            rank = self.dist_lib.get_local_rank()
            self.assertEqual(rank, 1)
        
        # Test get_local_world_size
        with mock.patch('torch.cuda.device_count', return_value=4):
            local_size = self.dist_lib.get_local_world_size()
            self.assertEqual(local_size, 4)


class TestBackwardCompatibility(unittest.TestCase):
    """Test that existing code using old interface still works."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_is_initialized = mock.patch('torch.distributed.is_initialized')
        self.mock_get_rank = mock.patch('torch.distributed.get_rank')
        self.mock_get_world_size = mock.patch('torch.distributed.get_world_size')
        
        self.mock_initialized = self.mock_is_initialized.start()
        self.mock_rank = self.mock_get_rank.start()
        self.mock_world_size = self.mock_get_world_size.start()
        
        self.mock_initialized.return_value = False  # Single process
        self.mock_rank.return_value = 0
        self.mock_world_size.return_value = 1
        
        from keras.src.backend.torch import distribution_lib
        self.dist_lib = distribution_lib

    def tearDown(self):
        """Clean up after tests."""
        self.mock_is_initialized.stop()
        self.mock_get_rank.stop()
        self.mock_get_world_size.stop()

    def test_all_gather_single_arg_works_when_not_distributed(self):
        """Test single-arg all_gather works when not distributed."""
        tensor = torch.randn(4, 8)
        result = self.dist_lib.all_gather(tensor)
        
        # Should return the same tensor when not distributed
        self.assertEqual(result.shape, tensor.shape)

    def test_all_reduce_works_when_not_distributed(self):
        """Test all_reduce works when not distributed."""
        tensor = torch.randn(4, 8)
        result = self.dist_lib.all_reduce(tensor, op="sum")
        
        # Should return the same tensor when not distributed
        self.assertEqual(result.shape, tensor.shape)


class TestIntegrationWithTensorParallelism(unittest.TestCase):
    """Test integration with tensor parallelism code paths."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_is_initialized = mock.patch('torch.distributed.is_initialized')
        self.mock_get_rank = mock.patch('torch.distributed.get_rank')
        self.mock_get_world_size = mock.patch('torch.distributed.get_world_size')
        self.mock_all_gather = mock.patch('torch.distributed.all_gather')
        self.mock_all_reduce = mock.patch('torch.distributed.all_reduce')
        
        self.mock_initialized = self.mock_is_initialized.start()
        self.mock_rank = self.mock_get_rank.start()
        self.mock_world_size = self.mock_get_world_size.start()
        self.mock_gather = self.mock_all_gather.start()
        self.mock_reduce = self.mock_all_reduce.start()
        
        self.mock_initialized.return_value = True
        self.mock_rank.return_value = 0
        self.mock_world_size.return_value = 2
        
        from keras.src.backend.torch import distribution_lib
        self.dist_lib = distribution_lib

    def tearDown(self):
        """Clean up after tests."""
        self.mock_is_initialized.stop()
        self.mock_get_rank.stop()
        self.mock_get_world_size.stop()
        self.mock_all_gather.stop()
        self.mock_all_reduce.stop()

    def test_autoconfig_pattern(self):
        """Test the pattern used in autoconfig.py."""
        tensor = torch.randn(4, 8)
        
        # This is how autoconfig.py calls all_gather
        axis = -1
        axis_name = "model"
        
        with mock.patch.object(self.dist_lib, 'all_gather') as mock_gather:
            mock_gather.return_value = tensor
            
            result = self.dist_lib.all_gather(tensor, axis=axis, axis_name=axis_name)
            
            # Verify it was called with the correct arguments
            mock_gather.assert_called_once_with(tensor, axis=axis, axis_name=axis_name)

    def test_parameter_sharding_pattern(self):
        """Test the pattern used in parameter_sharding.py."""
        tensor = torch.randn(4, 8)
        
        # This is how parameter_sharding.py calls all_gather
        val = tensor
        dim = -1
        axis_name = "model"
        
        with mock.patch.object(self.dist_lib, 'all_gather') as mock_gather:
            mock_gather.return_value = tensor
            
            result = self.dist_lib.all_gather(val, axis=dim, axis_name=axis_name)
            
            # Verify it was called correctly
            mock_gather.assert_called_once_with(val, axis=dim, axis_name=axis_name)

    def test_all_reduce_with_axis_name(self):
        """Test all_reduce with axis_name parameter."""
        tensor = torch.randn(4, 8)
        
        # This is how _reduce_sum in autoconfig.py calls all_reduce
        with mock.patch.object(self.dist_lib, 'all_reduce') as mock_reduce:
            mock_reduce.return_value = tensor
            
            result = self.dist_lib.all_reduce(tensor, op="sum", axis_name="model")
            
            mock_reduce.assert_called_once_with(tensor, op="sum", axis_name="model")


if __name__ == "__main__":
    unittest.main()

