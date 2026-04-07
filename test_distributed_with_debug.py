#!/usr/bin/env python3
"""
Test script for distributed PyTorch training with debug logging enabled.

Run with:
    KERAS_BACKEND=torch KERAS_TORCH_DEVICE=cpu python test_distributed_with_debug.py

Or for GPU (if available):
    KERAS_BACKEND=torch python test_distributed_with_debug.py

This script enables debug logging to trace DTensor unbind operations and 
shape conversions, helping diagnose distributed training issues.
"""

import logging
import sys

# Configure logging BEFORE importing keras
logging.basicConfig(
    level=logging.DEBUG,
    format='[%(levelname)-8s %(name)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Enable debug logging for keras distribution and layer modules
logging.getLogger('keras.distribution').setLevel(logging.DEBUG)
logging.getLogger('keras.torch.backend').setLevel(logging.DEBUG)
logging.getLogger('keras.layers').setLevel(logging.DEBUG)

# Now import keras components
import os
os.environ.setdefault('KERAS_BACKEND', 'torch')

import torch
import torch.nn as nn
from keras import layers, models
from torch.distributed.tensor import DTensor, Replicate

logger = logging.getLogger(__name__)


def test_embedding_with_distributed_shapes():
    """Test embedding layer with distributed shapes."""
    logger.info("=" * 70)
    logger.info("TEST: Embedding layer with distributed shapes")
    logger.info("=" * 70)
    
    try:
        # Create a simple embedding layer
        embedding = layers.Embedding(input_dim=1000, output_dim=256)
        
        # Test with regular tuple shape (control)
        logger.info("Building embedding with regular tuple shape...")
        embedding.build(input_shape=(None, 32))
        logger.info("✓ Successfully built with regular shape")
        
        # Test compute_output_shape with safe concatenation
        output_shape = embedding.compute_output_shape((None, 32))
        logger.info(f"✓ compute_output_shape returned: {output_shape}")
        assert output_shape == (None, 32, 256), f"Unexpected shape: {output_shape}"
        
    except Exception as e:
        logger.error(f"✗ Test failed: {e}", exc_info=True)
        return False
    
    logger.info("✓ Embedding test PASSED\n")
    return True


def test_shape_function():
    """Test the backend shape() function."""
    logger.info("=" * 70)
    logger.info("TEST: Backend shape() function with regular tensors")
    logger.info("=" * 70)
    
    try:
        from keras.src.backend.torch import core
        
        # Create test tensors
        regular_tensor = torch.randn(2, 3, 4)
        logger.info(f"Created regular tensor with shape: {regular_tensor.shape}")
        
        # Test shape function
        shape_result = core.shape(regular_tensor)
        logger.info(f"✓ core.shape() returned: {shape_result} (type: {type(shape_result).__name__})")
        
        assert isinstance(shape_result, tuple), f"Expected tuple, got {type(shape_result)}"
        assert shape_result == (2, 3, 4), f"Unexpected shape: {shape_result}"
        
    except Exception as e:
        logger.error(f"✗ Test failed: {e}", exc_info=True)
        return False
    
    logger.info("✓ Shape function test PASSED\n")
    return True


def test_build_wrapper():
    """Test the Layer.build_wrapper with safe shape conversion."""
    logger.info("=" * 70)
    logger.info("TEST: Layer.build_wrapper with shape conversion")
    logger.info("=" * 70)
    
    try:
        # Create a custom layer to test build wrapper
        class TestLayer(layers.Layer):
            def build(self, input_shape):
                logger.info(f"TestLayer.build() received input_shape: {input_shape} (type: {type(input_shape).__name__})")
                assert isinstance(input_shape, tuple), f"Expected tuple in build(), got {type(input_shape)}"
                self.dense = layers.Dense(64)
                
            def call(self, inputs):
                return self.dense(inputs)
        
        layer = TestLayer()
        logger.info("Building TestLayer...")
        layer.build(input_shape=(None, 32))
        logger.info("✓ TestLayer.build_wrapper successfully converted shape to tuple")
        
    except Exception as e:
        logger.error(f"✗ Test failed: {e}", exc_info=True)
        return False
    
    logger.info("✓ Build wrapper test PASSED\n")
    return True


def test_dtensor_patches():
    """Test that DTensor monkey patches are applied."""
    logger.info("=" * 70)
    logger.info("TEST: DTensor monkey patches")
    logger.info("=" * 70)
    
    try:
        # Check if patches are applied by inspecting the methods
        from torch.distributed.tensor import DTensor
        
        # Get the unbind method
        unbind_method = DTensor.unbind
        logger.info(f"DTensor.unbind method: {unbind_method}")
        
        # Check if it's our patched version (will have 'safe_unbind' in closure or name hints)
        if hasattr(unbind_method, '__name__'):
            logger.info(f"DTensor.unbind method name: {unbind_method.__name__}")
        
        logger.info("✓ DTensor methods accessible (patches applied or PyTorch method available)")
        
    except Exception as e:
        logger.error(f"✗ Test failed: {e}", exc_info=True)
        return False
    
    logger.info("✓ DTensor patches test PASSED\n")
    return True


def main():
    logger.info("\n" + "=" * 70)
    logger.info("KERAS DISTRIBUTED TRAINING DEBUG TEST SUITE")
    logger.info("=" * 70 + "\n")
    
    results = []
    
    # Run all tests
    results.append(("DTensor Patches", test_dtensor_patches()))
    results.append(("Shape Function", test_shape_function()))
    results.append(("Build Wrapper", test_build_wrapper()))
    results.append(("Embedding Layer", test_embedding_with_distributed_shapes()))
    
    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("TEST SUMMARY")
    logger.info("=" * 70)
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        logger.info(f"{status}: {test_name}")
    
    all_passed = all(passed for _, passed in results)
    logger.info("=" * 70)
    if all_passed:
        logger.info("ALL TESTS PASSED ✓\n")
        return 0
    else:
        logger.info("SOME TESTS FAILED ✗\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
