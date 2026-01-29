#!/usr/bin/env python3
"""Test script to verify optimizer serialization fix."""

import os
os.environ["KERAS_BACKEND"] = "torch"

import keras
from keras import layers
from keras.src import optimizers
import numpy as np


def test_optimizer_serialization():
    """Test that optimizer can be serialized and deserialized properly."""
    print("Testing optimizer serialization...")
    
    # Create an optimizer
    optimizer = optimizers.Adam(learning_rate=0.001)
    print(f"Created optimizer: {optimizer}")
    
    # Get config
    config = optimizer.get_config()
    print(f"Optimizer config: {config}")
    
    # Check that learning_rate is properly serialized
    assert "learning_rate" in config, "learning_rate not in config"
    assert isinstance(config["learning_rate"], float), f"learning_rate is not a float: {type(config['learning_rate'])}"
    assert config["learning_rate"] == 0.001, f"learning_rate value is wrong: {config['learning_rate']}"
    
    # Check other config values
    assert "beta_1" in config, "beta_1 not in config"
    assert "beta_2" in config, "beta_2 not in config"
    assert "epsilon" in config, "epsilon not in config"
    assert "amsgrad" in config, "amsgrad not in config"
    
    # Test deserialization
    new_optimizer = optimizers.Adam.from_config(config)
    print(f"Deserialized optimizer: {new_optimizer}")
    
    new_config = new_optimizer.get_config()
    print(f"Deserialized optimizer config: {new_config}")
    
    # Verify the configs match
    assert config["learning_rate"] == new_config["learning_rate"], "learning_rate mismatch"
    assert config["beta_1"] == new_config["beta_1"], "beta_1 mismatch"
    assert config["beta_2"] == new_config["beta_2"], "beta_2 mismatch"
    assert config["epsilon"] == new_config["epsilon"], "epsilon mismatch"
    assert config["amsgrad"] == new_config["amsgrad"], "amsgrad mismatch"
    
    print("✓ Optimizer serialization test PASSED!")
    return True


def test_model_compile_with_optimizer():
    """Test that model.compile works with optimizer serialization."""
    print("\nTesting model.compile with optimizer...")
    
    # Create a simple model
    model = keras.Sequential([
        layers.Dense(128, activation="relu", input_shape=(64,)),
        layers.Dense(10)
    ])
    
    # Compile with string optimizer
    model.compile(optimizer="adam", loss="mse")
    print(f"Model compiled successfully")
    
    # Get the optimizer config
    optimizer = model.optimizer
    assert optimizer is not None, "Optimizer is None"
    
    config = optimizer.get_config()
    print(f"Optimizer config: {config}")
    
    # Check config is not empty
    assert "learning_rate" in config, "learning_rate not in config"
    assert len(config) > 1, f"Config is too small: {config}"
    
    print("✓ Model compile test PASSED!")
    return True


def test_optimizer_with_custom_lr():
    """Test optimizer with custom learning rate values."""
    print("\nTesting optimizer with custom learning rates...")
    
    for lr in [0.1, 0.01, 0.001, 0.0001, 1.0]:
        optimizer = optimizers.Adam(learning_rate=lr)
        config = optimizer.get_config()
        assert config["learning_rate"] == lr, f"lr mismatch for {lr}: got {config['learning_rate']}"
        print(f"  ✓ lr={lr}: config learning_rate={config['learning_rate']}")
    
    print("✓ Custom learning rate test PASSED!")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("Optimizer Serialization Fix Test")
    print("=" * 60)
    
    try:
        test_optimizer_serialization()
        test_model_compile_with_optimizer()
        test_optimizer_with_custom_lr()
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

