#!/usr/bin/env python3
"""Simple verification script for PyTorch distribution support."""

import os
import sys

# Set backend before importing keras
os.environ["KERAS_BACKEND"] = "torch"

print("=" * 60)
print("PyTorch Distribution API Verification")
print("=" * 60)

# Test 1: Import verification
print("\n1. Testing imports...")
try:
    from keras.src.backend.torch import distribution_lib
    print("✓ Successfully imported torch distribution_lib")
except Exception as e:
    print(f"✗ Failed to import distribution_lib: {e}")
    sys.exit(1)

# Test 2: Device detection
print("\n2. Testing device detection...")
try:
    devices = distribution_lib.list_devices()
    print(f"✓ Found {len(devices)} devices: {devices}")
    
    cpu_devices = distribution_lib.list_devices("cpu")
    print(f"✓ CPU devices: {cpu_devices}")
    
    gpu_devices = distribution_lib.list_devices("gpu")
    print(f"✓ GPU devices: {gpu_devices}")
    
    device_count = distribution_lib.get_device_count()
    print(f"✓ Total device count: {device_count}")
except Exception as e:
    print(f"✗ Device detection failed: {e}")

# Test 3: Path conversion
print("\n3. Testing path conversion...")
try:
    _convert_keras = distribution_lib._convert_keras_path_to_torch
    _convert_torch = distribution_lib._convert_torch_path_to_keras
    
    # Test Keras to PyTorch
    tests = [
        ("dense/kernel", "dense.weight"),
        ("dense/bias", "dense.bias"),
        ("dense_1/kernel", "dense_1.weight"),
        ("conv2d/kernel", "conv2d.weight"),
        ("batch_normalization/gamma", "batch_normalization.weight"),
    ]
    
    print("Keras -> PyTorch:")
    for keras_path, expected_torch in tests:
        result = _convert_keras(keras_path)
        status = "✓" if result == expected_torch else "✗"
        print(f"  {status} {keras_path} -> {result} (expected: {expected_torch})")
    
    # Test PyTorch to Keras
    torch_tests = [
        ("dense.weight", "dense/kernel"),
        ("dense.bias", "dense/bias"),
        ("batch_normalization.running_mean", "batch_normalization/moving_mean"),
    ]
    
    print("\nPyTorch -> Keras:")
    for torch_path, expected_keras in torch_tests:
        result = _convert_torch(torch_path)
        status = "✓" if result == expected_keras else "✗"
        print(f"  {status} {torch_path} -> {result} (expected: {expected_keras})")
        
except Exception as e:
    print(f"✗ Path conversion failed: {e}")

# Test 4: DeviceMesh creation
print("\n4. Testing DeviceMesh creation...")
try:
    from keras.src.distribution import DeviceMesh
    
    devices = ["cpu:0", "cpu:1", "cpu:2", "cpu:3"]
    mesh = DeviceMesh(shape=(4,), axis_names=["batch"], devices=devices)
    print(f"✓ Created DeviceMesh: {mesh}")
    print(f"  Shape: {mesh.shape}")
    print(f"  Axis names: {mesh.axis_names}")
    print(f"  Devices: {mesh.devices}")
except Exception as e:
    print(f"✗ DeviceMesh creation failed: {e}")

# Test 5: LayoutMap creation
print("\n5. Testing LayoutMap creation...")
try:
    from keras.src.distribution import DeviceMesh, LayoutMap
    
    mesh = DeviceMesh(
        shape=(4,),
        axis_names=["batch"],
        devices=["cpu:0", "cpu:1", "cpu:2", "cpu:3"]
    )
    
    layout_map = LayoutMap(mesh)
    layout_map['dense.*kernel'] = (None, 'batch')
    layout_map['dense.*bias'] = ('batch',)
    
    print(f"✓ Created LayoutMap with {len(layout_map)} rules")
    
    # Test lookup
    layout = layout_map['dense/kernel']
    print(f"✓ Layout lookup for 'dense/kernel': {layout}")
    
except Exception as e:
    print(f"✗ LayoutMap creation failed: {e}")

# Test 6: TensorLayout creation
print("\n6. Testing TensorLayout creation...")
try:
    from keras.src.distribution import DeviceMesh, TensorLayout
    
    mesh = DeviceMesh(
        shape=(4,),
        axis_names=["batch"],
        devices=["cpu:0", "cpu:1", "cpu:2", "cpu:3"]
    )
    
    layout = TensorLayout(axes=(None, 'batch'), device_mesh=mesh)
    print(f"✓ Created TensorLayout: {layout}")
    print(f"  Axes: {layout.axes}")
    
except Exception as e:
    print(f"✗ TensorLayout creation failed: {e}")

# Test 7: DataParallel creation
print("\n7. Testing DataParallel creation...")
try:
    from keras.src.distribution import DataParallel, set_distribution, distribution
    
    devices = ["cpu:0", "cpu:1"]
    dp = DataParallel(devices=devices)
    print(f"✓ Created DataParallel: {dp}")
    
    # Test scope management
    set_distribution(dp)
    assert distribution() == dp
    print("✓ Set distribution successfully")
    
    set_distribution(None)
    assert distribution() is None
    print("✓ Reset distribution successfully")
    
except Exception as e:
    print(f"✗ DataParallel test failed: {e}")

# Test 8: ModelParallel creation
print("\n8. Testing ModelParallel creation...")
try:
    from keras.src.distribution import DeviceMesh, LayoutMap, ModelParallel, set_distribution
    
    mesh = DeviceMesh(
        shape=(2, 2),
        axis_names=["batch", "model"],
        devices=["cpu:0", "cpu:1", "cpu:2", "cpu:3"]
    )
    
    layout_map = LayoutMap(mesh)
    layout_map['dense.*kernel'] = (None, 'model')
    
    mp = ModelParallel(layout_map=layout_map)
    print(f"✓ Created ModelParallel: {mp}")
    print(f"  Device mesh: {mp.device_mesh}")
    print(f"  Batch dim: {mp.batch_dim_name}")
    
    # Test scope management
    with mp.scope():
        assert distribution() == mp
    print("✓ ModelParallel scope works")
    
except Exception as e:
    print(f"✗ ModelParallel test failed: {e}")

# Test 9: Simple model creation
print("\n9. Testing model creation with distribution...")
try:
    from keras.src.distribution import DataParallel, set_distribution
    import keras
    
    devices = ["cpu:0", "cpu:1"]
    dp = DataParallel(devices=devices)
    set_distribution(dp)
    
    # Create a simple model
    # Note: When input_shape is provided to the first layer, the model
    # is automatically built. Calling build() again with a different
    # input_shape would cause a conflict.
    model = keras.Sequential([
        keras.layers.Dense(8, activation='relu', input_shape=(4,)),
        keras.layers.Dense(4)
    ])
    
    print(f"✓ Created model with {len(model.trainable_variables)} variables")
    for var in model.trainable_variables:
        print(f"  - {var.path if hasattr(var, 'path') else 'unknown'}: {var.shape}")
    
    set_distribution(None)
    
except Exception as e:
    print(f"✗ Model creation test failed: {e}")

# Summary
print("\n" + "=" * 60)
print("Verification Complete!")
print("=" * 60)
print("\nIf all tests passed, the PyTorch distribution API is working correctly.")
print("\nNote: Full functionality requires PyTorch with:")
print("- CUDA support for multi-GPU training")
print("- torch-xla for TPU support")
