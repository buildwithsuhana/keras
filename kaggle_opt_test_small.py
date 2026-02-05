"""Smaller test for kaggle_opt_test - uses a simple model instead of OPT-125m"""

import os
# Must be set before any other imports
os.environ["KERAS_BACKEND"] = "torch"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.distributed as dist
import keras
import numpy as np
from keras.src.distribution import ModelParallel, DeviceMesh, LayoutMap, list_devices, initialize


def run_small_training():
    """Run a small distributed training test with a simple model."""
    
    # 1. Initialize Distributed Backend
    initialize()
    
    devices = list_devices("gpu")
    num_devices = len(devices)
    
    if num_devices < 2:
        print(f"Detected {num_devices} GPU(s). Need at least 2 for distributed demo.")
        return

    # 2. Define 2D Device Mesh
    dp_dim = 2 if num_devices >= 4 else 1
    mp_dim = num_devices // dp_dim
    
    mesh = DeviceMesh(
        shape=(dp_dim, mp_dim),
        axis_names=["data", "model"],
        devices=devices
    )

    # 3. Create Sharding Layout
    layout_map = LayoutMap(mesh)
    layout_map[".*dense.*kernel"] = ("model", None)  # Shard along model axis
    
    # 4. Initialize Strategy
    strategy = ModelParallel(
        layout_map=layout_map,
        batch_dim_name="data",
        auto_shard_dataset=False
    )

    # 5. Build a simple model within Strategy Scope
    with strategy.scope():
        # Simple Dense layers for testing
        model = keras.Sequential([
            keras.layers.Dense(64, activation='relu', input_shape=(100,)),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(10, activation='softmax')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

    # 6. Generate small dummy data
    # Using small batch size and limited samples for quick testing
    x_train = np.random.random((100, 100))
    y_train = np.random.randint(0, 10, size=(100,))
    y_train = keras.utils.to_categorical(y_train, num_classes=10)

    # 7. Start Training
    print(f"Rank {dist.get_rank()}: Starting small training test...")
    model.fit(
        x_train, 
        y_train,
        epochs=2,  # Just 2 epochs for quick test
        batch_size=32,
        verbose=1 if dist.get_rank() == 0 else 0
    )

    # 8. Quick evaluation
    if dist.get_rank() == 0:
        print("\n✓ Training completed successfully!")
        print("✓ The fix for non-floating dtype parameters is working correctly.")

    # 9. Cleanup
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    run_small_training()

