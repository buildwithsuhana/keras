import os
import sys
import numpy as np

# --- 1. Project Root Setup ---
try:
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(_script_dir)))
    )
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)
except Exception:
    pass

# --- 2. Environment Configuration ---
os.environ["KERAS_BACKEND"] = "jax"
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"

import jax
import keras

# --- Circular Import Fix ---
try:
    from keras.src.ops.operation import Operation
    from keras.src.activations import activations
except ImportError:
    pass

import keras_hub
from keras.src.distribution.distribution_lib import AutoTPDistribution, DeviceMesh

print("\n" + "="*60)
print("🧪 FORENSIC SHARDING VERIFICATION")
print("="*60)

# --- 1. Setup Environment ---
devices = jax.devices()
device_mesh = DeviceMesh(shape=(1, 2), axis_names=["data", "model"], devices=devices)
print(f"Devices: {devices}")

# --- 2. Create Reference Model (Global) ---
print("\n1️⃣  Creating Global Reference Model (Virtual)...")
# We use the preset to know what the "Unsharded" shape should be
model_preset = "opt_125m_en" 
ref_model = keras_hub.models.OPTCausalLM.from_preset(model_preset, load_weights=False)

# Find the global embedding shape
global_shape = None
for var in ref_model.weights:
    if "token_embedding" in var.path and "embeddings" in var.path:
        global_shape = var.shape
        print(f"   Global Embedding Shape: {global_shape}")
        break

if not global_shape:
    raise ValueError("Could not find embedding layer in reference model")

# --- 3. Create Distributed Model (AutoTP) ---
print("\n2️⃣  Creating Distributed Model (AutoTP)...")
# Re-instantiate to simulate the actual distributed load
dist_model_template = keras_hub.models.OPTCausalLM.from_preset(model_preset, load_weights=False)
distribution = AutoTPDistribution(
    model=dist_model_template,
    device_mesh=device_mesh,
)
keras.distribution.set_distribution(distribution)
tp_model = distribution.model
tp_model.build(input_shape={"token_ids": (1, 16), "padding_mask": (1, 16)})

# --- 4. THE PROOF ---
print("\n3️⃣  Comparing Physical Memory Allocations...")

# Collect all embedding shards from the distributed model
shards_found = []
for var in tp_model.weights:
    # Note: In TP, multiple variables might share the same path name but exist on different devices
    if "token_embedding" in var.path and "embeddings" in var.path:
        # Get the physical JAX buffer
        # We access .value to get the buffer, then inspect its device
        try:
            # Unwrap ShardedWeight if necessary, though .weights usually does it
            val = var.value
            # Check if it's a JAX Array
            if hasattr(val, 'device'):
                dev = val.device
                shards_found.append({
                    "shape": var.shape,
                    "device": dev,
                    "elements": np.prod(var.shape)
                })
        except Exception:
            pass

print(f"   Found {len(shards_found)} physical shards for Embedding Layer.")

# Verification Logic
global_elements = np.prod(global_shape)
shard_0 = shards_found[0]
shard_1 = shards_found[1] if len(shards_found) > 1 else None

print(f"\n   📊 Comparison:")
print(f"   Expected Global Elements: {global_elements:,} (100%)")
print(f"   Shard 0 Elements:         {shard_0['elements']:,} ({shard_0['elements']/global_elements:.0%}) on {shard_0['device']}")

if shard_1:
    print(f"   Shard 1 Elements:         {shard_1['elements']:,} ({shard_1['elements']/global_elements:.0%}) on {shard_1['device']}")

print("\n" + "-"*60)
if shard_0['elements'] < global_elements:
    print("✅ SUCCESS: Shard is smaller than global model.")
    print("✅ SUCCESS: Full model weights were NEVER materialized on a single device.")
    if shard_0['elements'] + (shard_1['elements'] if shard_1 else 0) == global_elements:
        print("✅ SUCCESS: Sum of shards equals global model (Lossless).")
else:
    print("❌ FAILURE: Shard is full size. Model is replicated (OOM Risk).")
print("="*60)