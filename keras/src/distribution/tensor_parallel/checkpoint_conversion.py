import os
import sys
import logging
import keras
import keras_hub

# --- Config ---
# We define these first so we can set environment variables
PRESET_NAME = "opt_1.3b_en"
SHARD_COUNT = 2
CHECKPOINT_DIR = f"./{PRESET_NAME}_checkpoint"

# --- Backend and Device Configuration ---
# MUST be set *BEFORE* importing JAX
os.environ["KERAS_BACKEND"] = "jax"
os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={SHARD_COUNT}"

# --- Project Root Setup ---
try:
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(_script_dir)))
    )
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)
except Exception:
    print(
        "Could not add project root to sys.path. "
        "Please run from the 'keras' directory or install as a package."
    )

# --- Now we can import JAX and your module ---
import jax
from keras.src.distribution.tensor_parallel.tensor_parallel_keras import (
    TensorParallelKeras,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- Model Class Mapping ---
MODEL_CLASS = keras_hub.models.OPTCausalLM

# --- Main Conversion Logic ---
if __name__ == "__main__":
    logger.info(f"--- CONVERSION SCRIPT for {PRESET_NAME} ---")
    
    if os.path.isdir(CHECKPOINT_DIR):
        logger.warning(f"Checkpoint directory {CHECKPOINT_DIR} already exists. Skipping.")
        sys.exit(0)

    # 1. Load the FULL model into RAM (This is the OOM-prone step)
    logger.info(f"Loading full {PRESET_NAME} model from KerasHub...")
    try:
        full_model = MODEL_CLASS.from_preset(PRESET_NAME, preprocessor=None)
    except Exception as e:
        logger.critical(f"Failed to load full model (OOM?): {e}")
        sys.exit(1)
    
    logger.info(f"Model loaded with {full_model.count_params():,} parameters.")

    # 2. Initialize TensorParallelKeras to shard the model
    logger.info(f"Sharding model across {SHARD_COUNT} devices...")
    devices = jax.devices()[:SHARD_COUNT]
    if len(devices) < SHARD_COUNT:
        logger.critical(f"Error: JAX only found {len(devices)} devices, but {SHARD_COUNT} were requested.")
        sys.exit(1)

    tp_model = TensorParallelKeras(
        model=full_model,
        device_count=SHARD_COUNT,
        device_ids=devices,
    )

    # 3. Save the sharded checkpoint
    # (No need to compile, we just want the weights)
    logger.info(f"Saving sharded checkpoint to {CHECKPOINT_DIR}...")
    tp_model.save_checkpoint(CHECKPOINT_DIR)

    logger.info("✅ Conversion complete. Sharded checkpoint is ready.")
    logger.info("You can now run test_models.py on any machine.")