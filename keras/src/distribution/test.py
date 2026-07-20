import os

# Force 8 virtual host CPU devices for JAX
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
os.environ["KERAS_BACKEND"] = "jax"
os.environ["KERAS_NNX_ENABLED"] = "true"

import gc

import numpy as np

from keras.src.distribution import distribution_lib
from keras.src.distribution.distribution_lib import ParallaxDistribution


def print_sharding_info(model, model_name="Model"):
    print(f"\n📊 --- Sharding & Memory Inspection for {model_name} ---")
    total_params = 0
    total_sharded_size = 0

    # We only print the first few variables to avoid huge logs, but sum all
    for i, variable in enumerate(model.variables):
        val = variable.value
        sharding = getattr(val, "sharding", None)

        total_elements = np.prod(variable.shape)
        local_elements = (
            val.addressable_shards[0].data.size
            if val.addressable_shards
            else total_elements
        )

        if i < 10:  # Only log first 10 variables to keep output readable
            if sharding is not None:
                is_sharded = (
                    not hasattr(sharding, "is_fully_replicated")
                    or not sharding.is_fully_replicated
                )
                status = "✅ SHARDED" if is_sharded else "REPLICATED"
                print(f"Variable: {variable.path:45} | Status: {status}")
                print(
                    f"  Full Shape: {str(variable.shape):15} | "
                    f"Local Shard Shape: "
                    f"{str(val.addressable_shards[0].data.shape)}"
                )
            else:
                print(f"Variable: {variable.path:45} | Status: NOT DISTRIBUTED")

        total_params += total_elements
        total_sharded_size += local_elements

    print(f"  ... (and {len(model.variables) - 10} more variables)")
    print(f"\n📈 Summary for {model_name}:")
    print(f"  Total Parameters: {total_params:,}")
    print(
        f"  Peak Parameter Memory per Device: {total_sharded_size:,} elements"
    )
    savings = (
        (1 - total_sharded_size / total_params) * 100 if total_params > 0 else 0
    )
    print(f"  Memory Reduction vs Single Device: {savings:.1f}%")


def run_opt_test(strategy="fsdp", mesh_shape=None, axis_names=None):
    import jax

    mesh_desc = f"{mesh_shape}" if mesh_shape else "DEFAULT"
    print(
        f"\n🚀 --- Testing OPT 125M with Strategy: {strategy.upper()} | "
        f"Mesh: {mesh_desc} --- 🚀"
    )

    import keras_hub

    seq_len = 128

    def model_builder():
        m = keras_hub.models.OPTCausalLM.from_preset("opt_125m_en")
        m.preprocessor.sequence_length = seq_len
        return m

    # --- FIX: Define appropriate meshes for different strategies ---
    devices = jax.devices()

    if mesh_shape is None:
        if strategy == "auto":
            # Parallax AUTO strategy default to rank-2 mesh (data, model).
            # Using (2, 4) means model is sharded across 4 devices ->
            # 75% reduction.
            mesh_shape = (2, 4)
            axis_names = ["data", "model"]
        elif strategy == "ddp":
            mesh_shape = (8,)
            axis_names = ["data"]
        else:
            # For FSDP, we can use a 1D mesh (8,) for maximum sharding ->
            # 87.5% reduction.
            mesh_shape = (8,)
            axis_names = ["model"]

    mesh = distribution_lib.DeviceMesh(
        shape=mesh_shape, axis_names=axis_names, devices=devices
    )
    print(f"Device Mesh: {mesh}")

    if strategy == "auto":
        sample_inputs = {
            "token_ids": np.ones((64, seq_len), dtype="int32"),
            "padding_mask": np.ones((64, seq_len), dtype="int32"),
        }
        # Pass the explicit 1D mesh to Parallax
        distribution = ParallaxDistribution(
            strategy=strategy,
            model_builder=model_builder,
            sample_inputs=sample_inputs,
            mesh=mesh,
        )
    else:
        # Pass the explicit mesh to Parallax
        distribution = ParallaxDistribution(strategy=strategy, mesh=mesh)

    distribution_lib.set_distribution(distribution)

    print("Loading OPT 125M...")
    model = model_builder()
    print("Model loaded.")

    model.compile(optimizer="adam")

    print_sharding_info(model, f"OPT-125M ({strategy.upper()})")

    x = ["Keras and Parallax make a great team."] * 64

    print(f"\nStarting model.fit() for {strategy.upper()}...")
    model.fit(x, epochs=1, batch_size=64, verbose=1)
    print("model.fit() completed.")

    # Clean up
    distribution_lib.set_distribution(None)
    del model
    gc.collect()
    print(f"✅ OPT 125M {strategy.upper()} test finished.")


if __name__ == "__main__":
    print("🌟 Starting Parallax OPT Demo Runs...")

    # Run FSDP
    # try:
    #     run_opt_test(strategy="fsdp")
    # except Exception as e:
    #     print(f"\n❌ FSDP failed: {e}")
    #     import traceback
    #     traceback.print_exc()

    # # Clear memory
    # gc.collect()

    # # Run AUTO (2D Mesh: 2x4)
    # try:
    #     run_opt_test(
    #         strategy="auto", mesh_shape=(2, 4), axis_names=["data", "model"]
    #     )
    # except Exception as e:
    #     print(f"\n❌ Auto Sharding (2x4) failed: {e}")
    #     import traceback
    #     traceback.print_exc()

    # Clear memory
    gc.collect()

    # Run AUTO (1D-style Mesh: 1x8)
    try:
        run_opt_test(
            strategy="auto", mesh_shape=(1, 8), axis_names=["data", "model"]
        )
    except Exception as e:
        print(f"\n❌ Auto Sharding (1x8) failed: {e}")
        import traceback

        traceback.print_exc()

    # Clear memory
    gc.collect()

    # Run DDP
    try:
        run_opt_test(strategy="ddp")
    except Exception as e:
        print(f"\n❌ DDP failed: {e}")
        import traceback

        traceback.print_exc()

    print("\n🏁 All OPT tests completed.")
