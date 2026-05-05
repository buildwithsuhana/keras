import os
import sys

import torch
import torch.distributed as dist

# Import and configure keras backend FIRST before any other imports
os.environ["KERAS_BACKEND"] = "torch"


def _test_fn(rank, world_size):
    try:
        # CRITICAL: Add local Keras code to path in EACH spawned process
        # so sys.path modifications in parent won't be inherited
        import numpy as np

        sys.path.insert(0, "/Users/suhanaaa/keras")

        # Now import keras (local fixed version) and keras-hub (preinstalled)
        import keras
        keras.utils.set_random_seed(42)
        import keras_hub
        import keras.distribution

        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(rank)
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29506"

        print(f"[PROCESS {rank}] Initializing distribution")
        keras.distribution.initialize()

        # Force CPU to avoid MPS issues with distributed operations
        devices = ["cpu:0", "cpu:1"]
        print(f"[PROCESS {rank}] Using devices: {devices}")

        mesh = keras.distribution.DeviceMesh(
            shape=(world_size,),
            axis_names=("model",),
            devices=devices[:world_size],
        )

        # ModelParallel strategy with non-overlapping patterns
        layout_map = keras.distribution.LayoutMap(mesh)

        # Keep embeddings replicated to avoid unbind issues
        # Use specific non-overlapping paths
        layout_map["embeddings/token_embedding/.*"] = (
            keras.distribution.TensorLayout((None, None), mesh)
        )
        layout_map["embeddings/position_embedding/.*"] = (
            keras.distribution.TensorLayout((None, None), mesh)
        )

        # Shard attention query/key/value projections for model parallelism
        layout_map[".*attention.*query.*kernel"] = (
            keras.distribution.TensorLayout(("model", None), mesh)
        )
        layout_map[".*attention.*key.*kernel"] = (
            keras.distribution.TensorLayout(("model", None), mesh)
        )
        layout_map[".*attention.*value.*kernel"] = (
            keras.distribution.TensorLayout(("model", None), mesh)
        )

        distribution = keras.distribution.ModelParallel(
            layout_map=layout_map,
            batch_dim_name="model",
            auto_shard_dataset=False,
        )

        with distribution.scope():
            print(f"[PROCESS {rank}] Creating model")
            # Load OPT 125m model from preset as requested
            model = keras_hub.models.OPTBackbone.from_preset("opt_125m_en")

            # Disable dropout for comparison
            for layer in model.layers:
                if hasattr(layer, "dropout"):
                    layer.dropout = 0.0

            # Compile model
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.001, epsilon=1e-7), 
                loss="sparse_categorical_crossentropy"
            )

            # Load shared data
            import data_utils
            x, y = data_utils.load_data()

            print(f"[PROCESS {rank}] Capturing initial weights (first 10)")
            # In DTensor, get_weights() is slow.
            w_list = model.get_weights()
            initial_weights = [np.array(w) for w in w_list[:10]]

            print(f"[PROCESS {rank}] Starting fit")
            history = model.fit(x, y, epochs=1, steps_per_epoch=1)
            loss = history.history["loss"][0]
            print(f"[PROCESS {rank}] Fit completed. Loss: {loss}")

            print(f"[PROCESS {rank}] Capturing final weights (first 10)")
            w_list_final = model.get_weights()
            final_weights = [np.array(w) for w in w_list_final[:10]]
            
            if rank == 0:
                print(f"[PROCESS {rank}] Saving results")
                weight_updates = [f - i for f, i in zip(final_weights, initial_weights)]
                np.savez("torch_results.npz", loss=loss, weight_updates=np.array(weight_updates, dtype=object))
                print("Torch results saved to torch_results.npz")

        if torch.distributed.is_initialized():
            dist.destroy_process_group()
    except Exception as e:
        print(f"[PROCESS {rank}] FAILED with error: {e}")
        import traceback

        traceback.print_exc()
        try:
            if torch.distributed.is_initialized():
                dist.destroy_process_group()
        except:
            pass
        raise e


def test_model_parallel_fit():
    world_size = 2
    print(f"Starting test with world_size={world_size}")
    torch.multiprocessing.spawn(
        _test_fn, args=(world_size,), nprocs=world_size, join=True
    )


if __name__ == "__main__":
    test_model_parallel_fit()