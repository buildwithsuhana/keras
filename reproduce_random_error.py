import os
import sys

import torch
import torch.distributed as dist

# Configure keras backend
os.environ["KERAS_BACKEND"] = "torch"
os.environ["KERAS_TORCH_DEVICE"] = "cpu"


def _test_fn(rank, world_size):
    try:
        sys.path.insert(0, "/Users/suhanaaa/keras")
        import keras
        from keras import distribution
        from keras import ops

        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(rank)
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29505"

        print(f"[PROCESS {rank}] Initializing distribution")
        keras.distribution.initialize()

        mesh = distribution.DeviceMesh(
            shape=(world_size,),
            axis_names=("model",),
            devices=["cpu"] * world_size,
        )

        layout_map = distribution.LayoutMap(mesh)
        layout_map[".*w"] = distribution.TensorLayout(("model", None), mesh)

        dist_strategy = distribution.ModelParallel(
            layout_map=layout_map,
            batch_dim_name="model",
        )

        class MyModel(keras.Model):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.w = self.add_weight(
                    shape=(4, 4), initializer="ones", name="w"
                )

            def call(self, x):
                print(f"[PROCESS {rank}] w type: {type(self.w.value)}")
                # Generate random noise - this will be a local torch.Tensor
                noise = ops.random.normal((4, 4))
                print(f"[PROCESS {rank}] noise type: {type(noise)}")

                # Mixing DTensor and Tensor in addition
                print(f"[PROCESS {rank}] Attempting addition with noise")
                return self.w + noise

        with dist_strategy.scope():
            model = MyModel()
            x = ops.ones((1, 4))
            print(f"[PROCESS {rank}] Running model")
            y = model(x)
            print(f"[PROCESS {rank}] Output shape: {y.shape}")

        if dist.is_initialized():
            dist.destroy_process_group()
    except Exception as e:
        print(f"[PROCESS {rank}] FAILED with error: {e}")
        if dist.is_initialized():
            dist.destroy_process_group()


def test_reproduction():
    world_size = 2
    torch.multiprocessing.spawn(
        _test_fn, args=(world_size,), nprocs=world_size, join=True
    )


if __name__ == "__main__":
    test_reproduction()
