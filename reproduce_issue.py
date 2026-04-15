import os

os.environ["KERAS_BACKEND"] = "torch"
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from keras import layers
from keras import models
from keras.src.distribution import distribution_lib


def find_free_port():
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def worker(rank, world_size, port):
    os.environ.update(
        {
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": str(port),
            "RANK": str(rank),
            "WORLD_SIZE": str(world_size),
            "LOCAL_RANK": str(rank),
            "KERAS_TORCH_DEVICE": "cpu",
        }
    )
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    # 0. Setup
    model = models.Sequential([layers.Input(shape=(8,)), layers.Dense(2)])
    model.compile(optimizer="adam", loss="mse")
    model.build()

    x = torch.randn(4, 8)
    y = torch.randn(4, 2)
    data = (x, y)

    # 1. DataParallel
    print(f"Rank {rank}: Part 1 (DataParallel)")
    mesh_dp = distribution_lib.DeviceMesh((world_size,), ["batch"])
    dist_dp = distribution_lib.DataParallel(mesh_dp)
    with dist_dp.scope():
        model._setup_ddp()
        logs = model.train_step(data)
        print(f"Rank {rank}: Part 1 succeeded, logs: {logs}")

    # 2. ModelParallel
    print(f"Rank {rank}: Part 2 (ModelParallel)")
    mesh_mp = distribution_lib.DeviceMesh((world_size,), ["model"])
    layout_map = distribution_lib.LayoutMap(mesh_mp)
    layout_map[".*kernel"] = distribution_lib.TensorLayout([None, "model"])
    dist_mp = distribution_lib.ModelParallel(
        layout_map=layout_map, batch_dim_name=None
    )

    with dist_mp.scope():
        try:
            logs = model.train_step(data)
            print(f"Rank {rank}: Part 2 succeeded, logs: {logs}")
        except Exception as e:
            print(f"Rank {rank}: Part 2 failed with error: {e}")
            import traceback

            traceback.print_exc()

    dist.destroy_process_group()


if __name__ == "__main__":
    world_size = 2
    port = find_free_port()
    mp.spawn(worker, args=(world_size, port), nprocs=world_size, join=True)
