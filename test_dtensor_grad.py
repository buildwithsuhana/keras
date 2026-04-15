import os

os.environ["KERAS_BACKEND"] = "torch"
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.tensor import DTensor
from torch.distributed.tensor import Replicate

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

    # Setup
    model = models.Sequential([layers.Input(shape=(8,)), layers.Dense(2)])
    model.compile(optimizer="adam", loss="mse")
    model.build()

    x_local = torch.randn(4, 8)
    y_local = torch.randn(4, 2)

    # ModelParallel scope
    mesh_mp = distribution_lib.DeviceMesh((world_size,), ["model"])
    # Map to Torch DeviceMesh
    from keras.src.backend.torch.distribution_lib import _to_backend_mesh

    torch_mesh = _to_backend_mesh(mesh_mp)

    # Manually create DTensor inputs to see what happens
    x = DTensor.from_local(x_local, torch_mesh, [Replicate()])
    y = DTensor.from_local(y_local, torch_mesh, [Replicate()])

    model.zero_grad()
    y_pred = model(x)
    loss = torch.mean((y_pred - y) ** 2)

    print(f"Rank {rank}: loss type: {type(loss)}")
    loss.backward()

    kernel = model.layers[0].kernel.value
    print(f"Rank {rank}: kernel grad: {kernel.grad}")

    dist.destroy_process_group()


if __name__ == "__main__":
    world_size = 2
    port = find_free_port()
    mp.spawn(worker, args=(world_size, port), nprocs=world_size, join=True)
