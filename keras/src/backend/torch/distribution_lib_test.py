import os
import random
import numpy as np
import pytest
import torch
import torch.multiprocessing as mp
import torch.distributed as dist

from keras.src import backend
from keras.src import layers
from keras.src import models
from keras.src import testing
from keras.src.distribution import distribution_lib

def _test_all_sections_worker(rank, world_size, port):
    import os
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)

    import keras
    keras.config.set_floatx("float32")
    from keras.src.backend.torch import distribution_lib as torch_dist_lib
    # 1. Setup Environment
    
    # 3. Distribution Backend Functions
    torch_dist_lib.initialize()
    
    # Test list_devices / get_device_count
    devices = torch_dist_lib.list_devices("cpu")
    assert len(devices) == world_size
    assert torch_dist_lib.get_device_count("cpu") == world_size
    
    # Test process management
    assert torch_dist_lib.num_processes() == world_size
    assert torch_dist_lib.process_id() == rank
    
    # Test DeviceMesh mapping
    keras_mesh = distribution_lib.DeviceMesh(
        shape=(1, world_size), 
        axis_names=("batch", "model"), 
        devices=[f"cpu:{i}" for i in range(world_size)]
    )
    torch_mesh = keras_mesh.backend_mesh
    assert torch_mesh.mesh_dim_names == ("batch", "model")
    assert torch_mesh.shape == (1, world_size)
    
    # Test TensorLayout mapping
    layout = distribution_lib.TensorLayout((None, "model"), keras_mesh)
    torch_mesh_from_layout, placements = torch_dist_lib._to_backend_layout(layout)
    assert torch_mesh_from_layout == torch_mesh
    # batch dim (0) -> Replicate, model dim (1) -> Shard(1)
    from torch.distributed.tensor import Replicate, Shard
    assert isinstance(placements[0], Replicate)
    assert isinstance(placements[1], Shard)
    assert placements[1].dim == 1

    # 4. Variables: Distribution Awareness
    from keras.src.distribution import LayoutMap
    layout_map = LayoutMap(keras_mesh)
    layout_map[".*dense.*kernel"] = (None, "model")
    
    mp_dist = distribution_lib.ModelParallel(layout_map=layout_map, batch_dim_name="batch")
    
    with mp_dist.scope():
        # Test variable initialization with ModelParallel
        v = keras.Variable(initializer="ones", shape=(16, 16), dtype="float32", name="dense_kernel")
        from torch.distributed.tensor import DTensor
        assert isinstance(v.value, torch.nn.Parameter)
        assert isinstance(v.value.data, DTensor)
        # Sharded on dim 1
        assert any(p.is_shard(dim=1) for p in v.value.data.placements)
        
        # Test direct assignment with redistribution
        new_val = torch.ones((16, 16)) * 2
        v.assign(new_val)
        # Check a local value to ensure it's distributed
        local_tensor = v.value.data.to_local()
        assert torch.allclose(local_tensor, torch.ones_like(local_tensor) * 2)

    # 5. TorchTrainer Architecture
    # 5.1 DataParallel (DDP)
    dp_dist = distribution_lib.DataParallel(devices=[f"cpu:{i}" for i in range(world_size)], auto_shard_dataset=False)
    with dp_dist.scope():
        model_dp = keras.Sequential([layers.Dense(1, input_shape=(16,), dtype="float32")])
        model_dp.compile(optimizer="sgd", loss="mse", metrics=["mae"])
        
        # Check DDP wrapping in make_train_function
        model_dp.make_train_function()
        assert hasattr(model_dp, "_ddp_model")
        assert isinstance(model_dp._ddp_model, torch.nn.parallel.DistributedDataParallel)
        
        # Test train_step with DDP
        x = np.random.normal(size=(16, 16)).astype("float32")
        y = np.random.normal(size=(16, 1)).astype("float32")
        model_dp.fit(x, y, epochs=1, batch_size=8, verbose=0)

    # 5.2 ModelParallel (DTensor)
    with mp_dist.scope():
        model_mp = keras.Sequential([layers.Dense(world_size, input_shape=(16,), dtype="float32")])
        # Initialize and verify forward pass works with DTensor
        x_dummy = torch.ones((1, 16))
        x_dummy = torch_dist_lib.distribute_data_input(
            x_dummy, mp_dist.get_data_layout(x_dummy.shape), mp_dist.batch_dim_name
        )
        output = model_mp(x_dummy)
        assert isinstance(output, DTensor)

    # 6. Cross-Framework Data Loading
    # Test DistributedSampler injection
    from torch.utils.data import DataLoader, TensorDataset
    dataset = TensorDataset(torch.randn(64, 16), torch.randn(64, 1))
    dataloader = DataLoader(dataset, batch_size=8)
    
    from keras.src.trainers.data_adapters import get_data_adapter
    with dp_dist.scope():
        adapter = get_data_adapter(dataloader)
        dist_dataloader = adapter.get_torch_dataloader()
        assert isinstance(dist_dataloader.sampler, torch.utils.data.distributed.DistributedSampler)
        assert dist_dataloader.sampler.num_replicas == world_size
        assert dist_dataloader.sampler.rank == rank

    # 7. Evaluation and Metric Aggregation
    # Test _sync_metrics
    with dp_dist.scope():
        model_dp.reset_metrics()
        # Manually update a metric variable
        metric = model_dp.metrics[0] # mae or loss
        var = metric.variables[0] # total
        var.assign(torch.tensor(float(rank + 1)))
        
        model_dp._sync_metrics()
        # Expected sum: 1 + 2 + ... + world_size
        expected_sum = sum(range(1, world_size + 1))
        assert torch.allclose(var.value, torch.tensor(float(expected_sum)))

    # Cleanup
    dist.destroy_process_group()

@pytest.mark.skipif(backend.backend() != "torch", reason="PyTorch backend only")
class TorchDistributionFullTest(testing.TestCase):
    def test_all_sections(self):
        world_size = 2
        port = random.randint(20000, 30000)
        # Set device to cpu for CI
        os.environ["KERAS_TORCH_DEVICE"] = "cpu"
        mp.spawn(_test_all_sections_worker, args=(world_size, port), nprocs=world_size, join=True)
