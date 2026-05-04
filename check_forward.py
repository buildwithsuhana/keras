import os
import sys
import numpy as np
import json
import subprocess
import textwrap

def run_backend_forward(backend):
    print(f"--- Running {backend.upper()} backend forward pass (distributed) ---")
    
    inner_code = textwrap.dedent(f"""
        import os
        os.environ["KERAS_BACKEND"] = "{backend}"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["KERAS_BACKEND_DEVICE"] = "cpu"
        
        import sys
        import numpy as np
        import json
        import keras
        import keras_hub

        # Disable Dropout for consistency
        keras.config.disable_interactive_logging()
        keras.utils.set_random_seed(42)

        if "{backend}" == "torch":
            keras.distribution.initialize()

        world_size = 2
        devices = keras.distribution.list_devices("cpu")
        if len(devices) < world_size:
            devices = ["cpu:0"] * world_size
        else:
            devices = devices[:world_size]

        mesh = keras.distribution.DeviceMesh(
            shape=(world_size,),
            axis_names=("model",),
            devices=devices,
        )

        layout_map = keras.distribution.LayoutMap(mesh)
        layout_map["embeddings/token_embedding/.*"] = keras.distribution.TensorLayout((None, None), mesh)
        layout_map["embeddings/position_embedding/.*"] = keras.distribution.TensorLayout((None, None), mesh)
        layout_map[".*attention.*query.*kernel"] = keras.distribution.TensorLayout(("model", None), mesh)
        layout_map[".*attention.*key.*kernel"] = keras.distribution.TensorLayout(("model", None), mesh)
        layout_map[".*attention.*value.*kernel"] = keras.distribution.TensorLayout(("model", None), mesh)

        distribution = keras.distribution.ModelParallel(
            layout_map=layout_map,
            batch_dim_name="model",
            auto_shard_dataset=False,
        )

        with distribution.scope():
            model = keras_hub.models.OPTBackbone.from_preset("opt_125m_en")
            model.load_weights("initial_weights.weights.h5")

            data = np.load("test_data.npz")
            x_tokens = data["x_tokens"][:2]
            x_mask = data["x_mask"][:2]
            
            if "{backend}" == "torch":
                rank = int(os.environ.get("RANK", 0))
                world_size = int(os.environ.get("WORLD_SIZE", 1))
                samples_per_rank = len(x_tokens) // world_size
                start = rank * samples_per_rank
                end = (rank + 1) * samples_per_rank
                x = {{"token_ids": x_tokens[start:end], "padding_mask": x_mask[start:end]}}
            else:
                x = {{"token_ids": x_tokens, "padding_mask": x_mask}}

            # We use model(x) directly
            output = model(x, training=False)
            output_np = keras.ops.convert_to_numpy(output)
    """)
    
    env = os.environ.copy()
    env["PYTHONPATH"] = os.getcwd() + (":" + env.get("PYTHONPATH", "") if env.get("PYTHONPATH") else "")
    
    if backend == "jax":
        full_code = inner_code + textwrap.dedent("""
            print(f"OUTPUT_SAMPLE: {output_np[0].flatten()[:5].tolist()}")
        """)
        env["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
        result = subprocess.run([sys.executable, "-c", full_code], env=env, capture_output=True, text=True)
    elif backend == "torch":
        torch_wrapper = textwrap.dedent(f"""
            import os
            import sys
            import numpy as np
            import json
            import torch.multiprocessing as mp

            def _worker(rank, world_size, return_dict):
                os.environ["RANK"] = str(rank)
                os.environ["WORLD_SIZE"] = str(world_size)
                os.environ["LOCAL_RANK"] = str(rank)
                os.environ["MASTER_ADDR"] = "localhost"
                os.environ["MASTER_PORT"] = "29568"
                
                {textwrap.indent(inner_code, '                ').strip()}
                
                if rank == 0:
                    # In rank 0, output_np should be the first sample
                    return_dict["output"] = output_np.flatten()[:5].tolist()

            if __name__ == "__main__":
                manager = mp.Manager()
                return_dict = manager.dict()
                mp.spawn(_worker, args=(2, return_dict), nprocs=2, join=True)
                print(f"OUTPUT_SAMPLE: {{return_dict['output']}}")
        """)
        result = subprocess.run([sys.executable, "-c", torch_wrapper], env=env, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"{backend} failed")
        print(result.stderr)
        print(result.stdout)
        return None
    
    for line in result.stdout.splitlines():
        if "OUTPUT_SAMPLE:" in line:
            parts = line.split("OUTPUT_SAMPLE:", 1)
            return json.loads(parts[1].strip())
    return None

if __name__ == "__main__":
    torch_out = run_backend_forward("torch")
    jax_out = run_backend_forward("jax")
    
    if torch_out and jax_out:
        torch_out = np.array(torch_out)
        jax_out = np.array(jax_out)
        diff = np.abs(torch_out - jax_out)
        print(f"\nTorch output sample (rank 0, first sample): {torch_out}")
        print(f"JAX output sample (first sample):           {jax_out}")
        print(f"Max difference:                             {np.max(diff):.8e}")
    else:
        print("Failed to get outputs")
