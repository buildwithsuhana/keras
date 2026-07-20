import jax
import numpy as np
from jax.sharding import Mesh

devices = jax.devices()
mesh = Mesh(
    np.array(devices).reshape((len(devices) // 2, 2)), ("data", "model")
)

with mesh:
    # How to get the current mesh?
    try:
        active_mesh = jax.sharding.Mesh.get_active_mesh()
        print(f"get_active_mesh: {active_mesh}")
    except Exception as e:
        print(f"get_active_mesh failed: {e}")

    # Try another way
    try:
        # In older JAX, it might be in jax.interpreters.pxla
        from jax.interpreters import pxla

        print(
            f"pxla.thread_resources.env.mesh: {pxla.thread_resources.env.mesh}"
        )
    except Exception as e:
        print(f"pxla way failed: {e}")

print(f"JAX version: {jax.__version__}")
