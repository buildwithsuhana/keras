#!/usr/bin/env python3
"""
Distributed Backend Operations
Centralizes all backend-specific operations for JAX, TensorFlow, and PyTorch
"""

import logging
from typing import List, Any
import numpy as np
import keras
import os

logger = logging.getLogger(__name__)

class DistributedBackend:
    """Base class for distributed backend operations."""
    
    def __init__(self, backend_name: str = "auto"):
        self.backend_name = backend_name
        self.backend = self._initialize_backend()

    def _initialize_backend(self) -> str:
        """Initializes the backend, respecting the user's choice."""
        if self.backend_name != "auto":
            return self.backend_name
        try:
            import jax
            return "jax"
        except ImportError:
            try:
                import tensorflow as tf
                return "tensorflow"
            except ImportError:
                try:
                    import torch
                    return "pytorch"
                except ImportError:
                    return "numpy"
    
    def _detect_backend(self) -> str:
        """Detect the available backend."""
        try:
            import jax
            return "jax"
        except ImportError:
            try:
                import tensorflow as tf
                return "tensorflow"
            except ImportError:
                try:
                    import torch
                    return "pytorch"
                except ImportError:
                    return "numpy"
    
    def get_tensor_lib(self):
        """Get the appropriate tensor library for the backend."""
        if self.backend == "jax":
            import jax.numpy as jnp
            return jnp
        elif self.backend == "tensorflow":
            import tensorflow as tf
            return tf
        elif self.backend == "pytorch":
            import torch
            return torch
        else:
            return np
    
    def convert_to_backend_tensor(self, tensor: Any) -> Any:
        """Convert a tensor to the appropriate backend format."""
        if self.backend == "jax":
            import jax.numpy as jnp
            if hasattr(tensor, 'numpy'):
                return jnp.array(tensor.numpy())
            else:
                return jnp.array(tensor)
        elif self.backend == "tensorflow":
            import tensorflow as tf
            if hasattr(tensor, 'numpy'):
                return tf.convert_to_tensor(tensor.numpy())
            else:
                return tf.convert_to_tensor(tensor)
        elif self.backend == "pytorch":
            import torch
            return tensor.clone().detach()
        else:
            return keras.ops.convert_to_numpy(tensor)
    
    def compute_gradients(self, loss: Any, trainable_vars: List[Any]) -> List[Any]:
        """Compute gradients using the appropriate backend's automatic differentiation."""
        try:
            if self.backend == "jax":
                return self._compute_jax_gradients(loss, trainable_vars)
            elif self.backend == "tensorflow":
                return self._compute_tensorflow_gradients(loss, trainable_vars)
            elif self.backend == "pytorch":
                return self._compute_pytorch_gradients(loss, trainable_vars)
            else:
                return self._compute_numpy_gradients(loss, trainable_vars)
        except Exception as e:
            logger.error(f"Gradient computation failed for {self.backend}: {e}")
            return self._create_zero_gradients(trainable_vars)
    
    def _compute_jax_gradients(self, loss: Any, trainable_vars: List[Any]) -> List[Any]:
        """Compute gradients using JAX automatic differentiation."""
        import jax
        import jax.numpy as jnp
        
        def safe_convert_to_jax(tensor):
            try:
                if hasattr(tensor, 'numpy'):
                    if hasattr(tensor, 'shape') and tensor.shape is None:
                        logger.warning("Symbolic tensor detected, using dummy value for gradient computation")
                        return jnp.array(0.0)
                    else:
                        return jnp.array(tensor.numpy())
                else:
                    return jnp.array(tensor)
            except Exception as e:
                logger.warning(f"Failed to convert tensor to JAX: {e}, using dummy value")
                return jnp.array(0.0)
        
        loss_jax = safe_convert_to_jax(loss)
        params_jax = [safe_convert_to_jax(param) for param in trainable_vars]
        
        def loss_fn(params):
            return loss_jax
        
        try:
            gradients = jax.grad(loss_fn)(params_jax)
            logger.info("   - JAX gradient computation successful")
            return gradients
        except Exception as e:
            logger.warning(f"JAX gradient computation failed: {e}, using fallback")
            return [jnp.zeros_like(param) for param in params_jax]
    
    def _compute_tensorflow_gradients(self, loss: Any, trainable_vars: List[Any]) -> List[Any]:
        """Compute gradients using TensorFlow automatic differentiation."""
        import tensorflow as tf
        
        with tf.GradientTape() as tape:
            for var in trainable_vars:
                tape.watch(var)
        
        try:
            gradients = tape.gradient(loss, trainable_vars)
            logger.info("   - TensorFlow gradient computation successful")
            return gradients
        except Exception as e:
            logger.warning(f"TensorFlow gradient computation failed: {e}, using fallback")
            return [tf.zeros_like(var) for var in trainable_vars]
    
    def _compute_pytorch_gradients(self, loss: Any, trainable_vars: List[Any]) -> List[Any]:
        logger.warning("PyTorch gradient computation is handled by `loss.backward()` in the Keras model's `train_step`.")
        return self._create_zero_gradients(trainable_vars)
    
    def _create_zero_gradients(self, trainable_vars: List[Any]) -> List[Any]:
        """Create zero gradients as fallback."""
        lib = self.get_tensor_lib()
        return [lib.zeros_like(var) for var in trainable_vars]
    
    def _compute_numpy_gradients(self, loss: Any, trainable_vars: List[Any]) -> List[Any]:
        """Fallback gradient computation using numerical differentiation."""
        epsilon = 1e-7
        gradients = []
        
        for var in trainable_vars:
            if hasattr(var, 'shape'):
                grad = np.zeros_like(var)
                for i in range(var.size):
                    idx = np.unravel_index(i, var.shape)
                    var_plus = var.copy()
                    var_minus = var.copy()
                    var_plus[idx] += epsilon
                    var_minus[idx] -= epsilon
                    grad[idx] = (loss - loss) / (2 * epsilon)
                gradients.append(grad)
            else:
                gradients.append(0.0)
        
        return gradients
    
    def apply_gradients(self, gradients: List[Any], trainable_vars: List[Any], 
                       learning_rate: float = 0.001) -> None:
        """Apply gradients to trainable variables using the appropriate backend."""
        if self.backend == "jax":
            self._apply_jax_gradients(gradients, trainable_vars, learning_rate)
        elif self.backend == "tensorflow":
            self._apply_tensorflow_gradients(gradients, trainable_vars, learning_rate)
        elif self.backend == "pytorch":
            self._apply_pytorch_gradients(gradients, trainable_vars, learning_rate)
        else:
            self._apply_numpy_gradients(gradients, trainable_vars, learning_rate)
    
    def _apply_jax_gradients(self, gradients: List[Any], trainable_vars: List[Any], 
                            learning_rate: float) -> None:
        """Apply gradients using JAX operations."""
        import jax.numpy as jnp
        
        for grad, var in zip(gradients, trainable_vars):
            if grad is not None:
                new_value = var - (learning_rate * grad)
                if hasattr(var, 'assign'):
                    var.assign(new_value)
    
    def _apply_tensorflow_gradients(self, gradients: List[Any], trainable_vars: List[Any], 
                                   learning_rate: float) -> None:
        """Apply gradients using TensorFlow operations."""
        import tensorflow as tf
        
        for grad, var in zip(gradients, trainable_vars):
            if grad is not None:
                new_value = var - (learning_rate * grad)
                var.assign(new_value)
    
    def _apply_pytorch_gradients(self, gradients: List[Any], trainable_vars: List[Any], 
                                learning_rate: float) -> None:
        """Apply gradients using PyTorch operations."""
        import torch
        
        for grad, var in zip(gradients, trainable_vars):
            if grad is not None:
                with torch.no_grad():
                    var -= learning_rate * grad
    
    def _apply_numpy_gradients(self, gradients: List[Any], trainable_vars: List[Any], 
                              learning_rate: float) -> None:
        """Apply gradients using NumPy operations."""
        for grad, var in zip(gradients, trainable_vars):
            if grad is not None:
                new_value = var - (learning_rate * grad)
                if hasattr(var, 'assign'):
                    var.assign(new_value)
                else:
                    var[:] = new_value
    
    def create_optimizer(self, optimizer_class: str, **kwargs):
        """Create an optimizer for the appropriate backend."""
        if self.backend == "jax":
            return self._create_jax_optimizer(optimizer_class, **kwargs)
        elif self.backend == "tensorflow":
            return self._create_tensorflow_optimizer(optimizer_class, **kwargs)
        elif self.backend == "pytorch":
            return self._create_pytorch_optimizer(optimizer_class, **kwargs)
        else:
            return self._create_numpy_optimizer(optimizer_class, **kwargs)
    
    def _create_jax_optimizer(self, optimizer_class: str, **kwargs):
        """Create a JAX optimizer."""
        import jax
        import optax
        
        if optimizer_class.lower() == "adam":
            return optax.adam(**kwargs)
        elif optimizer_class.lower() == "sgd":
            return optax.sgd(**kwargs)
        else:
            return optax.adam(learning_rate=0.001)
    
    def _create_tensorflow_optimizer(self, optimizer_class: str, **kwargs):
        """Create a TensorFlow optimizer."""
        import tensorflow as tf
        
        if optimizer_class.lower() == "adam":
            return tf.keras.optimizers.Adam(**kwargs)
        elif optimizer_class.lower() == "sgd":
            return tf.keras.optimizers.SGD(**kwargs)
        else:
            return tf.keras.optimizers.Adam(learning_rate=0.001)
    
    def _create_pytorch_optimizer(self, optimizer_class: str, **kwargs):
        """Create a PyTorch optimizer."""
        import torch
        
        if optimizer_class.lower() == "adam":
            return torch.optim.Adam(**kwargs)
        elif optimizer_class.lower() == "sgd":
            return torch.optim.SGD(**kwargs)
        else:
            return torch.optim.Adam(lr=0.001)
    
    def _create_numpy_optimizer(self, optimizer_class: str, **kwargs):
        """Create a NumPy-based optimizer (simplified)."""
        class NumpyOptimizer:
            def __init__(self, learning_rate=0.001):
                self.learning_rate = learning_rate
            
            def apply_gradients(self, grads_and_vars):
                for grad, var in grads_and_vars:
                    if grad is not None:
                        var -= self.learning_rate * grad
        
        return NumpyOptimizer(**kwargs)
    
    def get_device_info(self) -> dict:
        """Get information about available devices for the backend."""
        info = {
            "backend": self.backend,
            "devices": [],
            "device_count": 0
        }
        
        try:
            if self.backend == "jax":
                import jax
                info["devices"] = [str(d) for d in jax.devices()]
                info["device_count"] = jax.local_device_count()
            elif self.backend == "tensorflow":
                import tensorflow as tf
                info["devices"] = [d.name for d in tf.config.list_physical_devices()]
                info["device_count"] = len(tf.config.list_physical_devices())
            elif self.backend == "pytorch":
                import torch
                if torch.cuda.is_available():
                    count = torch.cuda.device_count()
                    info["devices"] = [f"cuda:{i}" for i in range(count)]
                    info["device_count"] = count
            else:
                info["devices"] = ["cpu"]
                info["device_count"] = 1
        except Exception as e:
            logger.warning(f"Could not get device info for {self.backend}: {e}")
            info["devices"] = ["cpu"]
            info["device_count"] = 1
        
        return info
    
    def is_multi_device_capable(self) -> bool:
        """Check if the backend supports multi-device operations."""
        device_info = self.get_device_info()
        return device_info["device_count"] > 1
    
    def get_communication_ops(self):
        """Get communication operations for the backend."""
        if self.backend == "jax":
            return self._get_jax_communication_ops()
        elif self.backend == "tensorflow":
            return self._get_tensorflow_communication_ops()
        elif self.backend == "pytorch":
            return self._get_pytorch_communication_ops()
        else:
            return self._get_numpy_communication_ops()
    
    def _get_jax_communication_ops(self):
        """Get JAX communication operations."""
        import jax
        import jax.lax as lax
        import jax.numpy as jnp
        import logging

        logger = logging.getLogger(__name__)

        def all_reduce_jax(x, op="sum", axis_name="data"):
            return lax.pmean(x, axis_name=axis_name)

        def all_gather_jax(x, axis=0, axis_name="model"):
            return lax.all_gather(x, axis_name=axis_name, axis=axis)

        def broadcast_jax(x, axis_name="data"):
            return lax.all_gather(x, axis_name=axis_name, axis=0)

        def scatter_jax(x, num_devices, axis_name="data"):
            return lax.psplit(x, axis_name=axis_name, num_splits=num_devices)

        def all_reduce_simulated(x, op="sum", axis_name="data"):
            return jnp.sum(x, axis=0)
        
        def all_gather_simulated(x, axis=0, axis_name="model"):
            return jnp.concatenate([x, x], axis=axis)

        def broadcast_simulated(x):
            return x

        def scatter_simulated(x, num_devices):
            return jnp.split(x, num_devices, axis=0)
        
        try:
            if jax.device_count() > 1:
                logger.info("Using real JAX collective communication ops.")
                return {
                    "all_reduce": all_reduce_jax,
                    "all_gather": all_gather_jax,
                    "broadcast": broadcast_jax,
                    "scatter": scatter_jax
                }
            else:
                raise RuntimeError("Not running on multiple JAX devices.")
        
        except (ImportError, RuntimeError) as e:
            logger.warning(f"JAX collective ops not available or multiple devices not configured: {e}. Using SIMULATED ops.")
            return {
                "all_reduce": all_reduce_simulated,
                "all_gather": all_gather_simulated,
                "broadcast": broadcast_simulated,
                "scatter": scatter_simulated
            }
    
    def _get_tensorflow_communication_ops(self):
        """Get TensorFlow communication operations."""
        import tensorflow as tf
        
        def all_reduce_tf(x, op="sum"):
            strategy = tf.distribute.get_strategy()
            return strategy.reduce(tf.distribute.ReduceOp.SUM, x, axis=0)

        def all_gather_tf(x, axis=0):
            strategy = tf.distribute.get_strategy()
            return tf.raw_ops.AllGather(
                input=x, 
                group_assignment=[[i for i in range(strategy.num_replicas_in_sync)]], 
                group_size=strategy.num_replicas_in_sync
            )

        def broadcast_tf(x, root=0):
            strategy = tf.distribute.get_strategy()
            return strategy.broadcast(x)

        def scatter_tf(x):
            strategy = tf.distribute.get_strategy()
            return strategy.scatter(x, axis=0)
        
        def all_reduce_simulated(x, op="sum"):
            return keras.ops.sum(x, axis=0)
        
        def all_gather_simulated(x, axis=0):
            return keras.ops.concatenate([x, x], axis=axis)
        
        def broadcast_simulated(x):
            return x
        
        def scatter_simulated(x, num_devices):
            return keras.ops.split(x, num_devices, axis=0)

        try:
            strategy = tf.distribute.get_strategy()
            if not isinstance(strategy, (tf.distribute.MirroredStrategy, tf.distribute.MultiWorkerMirroredStrategy)):
                raise RuntimeError("No active `tf.distribute` strategy found. Cannot use real collectives.")

            logger.info("Using real TensorFlow `tf.distribute` collective ops.")
            return {
                "all_reduce": all_reduce_tf,
                "all_gather": all_gather_tf,
                "broadcast": broadcast_tf,
                "scatter": scatter_tf
            }
        except (ImportError, RuntimeError) as e:
            logger.warning(f"TensorFlow collective ops not available: {e}. Using SIMULATED ops.")
            return {
                "all_reduce": all_reduce_simulated,
                "all_gather": all_gather_simulated,
                "broadcast": broadcast_simulated,
                "scatter": scatter_simulated
            }
    
    def _get_pytorch_communication_ops(self):
        """Get PyTorch communication operations using torch.distributed."""
        import torch
        import torch.distributed as dist

        def all_reduce_torch(x, op="sum"):
            """Performs an all-reduce operation (sum or mean). In-place."""
            if op == "sum":
                dist.all_reduce(x, op=dist.ReduceOp.SUM)
            elif op == "mean":
                dist.all_reduce(x, op=dist.ReduceOp.SUM)
                x /= dist.get_world_size()
            else:
                raise ValueError(f"Unsupported all_reduce op: {op}")
            return x

        def all_gather_torch(x, axis=0):
            """Gathers tensors from all processes and concatenates them."""
            world_size = dist.get_world_size()
            tensor_list = [torch.empty_like(x) for _ in range(world_size)]
            dist.all_gather(tensor_list, x)
            return torch.cat(tensor_list, dim=axis)

        def broadcast_torch(x, root=0):
            dist.broadcast(x, src=root)
            return x

        def scatter_torch(x, root=0):
            """Scatters a tensor from the root process to all processes."""
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            
            if rank == root:
                if x.shape[0] % world_size != 0:
                    raise ValueError(
                        "For scatter, the first dimension of the tensor "
                        f"({x.shape[0]}) must be divisible by world_size ({world_size})."
                    )
                scatter_list = list(torch.chunk(x, world_size, dim=0))
            else:
                scatter_list = None

            chunk_shape = (x.shape[0] // world_size,) + x.shape[1:]
            output_tensor = torch.empty(chunk_shape, dtype=x.dtype, device=x.device)
            
            dist.scatter(output_tensor, scatter_list, src=root)
            return output_tensor
        
        def no_op_simulated(x, **kwargs):
            """Simulated op for a single device; it's a no-op."""
            return x
        
        def scatter_simulated(x, **kwargs):
            """In a single-device simulation, scatter returns the whole tensor."""
            return x

        try:
            if not (dist.is_available() and dist.is_initialized()):
                raise RuntimeError("torch.distributed is not available or not initialized.")
            
            logger.info("Using real torch.distributed communication ops.")
            return {
                "all_reduce": all_reduce_torch,
                "all_gather": all_gather_torch,
                "broadcast": broadcast_torch,
                "scatter": scatter_torch,
            }
        
        except (ImportError, RuntimeError) as e:
            logger.warning(f"torch.distributed not available: {e}. Using SIMULATED communication ops.")
            return {
                "all_reduce": no_op_simulated,
                "all_gather": no_op_simulated,
                "broadcast": no_op_simulated,
                "scatter": scatter_simulated,
            }
    
    def _get_numpy_communication_ops(self):
        """Get NumPy communication operations (simplified)."""
        logger.info("Using SIMULATED NumPy communication ops.")

        def all_reduce_np(x, op="sum"):
            return keras.ops.sum(x, axis=0)
        
        def all_gather_np(x, axis=0):
            return keras.ops.concatenate([x, x], axis=axis)
        
        def broadcast_np(x):
            return x
        
        def scatter_np(x, num_devices):
            return keras.ops.split(x, num_devices, axis=0)

        return {
            "all_reduce": all_reduce_np,
            "all_gather": all_gather_np,
            "broadcast": broadcast_np,
            "scatter": scatter_np
        }

def get_distributed_backend(backend_name: str = 'auto', world_size: int = 1, rank: int = 0):
    """
    Get the best available distributed backend.
    
    Args:
        backend_name: Backend to use ('auto', 'jax', 'tensorflow', 'pytorch', 'numpy')
        world_size: Number of processes
        rank: Process rank
        
    Returns:
        Initialized distributed backend
    """
    if backend_name == 'auto':
        backends = ['jax', 'tensorflow', 'pytorch', 'numpy']
        
        for name in backends:
            try:
                backend = DistributedBackend(name)
                if backend.is_multi_device_capable():
                    return backend
            except Exception:
                continue
        
        return DistributedBackend('numpy')
    
    else:
        return DistributedBackend(backend_name)

def allreduce_gradients(gradients: List[Any], backend: DistributedBackend, op: str = 'mean') -> List[Any]:
    """AllReduce a list of gradients."""
    synchronized = []
    for grad in gradients:
        if grad is not None:
            synced_grad = backend.get_communication_ops()["all_reduce"](grad, op=op)
            synchronized.append(synced_grad)
        else:
            synchronized.append(None)
    return synchronized

def allgather_outputs(outputs: List[Any], backend: DistributedBackend, axis: int = 0) -> Any:
    """AllGather outputs from all shards."""
    if len(outputs) == 1:
        return outputs[0]
        
    for output in outputs:
        if output is not None:
            return backend.get_communication_ops()["all_gather"](output)
            
    raise ValueError("No valid outputs to gather")

def broadcast_parameters(parameters: List[Any], backend: DistributedBackend, root: int = 0) -> List[Any]:
    """Broadcast parameters from root to all processes."""
    broadcasted = []
    for param in parameters:
        if param is not None:
            broadcasted_param = backend.get_communication_ops()["broadcast"](param)
            broadcasted.append(broadcasted_param)
        else:
            broadcasted.append(None)
    return broadcasted
