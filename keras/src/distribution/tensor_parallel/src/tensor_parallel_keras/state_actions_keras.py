"""
State actions for Keras Tensor Parallel
Handles parameter splitting, gathering, and reduction operations
"""
from typing import Any, Sequence
import keras

class StateActionKeras:
    """Base class for state actions."""
    
    def __call__(self, tensor: Any, rank: int) -> Any:
        """Apply the action to a tensor."""
        raise NotImplementedError
        
    def undo(self, tensors: Sequence[Any]) -> Any:
        """Reverse the action to reconstruct the original tensor."""
        raise NotImplementedError


class SplitKeras(StateActionKeras):
    """Split a tensor along a specified dimension."""
    
    def __init__(self, world_size: int, dim: int, sharding_type: str = "auto"):
        self.world_size = world_size
        self.dim = dim
        self.sharding_type = sharding_type
        
        if dim == -1 and sharding_type != "auto":
            if sharding_type == "row":
                self.dim = 0
            elif sharding_type == "column":
                self.dim = 1
        
    def __call__(self, tensor: Any, rank: int) -> Any:
        """Split tensor and return the portion for this rank."""
        if self.dim == -1:
            dim = keras.ops.ndim(tensor) - 1
        else:
            dim = self.dim
            
        total_size = tensor.shape[dim]
        split_size = total_size // self.world_size
        remainder = total_size % self.world_size
        
        start_idx = rank * split_size + min(rank, remainder)
        end_idx = start_idx + split_size + (1 if rank < remainder else 0)
        
        slices = [slice(None)] * keras.ops.ndim(tensor)
        slices[dim] = slice(start_idx, end_idx)
        return tensor[tuple(slices)]
            
    def undo(self, tensors: Sequence[Any]) -> Any:
        """Concatenate split tensors back together."""
        if self.dim == -1:
            dim = keras.ops.ndim(tensors[0]) - 1
        else:
            dim = self.dim
            
        return keras.ops.concatenate(tensors, axis=dim)


class GatherKeras(StateActionKeras):
    """Gather tensors from all ranks along a specified dimension."""
    
    def __init__(self, world_size: int, dim: int):
        self.world_size = world_size
        self.dim = dim
        
    def __call__(self, tensor: Any, rank: int) -> Any:
        """Return the tensor as-is (gathering happens in communication layer)."""
        return tensor
        
    def undo(self, tensors: Sequence[Any]) -> Any:
        """Concatenate gathered tensors."""
        if self.dim == -1:
            dim = keras.ops.ndim(tensors[0]) - 1
        else:
            dim = self.dim
            
        return keras.ops.concatenate(tensors, axis=dim)


class SumKeras(StateActionKeras):
    """Sum tensors from all ranks."""
    
    def __init__(self, world_size: int):
        self.world_size = world_size
        
    def __call__(self, tensor: Any, rank: int) -> Any:
        """Return the tensor as-is (summing happens in communication layer)."""
        return tensor
        
    def undo(self, tensors: Sequence[Any]) -> Any:
        """Sum the tensors."""
        return sum(tensors)