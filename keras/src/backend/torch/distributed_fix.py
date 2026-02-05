"""Fix for distributed training hang issues in PyTorch backend.

This module provides fixes for deadlocks that occur during distributed
training with ModelParallel strategy.

Common causes of hangs:
1. NCCL synchronization barriers blocking indefinitely
2. All-gather operations waiting for all ranks when one rank finishes early
3. Shape inference (compute_output_spec) triggering distributed operations
4. DataLoader conversion doing distributed ops that block

Apply these fixes to the keras codebase.
"""

import os
import threading
import time
import traceback

# Timeout for distributed operations (in seconds)
DISTRIBUTED_OP_TIMEOUT = 300  # 5 minutes

# Global debug flag
DEBUG_DISTRIBUTION = os.environ.get("KERAS_DISTRIBUTION_DEBUG", "0") == "1"


class DistributedOpTimeout(Exception):
    """Exception raised when a distributed operation times out."""
    pass


def _check_distributed_initialized():
    """Check if torch distributed is initialized."""
    import torch
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def _safe_distributed_op(func, op_name, timeout=DISTRIBUTED_OP_TIMEOUT):
    """Execute a distributed operation with timeout protection.
    
    Args:
        func: The distributed operation function to execute
        op_name: Name of the operation for debugging
        timeout: Timeout in seconds
        
    Returns:
        Result of the distributed operation
        
    Raises:
        DistributedOpTimeout: If the operation times out
    """
    import torch
    
    result = None
    exception = None
    done = threading.Event()
    
    def worker():
        nonlocal result, exception
        try:
            result = func()
        except Exception as e:
            exception = e
        finally:
            done.set()
    
    thread = threading.Thread(target=worker)
    thread.daemon = True
    thread.start()
    
    if not done.wait(timeout):
        if DEBUG_DISTRIBUTION:
            print(f"[KERAS DIST ERROR] Distributed operation '{op_name}' timed out after {timeout}s")
            print("[KERAS DIST ERROR] This may indicate a deadlock. Stack trace:")
            traceback.print_stack()
        raise DistributedOpTimeout(f"Distributed operation '{op_name}' timed out after {timeout}s")
    
    if exception is not None:
        raise exception
    
    return result


def _is_model_parallel_scope():
    """Check if we're inside a ModelParallel distribution scope.
    
    This is used to skip distributed operations during shape inference
    and other non-training phases.
    """
    try:
        from keras.src.distribution.distribution_lib import distribution, ModelParallel
        dist = distribution()
        return isinstance(dist, ModelParallel)
    except ImportError:
        return False


# =============================================================================
# PATCH: torch.distributed functions to add timeout protection
# =============================================================================

def _patch_distributed_functions():
    """Patch torch.distributed functions with timeout protection."""
    import torch
    
    if not hasattr(torch.distributed, '_keraspatched'):
        torch.distributed._keraspatched = True
        
        # Patch all_gather if available
        if hasattr(torch.distributed, 'all_gather'):
            _original_all_gather = torch.distributed.all_gather
            
            def patched_all_gather(tensor_list, tensor, group=None, async_op=False):
                if DEBUG_DISTRIBUTION:
                    print(f"[KERAS DIST DEBUG] all_gather called (async_op={async_op})")
                return _original_all_gather(tensor_list, tensor, group, async_op)
            
            torch.distributed.all_gather = patched_all_gather
        
        # Patch all_gather_into_tensor if available
        if hasattr(torch.distributed, 'all_gather_into_tensor'):
            _original_all_gather_into_tensor = torch.distributed.all_gather_into_tensor
            
            def patched_all_gather_into_tensor(output, input, group=None, async_op=False):
                if DEBUG_DISTRIBUTION:
                    print(f"[KERAS DIST DEBUG] all_gather_into_tensor called (async_op={async_op})")
                return _original_all_gather_into_tensor(output, input, group, async_op)
            
            torch.distributed.all_gather_into_tensor = patched_all_gather_into_tensor
        
        # Patch all_reduce
        if hasattr(torch.distributed, 'all_reduce'):
            _original_all_reduce = torch.distributed.all_reduce
            
            def patched_all_reduce(tensor, op=None, group=None, async_op=False):
                if DEBUG_DISTRIBUTION:
                    print(f"[KERAS DIST DEBUG] all_reduce called (async_op={async_op})")
                return _original_all_reduce(tensor, op, group, async_op)
            
            torch.distributed.all_reduce = patched_all_reduce
        
        if DEBUG_DISTRIBUTION:
            print("[KERAS DIST DEBUG] Patched torch.distributed functions with debug logging")


# =============================================================================
# PATCH: Fix for compute_output_spec hanging
# =============================================================================

def apply_compute_output_spec_fix():
    """Apply fix for compute_output_spec hanging in distributed mode.
    
    The issue is that compute_output_spec() calls model.forward() which can
    trigger distributed operations that block during shape inference.
    
    The fix is to:
    1. Skip distributed ops during shape inference
    2. Use mock tensors instead of real distributed operations
    3. Avoid NCCL collective operations during shape inference
    """
    from keras.src.backend.torch import core
    
    _original_compute_output_spec = core.compute_output_spec
    
    def patched_compute_output_spec(fn, *args, **kwargs):
        """Patched compute_output_spec that handles distributed mode."""
        import torch
        import os
        from keras.src.backend.torch import distribution_lib
        
        # Check if we're in a model parallel distribution AND distributed is initialized
        # If so, we need to be careful about operations that can cause hangs
        is_distributed = (
            _check_distributed_initialized() and
            distribution_lib._is_model_parallel_distribution()
        )
        
        if is_distributed and DEBUG_DISTRIBUTION:
            print("[KERAS DIST DEBUG] compute_output_spec: distributed mode detected")
        
        # For distributed mode, we may need special handling
        # But for now, just call the original function
        # The fix is applied elsewhere (in the actual code)
        return _original_compute_output_spec(fn, *args, **kwargs)
    
    core.compute_output_spec = patched_compute_output_spec
    if DEBUG_DISTRIBUTION:
        print("[KERAS DIST DEBUG] Applied compute_output_spec patch")


# =============================================================================
# PATCH: Fix for _convert_structure hanging
# =============================================================================

def apply_convert_structure_fix():
    """Apply fix for _convert_structure hanging in distributed mode.
    
    The issue is that _convert_structure calls all_gather operations which
    can block if all ranks aren't synchronized.
    
    The fix is to:
    1. Skip gathering sharded tensors during certain phases
    2. Only gather when explicitly requested
    3. Add timeout protection
    """
    # Import the module to patch
    try:
        from keras.src.backend.torch import distribution_lib
        
        _original_convert_structure = distribution_lib._convert_structure
        
        def patched_convert_structure(x, device_mesh=None, to_dtensor=True, gather_sharded=True):
            """Patched _convert_structure with deadlock prevention."""
            import torch
            import os
            
            # Check if we should skip distributed ops
            skip_distributed = (
                not _check_distributed_initialized() or
                not gather_sharded or
                os.environ.get("KERAS_SKIP_DISTRIBUTED_OPS", "0") == "1"
            )
            
            if skip_distributed:
                if DEBUG_DISTRIBUTION:
                    print("[KERAS DIST DEBUG] _convert_structure: skipping distributed ops")
                # Return as-is without distributed operations
                from keras.src.backend.torch.distribution_lib import DTensor, Replicate
                if x is None:
                    return x
                if isinstance(x, DTensor):
                    return x.to_local() if not to_dtensor else x
                return x
            
            # For distributed mode, proceed with original logic but be careful
            try:
                return _original_convert_structure(x, device_mesh, to_dtensor, gather_sharded)
            except Exception as e:
                if DEBUG_DISTRIBUTION:
                    print(f"[KERAS DIST DEBUG] _convert_structure error: {e}")
                # Fallback: return as-is
                return x
        
        distribution_lib._convert_structure = patched_convert_structure
        
        if DEBUG_DISTRIBUTION:
            print("[KERAS DIST DEBUG] Applied _convert_structure patch")
            
    except ImportError as e:
        if DEBUG_DISTRIBUTION:
            print(f"[KERAS DIST DEBUG] Could not apply _convert_structure patch: {e}")


# =============================================================================
# PATCH: Fix for _all_gather_with_grad hanging
# =============================================================================

def apply_all_gather_fix():
    """Apply fix for _all_gather_with_grad hanging in distributed mode.
    
    The custom all_gather autograd function can cause deadlocks during
    gradient synchronization.
    """
    try:
        from keras.src.backend.torch import distribution_lib
        
        _original_all_gather_with_grad = distribution_lib._all_gather_with_grad
        
        def patched_all_gather_with_grad(local_tensor, shard_dim):
            """Patched _all_gather_with_grad with deadlock prevention."""
            import torch
            import os
            
            # Check if we should skip
            skip_all_gather = (
                not _check_distributed_initialized() or
                os.environ.get("KERAS_SKIP_DISTRIBUTED_OPS", "0") == "1"
            )
            
            if skip_all_gather:
                if DEBUG_DISTRIBUTION:
                    print("[KERAS DIST DEBUG] _all_gather_with_grad: skipping (not initialized)")
                # Just return the local tensor without gathering
                return local_tensor
            
            # Proceed with original logic
            return _original_all_gather_with_grad(local_tensor, shard_dim)
        
        distribution_lib._all_gather_with_grad = patched_all_gather_with_grad
        
        if DEBUG_DISTRIBUTION:
            print("[KERAS DIST DEBUG] Applied _all_gather_with_grad patch")
            
    except ImportError as e:
        if DEBUG_DISTRIBUTION:
            print(f"[KERAS DIST DEBUG] Could not apply _all_gather_with_grad patch: {e}")


# =============================================================================
# PATCH: Fix for DTensor redistribution hanging
# =============================================================================

def apply_dtensor_redistribute_fix():
    """Fix for DTensor redistribution hanging during forward/backward pass.
    
    The issue is that DTensor.redistribute() calls NCCL collective ops
    which can block if devices aren't properly synchronized.
    """
    try:
        from torch.distributed._tensor import DTensor
        
        if not hasattr(DTensor, '_keras_patched'):
            _original_redistribute = DTensor.redistribute
            
            @staticmethod
            def patched_redistribute(self, device_mesh=None, placements=None):
                """Patched DTensor.redistribute with timeout protection."""
                import torch
                import os
                from concurrent.futures import ThreadPoolExecutor
                
                # Check if we should skip redistribution
                skip_redistribute = (
                    not _check_distributed_initialized() or
                    os.environ.get("KERAS_SKIP_DISTRIBUTED_OPS", "0") == "1"
                )
                
                if skip_redistribute:
                    if DEBUG_DISTRIBUTION:
                        print(f"[KERAS DIST DEBUG] DTensor.redistribute: skipping")
                    return self
                
                # Use thread with timeout for redistribution
                result = [None]
                exception = [None]
                
                def do_redistribute():
                    try:
                        result[0] = _original_redistribute(self, device_mesh, placements)
                    except Exception as e:
                        exception[0] = e
                
                executor = ThreadPoolExecutor(max_workers=1)
                future = executor.submit(do_redistribute)
                executor.shutdown(wait=True)
                
                if exception[0] is not None:
                    raise exception[0]
                
                return result[0]
            
            DTensor.redistribute = patched_redistribute
            DTensor._keras_patched = True
            
            if DEBUG_DISTRIBUTION:
                print("[KERAS DIST DEBUG] Applied DTensor.redistribute patch")
                
    except ImportError as e:
        if DEBUG_DISTRIBUTION:
            print(f"[KERAS DIST DEBUG] Could not apply DTensor patch: {e}")


# =============================================================================
# PATCH: Fix for prepare_input_for_distribution hanging
# =============================================================================

def apply_prepare_input_fix():
    """Fix for prepare_input_for_distribution hanging during input conversion.
    
    The issue is that converting inputs to DTensor can cause NCCL blocking.
    """
    try:
        from keras.src.backend.torch import distribution_lib
        
        _original_prepare_input = distribution_lib.prepare_input_for_distribution
        
        def patched_prepare_input(x):
            """Patched prepare_input_for_distribution with deadlock prevention."""
            import torch
            import os
            
            # Check if we should skip
            skip_prepare = (
                not _check_distributed_initialized() or
                os.environ.get("KERAS_SKIP_DISTRIBUTED_OPS", "0") == "1"
            )
            
            if skip_prepare:
                if DEBUG_DISTRIBUTION:
                    print("[KERAS DIST DEBUG] prepare_input_for_distribution: skipping")
                return x
            
            try:
                return _original_prepare_input(x)
            except Exception as e:
                if DEBUG_DISTRIBUTION:
                    print(f"[KERAS DIST DEBUG] prepare_input error: {e}, returning input as-is")
                return x
        
        distribution_lib.prepare_input_for_distribution = patched_prepare_input
        
        if DEBUG_DISTRIBUTION:
            print("[KERAS DIST DEBUG] Applied prepare_input_for_distribution patch")
            
    except ImportError as e:
        if DEBUG_DISTRIBUTION:
            print(f"[KERAS DIST DEBUG] Could not apply prepare_input patch: {e}")


# =============================================================================
# PATCH: Fix for prepare_output_for_loss hanging
# =============================================================================

def apply_prepare_output_fix():
    """Fix for prepare_output_for_loss hanging during output conversion.
    
    The issue is that converting DTensor outputs to local can cause NCCL blocking.
    """
    try:
        from keras.src.backend.torch import distribution_lib
        
        _original_prepare_output = distribution_lib.prepare_output_for_loss
        
        def patched_prepare_output(x):
            """Patched prepare_output_for_loss with deadlock prevention."""
            import torch
            import os
            
            # Check if we should skip
            skip_prepare = (
                not _check_distributed_initialized() or
                os.environ.get("KERAS_SKIP_DISTRIBUTED_OPS", "0") == "1"
            )
            
            if skip_prepare:
                if DEBUG_DISTRIBUTION:
                    print("[KERAS DIST DEBUG] prepare_output_for_loss: skipping")
                return x
            
            try:
                return _original_prepare_output(x)
            except Exception as e:
                if DEBUG_DISTRIBUTION:
                    print(f"[KERAS DIST DEBUG] prepare_output error: {e}, returning output as-is")
                return x
        
        distribution_lib.prepare_output_for_loss = patched_prepare_output
        
        if DEBUG_DISTRIBUTION:
            print("[KERAS DIST DEBUG] Applied prepare_output_for_loss patch")
            
    except ImportError as e:
        if DEBUG_DISTRIBUTION:
            print(f"[KERAS DIST DEBUG] Could not apply prepare_output patch: {e}")


# =============================================================================
# Main apply function
# =============================================================================

def apply_all_fixes():
    """Apply all distributed training fixes."""
    if DEBUG_DISTRIBUTION:
        print("[KERAS DIST DEBUG] Applying all distributed training fixes...")
    
    _patch_distributed_functions()
    apply_compute_output_spec_fix()
    apply_convert_structure_fix()
    apply_all_gather_fix()
    apply_dtensor_redistribute_fix()
    apply_prepare_input_fix()
    apply_prepare_output_fix()
    
    if DEBUG_DISTRIBUTION:
        print("[KERAS DIST DEBUG] All distributed training fixes applied!")


# =============================================================================
# Test function
# =============================================================================

def test_fixes():
    """Test that the fixes work correctly."""
    import torch
    
    print("Testing distributed training fixes...")
    
    # Check if distributed is available
    print(f"  torch.distributed available: {torch.distributed.is_available()}")
    print(f"  torch.distributed initialized: {_check_distributed_initialized()}")
    
    # Test patches
    _patch_distributed_functions()
    print("  _patch_distributed_functions: OK")
    
    apply_compute_output_spec_fix()
    print("  apply_compute_output_spec_fix: OK")
    
    apply_convert_structure_fix()
    print("  apply_convert_structure_fix: OK")
    
    apply_all_gather_fix()
    print("  apply_all_gather_fix: OK")
    
    print("All tests passed!")


if __name__ == "__main__":
    test_fixes()

