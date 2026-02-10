#!/usr/bin/env python3
"""
Detailed Forward Pass Tracing with Custom Train Step

This script adds extensive logging to trace exactly where the forward pass
hangs during distributed training with the OPT model.
"""

import os
# Must be set before any other imports
os.environ["KERAS_BACKEND"] = "torch"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Set NCCL environment variables to avoid timeouts
os.environ["NCCL_BLOCKING_WAIT"] = "0"
os.environ["NCCL_DEBUG"] = "WARN"

import torch
import numpy as np
import signal
import sys
import time

# Global flag for graceful shutdown
shutdown_requested = False


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    global shutdown_requested
    rank = os.environ.get("LOCAL_RANK", "?")
    print(f"\n[Rank {rank}] Received signal {signum}, shutting down gracefully...")
    shutdown_requested = True


# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def _sync_cuda():
    """Synchronize CUDA streams to ensure all GPU operations complete."""
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass


def _safe_barrier():
    """Perform a barrier with error handling."""
    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
            _sync_cuda()
    except Exception as e:
        rank = os.environ.get("LOCAL_RANK", "?")
        print(f"[Rank {rank}] Barrier warning: {e}")
        _sync_cuda()


class TracedOPTModel:
    """Wrapper around OPT model with detailed logging for forward pass."""
    
    def __init__(self, model, strategy):
        self.model = model
        self.strategy = strategy
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        
    def traced_forward(self, inputs):
        """Forward pass with detailed logging at each step."""
        rank = self.local_rank
        
        print(f"\n[{rank}] === TRACED FORWARD PASS START ===")
        
        # Step 1: Prepare inputs
        print(f"[{rank}] Step 1: Preparing inputs...")
        from keras.src.backend.torch import distribution_lib
        _sync_cuda()
        
        inputs_prepared = distribution_lib.prepare_input_for_distribution(inputs)
        print(f"[{rank}] Step 1: Inputs prepared. Type: {type(inputs_prepared)}")
        _sync_cuda()
        
        # Step 2: Check if model is DTensor
        print(f"[{rank}] Step 2: Checking model weights...")
        first_var = self.model.trainable_variables[0]
        torch_tensor = getattr(first_var, 'value', first_var)
        print(f"[{rank}] First variable type: {type(torch_tensor)}")
        _sync_cuda()
        
        # Step 3: Model call - THIS IS WHERE IT HANGS
        print(f"[{rank}] Step 3: Calling model...")
        _sync_cuda()
        start_time = time.time()
        
        try:
            # Use torch.no_grad() to avoid gradient tracking during trace
            with torch.no_grad():
                outputs = self.model(inputs_prepared, training=False)
            
            elapsed = time.time() - start_time
            print(f"[{rank}] Step 3: Model call completed in {elapsed:.3f}s")
            
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"[{rank}] Step 3: Model call FAILED after {elapsed:.3f}s: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        _sync_cuda()
        
        # Step 4: Process outputs
        print(f"[{rank}] Step 4: Processing outputs...")
        print(f"[{rank}] Output type: {type(outputs)}")
        _sync_cuda()
        
        # Step 5: Prepare output for loss
        print(f"[{rank}] Step 5: Preparing output for loss...")
        outputs_for_loss = distribution_lib.prepare_output_for_loss(outputs)
        print(f"[{rank}] Output for loss type: {type(outputs_for_loss)}")
        _sync_cuda()
        
        print(f"[{rank}] === TRACED FORWARD PASS END ===\n")
        
        return outputs_for_loss
    
    def traced_train_step(self, x, y):
        """Custom train step with detailed logging."""
        rank = self.local_rank
        
        print(f"\n[{rank}] === TRACED TRAIN STEP START ===")
        
        # Step 1: Forward pass
        print(f"[{rank}] Step 1: Forward pass...")
        _sync_cuda()
        start_time = time.time()
        
        try:
            with torch.set_grad_enabled(True):
                y_pred = self.traced_forward(x)
            
            elapsed = time.time() - start_time
            print(f"[{rank}] Step 1: Forward pass completed in {elapsed:.3f}s")
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"[{rank}] Step 1: Forward pass FAILED after {elapsed:.3f}s: {e}")
            raise
        
        _sync_cuda()
        
        # Step 2: Compute loss
        print(f"[{rank}] Step 2: Computing loss...")
        _sync_cuda()
        start_time = time.time()
        
        try:
            loss = self._compute_loss(x, y, y_pred)
            elapsed = time.time() - start_time
            print(f"[{rank}] Step 2: Loss computed in {elapsed:.3f}s: {loss.item():.6f}")
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"[{rank}] Step 2: Loss computation FAILED after {elapsed:.3f}s: {e}")
            raise
        
        _sync_cuda()
        
        # Step 3: Backward pass
        print(f"[{rank}] Step 3: Backward pass...")
        _sync_cuda()
        start_time = time.time()
        
        try:
            loss.backward()
            elapsed = time.time() - start_time
            print(f"[{rank}] Step 3: Backward pass completed in {elapsed:.3f}s")
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"[{rank}] Step 3: Backward pass FAILED after {elapsed:.3f}s: {e}")
            raise
        
        _sync_cuda()
        
        # Step 4: Optimizer step
        print(f"[{rank}] Step 4: Optimizer step...")
        _sync_cuda()
        start_time = time.time()
        
        try:
            self.model.optimizer.apply(
                [v.value.grad for v in self.model.trainable_variables],
                self.model.trainable_variables
            )
            elapsed = time.time() - start_time
            print(f"[{rank}] Step 4: Optimizer step completed in {elapsed:.3f}s")
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"[{rank}] Step 4: Optimizer step FAILED after {elapsed:.3f}s: {e}")
            raise
        
        _sync_cuda()
        
        print(f"[{rank}] === TRACED TRAIN STEP END ===\n")
        
        return {"loss": loss.item()}
    
    def _compute_loss(self, x, y, y_pred):
        """Compute loss with proper handling of DTensor inputs."""
        from keras.src.backend.torch import distribution_lib
        
        # Convert y to local if it's a DTensor
        y_local = distribution_lib.dtensor_to_local(y)
        y_pred_local = distribution_lib.dtensor_to_local(y_pred)
        
        # Compute loss
        loss_fn = torch.nn.CrossEntropyLoss()
        return loss_fn(y_pred_local.view(-1, y_pred_local.size(-1)), y_local.view(-1))


def run_traced_forward_test():
    """Run a traced forward pass test to identify hang location."""
    
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    print(f"\n{'='*70}")
    print(f"TEST: TRACED FORWARD PASS")
    print(f"{'='*70}")
    print(f"Local rank: {local_rank}, World size: {world_size}")
    
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        gpu_id = local_rank % num_gpus
        torch.cuda.set_device(gpu_id)
        print(f"Process {local_rank} -> GPU {gpu_id}")
    
    # Initialize distributed backend
    from keras.src.distribution import DeviceMesh, LayoutMap, ModelParallel, initialize
    from keras.src.backend.torch import distribution_lib
    from torch.distributed._tensor import DTensor, Replicate
    
    print(f"\n[{local_rank}] Initializing distributed backend...")
    initialize()
    
    if torch.distributed.is_initialized():
        print(f"[{local_rank}] ✓ Distributed backend initialized")
    else:
        print(f"[{local_rank}] ✗ Distributed backend NOT initialized")
        return False
    
    # Create DeviceMesh
    devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    mesh = DeviceMesh(
        shape=(len(devices),),
        axis_names=["model"],
        devices=devices
    )
    
    print(f"[{local_rank}] DeviceMesh: shape={mesh.shape}")
    
    # Create minimal LayoutMap - only shard one layer
    layout_map = LayoutMap(mesh)
    layout_map[".*output.*kernel"] = (None, "model")
    layout_map[".*output.*bias"] = ()
    
    print(f"[{local_rank}] LayoutMap: Minimal sharding")
    
    # Create strategy
    strategy = ModelParallel(
        layout_map=layout_map,
        batch_dim_name="data",
        auto_shard_dataset=False
    )
    
    # Build simple model for tracing
    import keras
    from keras import layers
    
    with strategy.scope():
        print(f"\n[{local_rank}] Building simple model...")
        
        model = keras.Sequential([
            layers.Input(shape=(64,)),
            layers.Dense(128, activation="relu", name="dense_1"),
            layers.Dense(64, activation="relu", name="dense_2"),
            layers.Dense(10, name="output")
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss='mse',
            metrics=['mae']
        )
        
        print(f"[{local_rank}] ✓ Model built")
    
    # Create traced wrapper
    traced_model = TracedOPTModel(model, strategy)
    
    # Create test input
    print(f"\n[{local_rank}] Creating test input...")
    batch_size = 1
    x = np.random.random((batch_size, 64)).astype("float32")
    x_tensor = torch.from_numpy(x).cuda()
    
    # Sync all ranks before starting
    _safe_barrier()
    
    # Run traced forward pass
    print(f"\n[{local_rank}] Running traced forward pass...")
    try:
        output = traced_model.traced_forward(x_tensor)
        print(f"[{local_rank}] ✓ Traced forward pass completed!")
        print(f"[{local_rank}] Output shape: {tuple(output.shape) if hasattr(output, 'shape') else 'N/A'}")
    except Exception as e:
        print(f"[{local_rank}] ✗ Traced forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Sync after forward pass
    _safe_barrier()
    
    print(f"\n{'='*70}")
    print("TEST COMPLETE - SUCCESS")
    print(f"{'='*70}")
    
    return True


if __name__ == "__main__":
    success = False
    try:
        success = run_traced_forward_test()
    except KeyboardInterrupt:
        rank = os.environ.get("LOCAL_RANK", "0")
        print(f"\n[Rank {rank}] Interrupted by user")
    except Exception as e:
        rank = os.environ.get("LOCAL_RANK", "0")
        print(f"\n[Rank {rank}] Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        try:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                dist.destroy_process_group()
                print(f"[Cleanup] Process group destroyed")
        except:
            pass
    
    sys.exit(0 if success else 1)

