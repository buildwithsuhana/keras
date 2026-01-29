# Fix Plan: Variables Should Be Sharded at Creation Time

## Problem
Variables are NOT sharded when created - they are created as regular `torch.nn.Parameter`, then sharded later by `parallelize_module`. This causes OOM because each device holds the full model before sharding.

**Current Flow (CAUSES OOM)**:
1. `model = create_model()` → Variables created as full Parameters
2. `model.compile()` 
3. `model.fit()` → `parallelize_module` called → shards weights
4. **Problem**: Full model was in memory before parallelization

## Root Cause
In `TorchTrainer.fit()`, `_parallelize_if_needed()` is called AFTER the model is already created and built:

```python
def fit(self, ...):
    # Parallelize here (but model is already created!)
    self._parallelize_if_needed()  
    
    # Create iterator (triggers _symbolic_build which creates weights)
    epoch_iterator = TorchEpochIterator(...)
    self._symbolic_build(iterator=epoch_iterator)  # <-- WEIGHTS CREATED HERE
```

The model is created when user calls `model = create_model()`, which is BEFORE `fit()`.

## Solution
Move parallelization to happen BEFORE `_symbolic_build()` (weight creation) in `fit()`, `evaluate()`, and `predict()`.

## Files to Modify

### `keras/src/backend/torch/trainer.py`

1. **Modify `_symbolic_build()`**: Call `parallelize_keras_model()` at the start, before any weight creation
2. **Remove/redundant `_parallelize_if_needed()` calls**: These are no longer needed since parallelization happens in `_symbolic_build()`

## Implementation Details

### Current `_symbolic_build()`:
```python
def _symbolic_build(self, *args, **kwargs):
    # Try to auto-parallelize
    if _should_auto_parallelize():
        _auto_parallelize_model(torch_module)
    # Then call parent (creates weights)
    return super()._symbolic_build(*args, **kwargs)
```

**Problem**: `_auto_parallelize_model()` is called, but it needs to happen BEFORE parent's `_symbolic_build()`.

### Fix: Reorder `_symbolic_build()`
```python
def _symbolic_build(self, *args, **kwargs):
    # FIRST: Call parent _symbolic_build to create the model structure
    result = super()._symbolic_build(*args, **kwargs)
    
    # THEN: Apply parallelization (now model is built and can be parallelized)
    try:
        if _should_auto_parallelize():
            _auto_parallelize_model(torch_module)
    except Exception:
        pass
    
    return result
```

Wait, this still creates weights before parallelization. We need the OPPOSITE order!

### Correct Fix: Parallelize BEFORE model build
```python
def _symbolic_build(self, *args, **kwargs):
    # FIRST: Parallelize the model BEFORE any weights are created
    try:
        if _should_auto_parallelize():
            _auto_parallelize_model(torch_module)
    except Exception:
        pass
    
    # THEN: Call parent _symbolic_build to create weights
    return super()._symbolic_build(*args, **kwargs)
```

Wait, this won't work either because `_auto_parallelize_model()` is called BEFORE the model is built. We need the model to be built first before parallelizing it.

### The Correct Solution
The issue is timing. We need to:
1. Build the model first (create layers, but NOT call `forward` which creates weights)
2. Apply parallelization
3. Then trigger weight creation

Looking at how Keras builds models:
- `_symbolic_build()` calls `self.build()` which calls `call()`
- `call()` creates the computational graph and triggers weight creation

The solution is to modify `_symbolic_build()` to:
1. Call parent's `_symbolic_build()` with a placeholder to just build the structure
2. Apply parallelization
3. Then trigger actual weight creation with parallelized modules

Actually, a simpler approach: Move parallelization to happen at the START of `fit()`/`evaluate()`/`predict()`, BEFORE `_symbolic_build()` is called. But we need to ensure the model is built first.

### Simpler Solution: Add parallelization before `_symbolic_build()` in each method

In `fit()`:
```python
def fit(self, ...):
    # Parallelize FIRST if model is already built
    if self.built:
        self._parallelize_if_needed()
    
    # Then build (if not already built)
    self._symbolic_build(iterator=epoch_iterator)
```

This handles the case where:
- Model is NOT built yet: `_symbolic_build()` builds it, then parallelization happens inside
- Model IS already built: `_parallelize_if_needed()` parallelizes it before anything else

But the issue is `_parallelize_if_needed()` is called AFTER `_symbolic_build()` in the current code!

### The Actual Fix
Change the ORDER in `fit()`, `evaluate()`, `predict()`:

```python
def fit(self, ...):
    # Parallelize BEFORE any build/weight creation
    self._parallelize_if_needed()
    
    # Then build and train
    self._symbolic_build(iterator=epoch_iterator)
    ...
```

But we need to ensure `_parallelize_if_needed()` can work even if model isn't fully built yet.

Looking at `_parallelize_if_needed()`:
```python
def _parallelize_if_needed(self):
    if self._torch_module_parallelized:
        return
    
    # Get torch_module
    if hasattr(self, '_torch_layers'):
        torch_module = self._torch_layers
    else:
        torch_module = self
    
    # Parallelize
    parallelize_keras_model(torch_module, device_mesh=device_mesh, layout_map=dist._layout_map)
```

This should work even if the model isn't fully built yet, because `parallelize_keras_model()` doesn't need weights to be created - it just sets up the module for parallel execution.

## Implementation

### Change 1: Modify `fit()` - call `_parallelize_if_needed()` BEFORE `_symbolic_build()`
```python
# In fit() method, around line 250:
# BEFORE:
#     self._symbolic_build(iterator=epoch_iterator)

# AFTER:
self._parallelize_if_needed()  # <-- ADD THIS BEFORE
self._symbolic_build(iterator=epoch_iterator)
```

### Change 2: Same for `evaluate()` and `predict()`

### Change 3: Update `_symbolic_build()` to not call `_auto_parallelize_model()` (since it's now handled in each method)

Actually, looking more carefully at `_parallelize_if_needed()`:
```python
def _parallelize_if_needed(self):
    if self._torch_module_parallelized:
        return
    ...
    parallelize_keras_model(...)
    self._torch_module_parallelized = True
```

It already has the check `if self._torch_module_parallelized: return`. So calling it multiple times is safe (idempotent).

The current issue is the ORDER in `fit()`:
```python
# Current order in fit():
epoch_iterator = TorchEpochIterator(...)
self._symbolic_build(iterator=epoch_iterator)  # <-- Creates weights
self._parallelize_if_needed()  # <-- Too late!
```

The fix is to swap the order:
```python
# Fixed order:
self._parallelize_if_needed()  # <-- Parallelize FIRST
epoch_iterator = TorchEpochIterator(...)
self._symbolic_build(iterator=epoch_iterator)  # <-- Creates weights with parallelization
```

But wait, `_parallelize_if_needed()` parallelizes `self._torch_layers` or `self`. If the model isn't built yet, these might not have the layers set up properly.

Let me check what `parallelize_keras_model()` does:
```python
def parallelize_keras_model(model, device_mesh, layout_map):
    # Get torch_module
    if hasattr(model, '_torch_layers'):
        torch_module = model._torch_layers
    ...
    # Create parallel plan
    parallel_plan = create_tp_plan_from_layout_map(torch_module, device_mesh, keras_layout_map)
    
    # Apply tensor parallelism
    parallelized_module = parallelize_module(torch_module, tp_mesh, parallelize_plan=parallel_plan)
```

It needs the torch module to be built to inspect layer types for the parallel plan. If the model isn't built, this might not work correctly.

### The Real Solution
The parallelization should happen AFTER the model is built (layers are created) but BEFORE weights are created.

Looking at the Keras model build flow:
1. `Layer.build()` is called
2. Inside `build()`, `self.call()` is called with a placeholder input
3. This creates the computational graph and triggers weight creation

The key is that weight creation happens inside `call()` during build. We need to parallelize BEFORE `call()` is invoked.

Looking at how Keras handles this for Torch layers:
- The `_torch_layers` attribute contains the actual torch modules
- These are created when the Keras layer is first called
- Weight creation happens when torch parameters are created

The solution is to trigger parallelization at the start of `fit()`/`evaluate()`/`predict()`, but we need to ensure the model is built first.

### The Fix (Final Version)

```python
def fit(self, ...):
    # First, ensure model is built (creates layers, but not weights yet)
    # Actually, Keras layers are created at model construction time, not build time
    # The build just triggers weight creation
    
    # So we can parallelize at the start of fit()
    self._parallelize_if_needed()  # Parallelize the model
    
    # Then build weights (which will now be created as DTensors)
    self._symbolic_build(iterator=epoch_iterator)
```

But `_parallelize_if_needed()` needs the model to be built to create the parallel plan. Let me check if this is actually required...

Looking at `create_tp_plan_from_layout_map()`:
```python
def create_tp_plan_from_layout_map(module, device_mesh, keras_layout_map):
    plan = {}
    
    for pattern, sharding_spec in keras_layout_map.items():
        # Convert pattern from Keras format to PyTorch
        pytorch_pattern = pattern.replace('/', '.')
        
        if isinstance(sharding_spec, tuple):
            # Determine parallel style based on position
            if model_idx == 1:
                plan[pytorch_pattern] = ColwiseParallel()
            elif model_idx == 0:
                plan[pytorch_pattern] = RowwiseParallel()
    
    return plan
```

This doesn't actually need the module to be built - it just creates a plan based on patterns. The plan is then applied by `parallelize_module()` which sets up the module for parallel execution.

So `_parallelize_if_needed()` should work even if the model isn't fully built!

## Summary of Changes

### File: `keras/src/backend/torch/trainer.py`

1. **In `fit()`**: Move `_parallelize_if_needed()` to BEFORE `_symbolic_build()`
2. **In `evaluate()`**: Move `_parallelize_if_needed()` to BEFORE `_symbolic_build()`
3. **In `predict()`**: Move `_parallelize_if_needed()` to BEFORE `_symbolic_build()`
4. **In `_symbolic_build()`**: Remove the `_auto_parallelize_model()` call since parallelization now happens before

## Expected Behavior After Fix

1. User creates model inside `with distribution.scope():`
2. Variables are created as regular Parameters (this is expected - we can't avoid this)
3. User calls `model.fit()`
4. `_parallelize_if_needed()` is called FIRST → `parallelize_module` shards the model
5. `_symbolic_build()` is called (if needed) → creates weights that are properly sharded
6. Training starts with sharded weights

This ensures parallelization happens BEFORE any heavy computation, reducing peak memory usage.

## Testing

Run existing tests to verify:
1. `test_auto_parallelize.py` - Auto parallelization still works
2. `test_model_parallel_fix.py` - Model parallel training doesn't OOM
3. `test_sharding_fix.py` - Variables are properly sharded

## Follow-up

- Ensure `_parallelize_if_needed()` handles edge cases (model not built, etc.)
- Consider adding a warning if parallelization fails
- Monitor memory usage during training

