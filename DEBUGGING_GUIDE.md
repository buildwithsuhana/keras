# Debugging Guide for PyTorch DTensor Unbind Issue

## Overview

We've added comprehensive logging to help diagnose and verify that our DTensor unbind fixes are working correctly. When errors occur or during normal operation, the debug logs will show:

1. **DTensor patches being applied** on module import
2. **Shape conversions** when DTensor shapes are processed
3. **Unbind operations** when they're triggered
4. **Build operations** for layers

## Key Log Messages

### 1. DTensor Patches Applied (on keras.distribution import)

```
[DEBUG keras.distribution] DTensor patching: Starting monkey-patch setup
[DEBUG keras.distribution] DTensor patching: Successfully patched DTensor.unbind method
[DEBUG keras.distribution] DTensor patching: Successfully patched DTensor.__iter__ method
```

**What it means:** The monkey-patches to intercept unsafe unbind operations have been successfully installed.

**If you DON'T see these:** The patches failed to apply, likely because PyTorch DTensor isn't available or there's a compatibility issue.

### 2. Safe Shape Conversion (in backend shape() function)

```
[DEBUG keras.torch.backend] shape() detected DTensor with shape torch.Size([2, 3, 4]), converting to local
```

**What it means:** The backend's shape() function detected a distributed tensor and safely converted it to a local shape before returning it as a plain Python tuple.

**When you see this:** Whenever code requests the shape of a distributed tensor.

### 3. Unbind Operations (in distribution library)

```
[DEBUG keras.distribution] unbind_dtensor: Converting sharded tensor shape torch.Size([2, 3, 4]) to local for unbind on dim 0
[DEBUG keras.distribution] unbind_dtensor: Successfully unbound 2 tensors locally, redistributing as replicated
```

**What it means:** 
- First line: A distributed tensor is being unbind, so we convert it to local first
- Second line: The unbind succeeded, and the resulting tensors are being redistributed as replicated

**If you see this:** It means our safe unbind path was triggered. This is GOOD - it means code tried to iterate over or unpack a distributed tensor, and our patches caught and handled it safely.

### 4. Patched DTensor Methods Called

```
[DEBUG keras.distribution] DTensor.unbind called with dim=0, shape=torch.Size([2, 3, 4])
[DEBUG keras.distribution] DTensor.__iter__ called, shape=torch.Size([2, 3, 4]), converting to replicated unbind
```

**What it means:** 
- First: Code called unbind() directly on a distributed tensor (our patch redirected it to the safe version)
- Second: Code tried to iterate over a distributed tensor (e.g., `tuple(dtensor)` or `for x in dtensor`), and our patch handled it

**This is expected** during shape operations and build methods.

### 5. Layer Build Shape Conversion

```
[DEBUG keras.layers] Layer.build_wrapper: Converting DTensor shape torch.Size([None, 32]) to local for TokenAndPositionEmbedding
[DEBUG keras.layers] Layer.build_wrapper: TokenAndPositionEmbedding.build() will receive safe shape (None, 32) (type: tuple)
```

**What it means:** The layer's build_wrapper intercepted the shape argument and converted any distributed tensor to a safe plain Python tuple before passing it to the layer's build method.

**If you see this:** The build_wrapper preprocessing is working correctly.

## Running with Debug Output

### Option 1: Run the test script

```bash
cd /Users/suhanaaa/keras
KERAS_BACKEND=torch KERAS_TORCH_DEVICE=cpu python test_distributed_with_debug.py
```

This will run a suite of tests with debug logging enabled.

### Option 2: Enable logging in your own script

Add this at the TOP of your Python script, BEFORE importing keras:

```python
import logging
import sys

logging.basicConfig(
    level=logging.DEBUG,
    format='[%(levelname)-8s %(name)s] %(message)s',
)

# Enable specific loggers
logging.getLogger('keras.distribution').setLevel(logging.DEBUG)
logging.getLogger('keras.torch.backend').setLevel(logging.DEBUG)
logging.getLogger('keras.layers').setLevel(logging.DEBUG)

# Now import keras
os.environ['KERAS_BACKEND'] = 'torch'
import keras
```

Then run your script with:

```bash
KERAS_BACKEND=torch python your_script.py 2>&1 | grep -E "(DEBUG|ERROR|WARNING)"
```

The `grep` filter shows only important messages.

### Option 3: For distributed multi-GPU training

```bash
KERAS_BACKEND=torch torchrun --nproc_per_node=2 your_script.py 2>&1 | tee debug.log
```

The output will be saved to `debug.log` for analysis.

## Interpreting Error Messages

### If you see: "NotImplementedError: Operator aten.unbind.int does not have a sharding strategy registered"

**Means:** Our patches didn't catch this operation in time. 

**Check logs for:**
- Did you see "DTensor patching: Successfully patched..."? If not, patches failed.
- Did you see "DTensor.unbind called"? If yes, but still get error, there might be a code path we missed.

**Next steps:**
1. Provide the error traceback
2. Share the DEBUG logs from that run
3. We'll identify which code path triggered the unbind

### If you see: "Layer.build_wrapper: Converting DTensor shape..."

**Means:** The shape preprocessing is working correctly. Continue monitoring for actual build execution.

### If you DON'T see expected DEBUG logs

**Check:**
1. Is logging level set to DEBUG? (not INFO or WARNING)
2. Are the logger names spelled correctly?
3. Is KERAS_BACKEND=torch set in environment?

## Troubleshooting Checklist

- [ ] Set `KERAS_BACKEND=torch` environment variable
- [ ] Enable logging BEFORE importing keras
- [ ] Run script with redirected stderr: `python script.py 2>&1 | grep DEBUG`
- [ ] For distributed training, check logs from all processes (rank 0, rank 1, etc.)
- [ ] Look for "DTensor patching: Successfully patched" confirming patches are applied
- [ ] Watch for actual unbind failures - they'll have full tracebacks

## Expected Debug Flow for a Simple Model Build

```
1. [Distribution import]
   DTensor patching: Starting monkey-patch setup
   DTensor patching: Successfully patched DTensor.unbind method
   DTensor patching: Successfully patched DTensor.__iter__ method

2. [Layer instantiation/build]
   Layer.build_wrapper: {LayerName}.build() will receive safe shape (...)
   
3. [Shape operations]
   shape() detected DTensor with shape torch.Size([...])
   
4. [If unbind is triggered]
   DTensor.unbind called with dim=0
   unbind_dtensor: Converting sharded tensor shape ... to local

5. [Build completes]
   (No errors, model is ready)
```

If you see logs progressing through these steps without errors, the fixes are working!
