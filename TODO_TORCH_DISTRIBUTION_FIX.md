# TODO: Fix torch distribution_lib.py for tensor parallelism support

## Plan
1. Update `all_gather` function to accept `axis` and `axis_name` parameters (JAX-compatible)
2. Update `all_reduce` function to accept `axis_name` parameter (JAX-compatible)

## Files to modify
- keras/src/backend/torch/distribution_lib.py

## Changes
1. `all_gather(tensor)` → `all_gather(tensor, axis=0, axis_name=None)`
2. `all_reduce(tensor, reduce_op="sum")` → `all_reduce(tensor, reduce_op="sum", axis_name=None)`

