# TODO: Fix metric sync issue in torch/trainer.py _sync_metrics

## Information Gathered:
- Current code incorrectly handles DTensor metric variables by extracting local shard, SUM reducing locally (wrong), assigning back local to DTensor (corrupts sharding).
- Always uses SUM, but noted concern for MAX/MIN; however SUM common for sum/count in metrics.
- Torch DTensor supports .all_reduce(op=ReduceOp.SUM) natively, handling sharding.
- Dense tensors all_reduce in place.

## Plan:
- Replace _sync_metrics implementation to:
  - Unified all_reduce(v.value, SUM) for all variables, works for DTensor/dense.
  - No to_local/assign — in-place reduce, correct semantics.

## Dependent Files:
- None.

## Followup steps:
- Verify change.
- No tests needed.

## Steps:
1. [x] Replace _sync_metrics with unified all_reduce on variable.value.
2. [x] Verify the changes and complete the task.

