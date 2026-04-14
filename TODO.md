# Refactoring keras/src/backend/torch/distribution_lib_test.py

## Steps:
- [ ] Step 1: Add pytest fixtures and helper functions at top of file.
- [ ] Step 2: Refactor Base test class and run_distributed to use fixtures; remove try-except via finalizers.
- [ ] Step 3: Merge/Parametrize DistributionBasicsTest (list_devices, get_device_count, num_processes, process_id).
- [ ] Step 4: Parametrize MeshLayoutTest (to_backend_mesh, to_backend_layout).
- [ ] Step 5: Refactor TensorDistributionTest using new helpers/fixtures.
- [ ] Step 6: Merge DataAdapterTest (dataloader, py_dataset, shuffle); use fixtures.
- [ ] Step 7: Parametrize CheckpointTest (checkpoint, full_model, dp) with strategy params and temp fixtures.
- [ ] Step 8: Parametrize TrainerE2ETest (data_parallel_fit, model_parallel_fit).
- [ ] Step 9: Refactor remaining tests (metrics, unbind, utils, variables) minimally.
- [ ] Step 10: Run pytest verification + update TODO with completion.
- [ ] Complete: attempt_completion

**Current: Starting Step 1**
