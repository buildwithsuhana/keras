# TODO: Fix DTensor.unbind global patch

## Steps:
✅ Step 1: Remove monkey-patch from keras/src/backend/torch/distribution_lib.py (done)
- [ ] Step 2: Test distribution_lib_test.py (run pytest)
- [ ] Step 3: Verify distributed training (embedding iteration)
- [ ] Step 3: Verify with pytest keras/src/backend/torch/distribution_lib_test.py
- [ ] Step 4: Run distributed model parallel tests (e.g., embedding layers)
- [ ] Step 5: Mark complete and remove TODO.md

