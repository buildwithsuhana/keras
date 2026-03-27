# TODO: Fix NCCL duplicate GPU error in mp.py

## Plan Steps:
1. [x] User approved the edit plan
2. [x] Create TODO.md with steps (done)
3. [x] Edit mp.py with fixes:
   - Remove CUDA_VISIBLE_DEVICES masking
   - Fix LOCAL_RANK to str(rank)
   - Remove redundant torch.cuda.set_device(0)
4. [ ] Test by running `python mp.py`
5. [ ] Verify "✅ ModelParallel test PASSED!"
6. [ ] Update TODO.md as complete
7. [ ] attempt_completion

