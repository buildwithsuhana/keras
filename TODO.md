# Fix DTensor unbind.int error in mp.py ModelParallel test

## Steps:
- [x] Step 1: Update mp.py to load OPTCausalLM preset outside distribution.scope() to avoid DTensor promotion during build.
- [x] Step 2: Inside scope, recompile model to apply layouts to Variables.
- [ ] Step 3: Adjust layout_map patterns if needed after inspecting model weights.
- [ ] Step 4: Test with `python mp.py`.
- [ ] Step 5: Verify no regressions in keras_opt_test.py.
- [ ] Step 6: attempt_completion.

Current progress: Steps 1-2 done. Step 4: Test `python mp.py` in progress.

