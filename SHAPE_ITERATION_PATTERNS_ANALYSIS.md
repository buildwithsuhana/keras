# Keras Shape Iteration Patterns - Comprehensive Analysis

## Summary
This document details all places in the Keras codebase where shape parameters are iterated over during layer building, with specific focus on patterns that might trigger `__iter__` calls on distributed tensors.

---

## 1. CRITICAL PATTERNS: Unpacking Operations (`__iter__` Triggers)

### ✅ SAFE - Using tuple() with tuple concatenation:
**File:** [keras/src/layers/core/embedding.py](keras/src/layers/core/embedding.py#L178-L180)
```python
Line 178: # Avoid unpacking (*) which triggers __iter__ on distributed tensors.
Line 179: # Use tuple concatenation instead, which works safely with DTensors.
Line 180: return tuple(input_shape) + (self.output_dim,)
```
**Pattern:** Safe - Uses `tuple(input_shape) + (...)` instead of unpacking with `*`

**File:** [keras/src/layers/core/reversible_embedding.py](keras/src/layers/core/reversible_embedding.py#L151-L156)
```python
Line 151: # Avoid list() which triggers __iter__ on distributed tensors.
Line 152: # Use tuple operations instead for safe handling of both regular and distributed shapes.
Line 154: if reverse:
Line 155:     output_shape = input_shape[:-1] + (self.input_dim,)
Line 156: else:
Line 157:     output_shape = tuple(input_shape) + (self.output_dim,)
```
**Pattern:** Safe - Uses tuple slicing `[:-1]` and tuple concatenation, not unpacking

---

## 2. SHAPE ITERATION IN BUILD METHODS

### Pattern 1: Iterating over axis dimensions with comprehension
**File:** [keras/src/layers/normalization/rms_normalization.py](keras/src/layers/normalization/rms_normalization.py#L53)
```python
Line 43: def build(self, input_shape):
Line 50:     if isinstance(self.axis, (list, tuple)):
Line 51:         self.axis = sorted(self.axis)
Line 53:         shape = tuple(input_shape[dim] for dim in self.axis)
Line 54:     else:
Line 55:         shape = (input_shape[self.axis],)
Line 56:         self.axis = [self.axis]
```
**Pattern:** ⚠️ ITERATION - Iterates over `self.axis` (list of indices) and indexes into `input_shape`
**Risk:** Medium - Indexes into shape, not iterating over shape directly

**File:** [keras/src/layers/normalization/layer_normalization.py](keras/src/layers/normalization/layer_normalization.py#L156)
```python
Line 156: if isinstance(self.axis, (list, tuple)):
Line 157:     self.axis = sorted(self.axis)
Line 158:     shape = tuple(input_shape[dim] for dim in self.axis)
Line 159: else:
Line 160:     shape = (input_shape[self.axis],)
Line 161:     self.axis = [self.axis]
```
**Pattern:** ⚠️ ITERATION - Same pattern as RMSNormalization

**File:** [keras/src/layers/preprocessing/normalization.py](keras/src/layers/preprocessing/normalization.py#L192)
```python
Line 192: mean_and_var_shape = tuple(input_shape[d] for d in self._keep_axis)
```
**Pattern:** ⚠️ ITERATION - Iterates over `self._keep_axis` indices

---

### Pattern 2: Converting shape to list (unpacking trigger)
**File:** [keras/src/layers/core/dense.py](keras/src/layers/core/dense.py#L225)
```python
Line 224: def compute_output_shape(self, input_shape):
Line 225:     output_shape = list(input_shape)
Line 226:     output_shape[-1] = self.units
Line 227:     return tuple(output_shape)
```
**Pattern:** ⚠️ UNPACKING - Creates list from `input_shape` which triggers `__iter__`
**Risk:** HIGH - Explicit iteration via `list()`

**File:** [keras/src/layers/merging/dot.py](keras/src/layers/merging/dot.py#L331-L332)
```python
Line 331: shape1 = list(input_shape[0])
Line 332: shape2 = list(input_shape[1])
```
**Pattern:** ⚠️ UNPACKING - Creates lists from each shape

**File:** [keras/src/layers/activations/prelu.py](keras/src/layers/activations/prelu.py#L55)
```python
Line 55: param_shape = list(input_shape[1:])
```
**Pattern:** ⚠️ UNPACKING - Creates list from shape slice

**File:** [keras/src/layers/merging/concatenate.py](keras/src/layers/merging/concatenate.py#L54)
```python
Line 54: reduced_inputs_shapes = [list(shape) for shape in input_shape]
```
**Pattern:** ⚠️ ITERATION + UNPACKING - Nested iteration over `input_shape` (list of shapes) and converts each to list

**File:** [keras/src/layers/merging/base_merge.py](keras/src/layers/merging/base_merge.py#L115)
```python
Line 115: batch_sizes = {s[0] for s in input_shape if s} - {None}
```
**Pattern:** ⚠️ ITERATION - Set comprehension iterates over `input_shape` (list of shapes)

---

### Pattern 3: Safe tuple operations (no iteration)
**File:** [keras/src/layers/preprocessing/category_encoding.py](keras/src/layers/preprocessing/category_encoding.py#L134-L139)
```python
Line 134:     return tuple(input_shape) + (self.num_tokens,)
Line 136:     return tuple(input_shape) + (self.num_tokens,)
Line 138:     return tuple(input_shape[:-1]) + (self.num_tokens,)
Line 139: return tuple(input_shape[:-1]) + (self.num_tokens,)
```
**Pattern:** ✅ SAFE - Uses tuple slicing and concatenation

**File:** [keras/src/layers/preprocessing/image_preprocessing/random_crop.py](keras/src/layers/preprocessing/image_preprocessing/random_crop.py#L268-L271)
```python
Line 268: input_shape = list(input_shape)
Line 271: return tuple(input_shape)
```
**Pattern:** ⚠️ UNPACKING - Uses `list(input_shape)`

**File:** [keras/src/layers/preprocessing/image_preprocessing/resizing.py](keras/src/layers/preprocessing/image_preprocessing/resizing.py#L295-L310)
```python
Line 295: input_shape = list(input_shape)
Line 310: return tuple(input_shape)
```
**Pattern:** ⚠️ UNPACKING - Uses `list(input_shape)`

**File:** [keras/src/layers/preprocessing/image_preprocessing/center_crop.py](keras/src/layers/preprocessing/image_preprocessing/center_crop.py#L240-L262)
```python
Line 240: input_shape = list(input_shape)
Line 262: return tuple(input_shape)
```
**Pattern:** ⚠️ UNPACKING - Uses `list(input_shape)`

**File:** [keras/src/layers/preprocessing/discretization.py](keras/src/layers/preprocessing/discretization.py#L241-L248)
```python
Line 241:     return tuple(input_shape[:-1]) + (depth,)
Line 243:     return tuple(input_shape) + (depth,)
Line 248:     (input_shape[0],) + tuple(input_shape[2:]) + (depth,)
```
**Pattern:** ✅ SAFE - Uses tuple slicing and concatenation

---

## 3. ITERATION OVER MULTIPLE INPUT SHAPES

**File:** [keras/src/layers/merging/concatenate.py](keras/src/layers/merging/concatenate.py#L51-L95)
```python
Line 51:  if all(shape is None for shape in input_shape):
Line 52:      return
Line 54:  reduced_inputs_shapes = [list(shape) for shape in input_shape]
Line 85:  ranks = set(len(shape) for shape in shape_set)
Line 95:  for shape in shape_set
```
**Pattern:** ⚠️ ITERATION - Multiple patterns:
- **Line 51:** Iterates over `input_shape` to check if all are None
- **Line 54:** List comprehension iterating over `input_shape`, converting each to list
- **Line 85:** Set comprehension iterating over shapes counting ranks
- **Line 95:** Comprehension iterating over shape_set

**File:** [keras/src/layers/merging/concatenate.py](keras/src/layers/merging/concatenate.py#L113-L115)
```python
Line 113: output_shape = list(input_shapes[0])
Line 115: for shape in input_shapes[1:]:
```
**Pattern:** ⚠️ UNPACKING + ITERATION
- **Line 113:** Converts first shape to list
- **Line 115:** Direct `for` loop over remaining input shapes

**File:** [keras/src/layers/merging/base_merge.py](keras/src/layers/merging/base_merge.py#L241)
```python
Line 241: batch_sizes = {s[0] for s in input_shape if s is not None} - {None}
```
**Pattern:** ⚠️ ITERATION - Set comprehension over input_shape

---

## 4. SHAPE OPERATIONS WITH LIST COMPREHENSION

**File:** [keras/src/layers/reshaping/zero_padding1d.py](keras/src/layers/reshaping/zero_padding1d.py#L77)
```python
Line 77: output_shape = list(input_shape)
```
**Pattern:** ⚠️ UNPACKING

**File:** [keras/src/layers/reshaping/zero_padding2d.py](keras/src/layers/reshaping/zero_padding2d.py#L100)
```python
Line 100: output_shape = list(input_shape)
```
**Pattern:** ⚠️ UNPACKING

**File:** [keras/src/layers/reshaping/zero_padding3d.py](keras/src/layers/reshaping/zero_padding3d.py#L99)
```python
Line 99: output_shape = list(input_shape)
```
**Pattern:** ⚠️ UNPACKING

**File:** [keras/src/layers/reshaping/cropping3d.py](keras/src/layers/reshaping/cropping3d.py#L105-L107)
```python
Line 105: spatial_dims = list(input_shape[2:5])
Line 107: spatial_dims = list(input_shape[1:4])
```
**Pattern:** ⚠️ UNPACKING - Creates list from shape slices

**File:** [keras/src/layers/preprocessing/image_preprocessing/max_num_bounding_box.py](keras/src/layers/preprocessing/image_preprocessing/max_num_bounding_box.py#L104-L106)
```python
Line 104: boxes_shape = list(input_shape["bounding_boxes"]["boxes"])
Line 106: labels_shape = list(input_shape["bounding_boxes"]["labels"])
```
**Pattern:** ⚠️ UNPACKING - Creates lists from nested shape dicts

---

## 5. SPECIAL CASES: hasattr(__iter__)

**File:** [keras/src/layers/preprocessing/discretization.py](keras/src/layers/preprocessing/discretization.py#L196)
```python
Line 196: elif hasattr(data, "__iter__") and not (...):
```
**Pattern:** ⚠️ CHECK - Directly checks for `__iter__` attribute on data

**File:** [keras/src/layers/preprocessing/index_lookup.py](keras/src/layers/preprocessing/index_lookup.py#L652)
```python
Line 652: elif hasattr(data, "__iter__") and not (...):
```
**Pattern:** ⚠️ CHECK - Directly checks for `__iter__` attribute on data

**File:** [keras/src/layers/preprocessing/normalization.py](keras/src/layers/preprocessing/normalization.py#L293)
```python
Line 293: elif hasattr(data, "__iter__"):
```
**Pattern:** ⚠️ CHECK - Directly checks for `__iter__` attribute on data

**File:** [keras/src/layers/preprocessing/text_vectorization.py](keras/src/layers/preprocessing/text_vectorization.py#L423)
```python
Line 423: elif hasattr(data, "__iter__") and not (...):
```
**Pattern:** ⚠️ CHECK - Directly checks for `__iter__` attribute on data

---

## 6. ATTENTION LAYER PATTERNS

**File:** [keras/src/layers/activations/softmax.py](keras/src/layers/activations/softmax.py#L64)
```python
Line 64: for m_dim, i_dim in zip(mask.shape[::-1], inputs.shape[::-1]):
```
**Pattern:** ⚠️ ZIP ITERATION - Zips reversed shapes together

**File:** [keras/src/layers/attention/multi_head_attention.py](keras/src/layers/attention/multi_head_attention.py#L247)
```python
Line 247: if value_shape[1:-1] != key_shape[1:-1]:
```
**Pattern:** ✅ SAFE - Tuple slicing, no iteration

**File:** [keras/src/layers/attention/multi_head_attention.py](keras/src/layers/attention/multi_head_attention.py#L761)
```python
Line 761: attention_shape = (query.shape[0], self.num_heads, length, length)
```
**Pattern:** ✅ SAFE - Indexing specific shape elements

---

## 7. RNN LAYER PATTERNS (Shape indexing)

**File:** [keras/src/layers/rnn/rnn.py](keras/src/layers/rnn/rnn.py#L272)
```python
Line 272: step_input_shape = (sequences_shape[0],) + tuple(sequences_shape[2:])
```
**Pattern:** ⚠️ UNPACKING - Creates tuple from slice of shape

**File:** [keras/src/layers/rnn/conv_lstm.py](keras/src/layers/rnn/conv_lstm.py#L184)
```python
Line 184: ndim=self.rank + 3, shape=(None,) + inputs_shape[1:]
```
**Pattern:** ⚠️ UNPACKING - Slicing shape for tuple concatenation

**File:** [keras/src/layers/rnn/gru.py](keras/src/layers/rnn/gru.py#L193)
```python
Line 193: for e in ops.split(self.bias, self.bias.shape[0], axis=0)
```
**Pattern:** ⚠️ ITERATION - For loop with ops.split() which may trigger iterations

---

## 8. QUANTIZATION UNPACKING (NOT shape iteration, but relevant context)

**File:** [keras/src/layers/core/embedding.py](keras/src/layers/core/embedding.py#L157)
```python
Line 157: unpacked_embeddings = quantizers.unpack_int4(...)
```
**Note:** This unpacks quantized values, NOT shapes. Safe context.

---

## SUMMARY BY RISK LEVEL

### HIGH RISK (Direct iteration via list()):
1. [keras/src/layers/core/dense.py](keras/src/layers/core/dense.py#L225) - `list(input_shape)`
2. [keras/src/layers/merging/dot.py](keras/src/layers/merging/dot.py#L331) - `list(input_shape[0])`
3. [keras/src/layers/merging/concatenate.py](keras/src/layers/merging/concatenate.py#L54) - `[list(shape) for shape in input_shape]`
4. [keras/src/layers/activations/prelu.py](keras/src/layers/activations/prelu.py#L55) - `list(input_shape[1:])`
5. Multiple padding/crop layers with `list(input_shape)`

### MEDIUM RISK (Comprehensions over axis/dimensions):
1. [keras/src/layers/normalization/rms_normalization.py](keras/src/layers/normalization/rms_normalization.py#L53) - `tuple(input_shape[dim] for dim in self.axis)`
2. [keras/src/layers/normalization/layer_normalization.py](keras/src/layers/normalization/layer_normalization.py#L156) - Same pattern
3. [keras/src/layers/preprocessing/normalization.py](keras/src/layers/preprocessing/normalization.py#L192) - `tuple(input_shape[d] for d in self._keep_axis)`
4. [keras/src/layers/merging/base_merge.py](keras/src/layers/merging/base_merge.py#L115) - Set comprehension over input_shape
5. [keras/src/layers/merging/concatenate.py](keras/src/layers/merging/concatenate.py#L85-L95) - Multiple comprehensions

### SAFE PATTERNS (Tuple slicing/concatenation):
1. [keras/src/layers/core/embedding.py](keras/src/layers/core/embedding.py#L180) - `tuple(input_shape) + (...)`
2. [keras/src/layers/core/reversible_embedding.py](keras/src/layers/core/reversible_embedding.py#L155-L156) - Tuple slicing
3. [keras/src/layers/preprocessing/category_encoding.py](keras/src/layers/preprocessing/category_encoding.py) - All safe patterns
4. [keras/src/layers/preprocessing/discretization.py](keras/src/layers/preprocessing/discretization.py) - Safe tuple operations

---

## RELATED: Comments About Distributed Tensor Handling

The embedding and reversible_embedding layers explicitly mention the issue:
- "Avoid unpacking (*) which triggers __iter__ on distributed tensors." 
- "Avoid list() which triggers __iter__ on distributed tensors."

This suggests that distributed tensor support (likely for DTensor) requires avoiding operations that call `__iter__` on shape objects.
