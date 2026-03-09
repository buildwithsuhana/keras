import builtins
import contextlib
import functools
import os

import ml_dtypes
import numpy as np
import torch

from keras.src import tree
from keras.src.backend.common import KerasVariable
from keras.src.backend.common import global_state
from keras.src.backend.common import standardize_dtype
from keras.src.backend.common.backend_utils import slice_along_axis
from keras.src.backend.common.dtypes import result_type
from keras.src.backend.common.keras_tensor import KerasTensor
from keras.src.backend.common.stateless_scope import StatelessScope
from keras.src.backend.common.stateless_scope import get_stateless_scope
from keras.src.backend.common.stateless_scope import in_stateless_scope
from keras.src.backend.common.symbolic_scope import SymbolicScope
from keras.src.backend.config import floatx

SUPPORTS_SPARSE_TENSORS = False
SUPPORTS_RAGGED_TENSORS = False
IS_THREAD_SAFE = True

# Some operators such as 'aten::_foreach_mul_.Scalar'
# are not currently implemented for the MPS device.
# check https://github.com/pytorch/pytorch/issues/77764.
if "KERAS_TORCH_DEVICE" in os.environ:
    DEFAULT_DEVICE = os.environ["KERAS_TORCH_DEVICE"]
elif torch.backends.mps.is_available():
    DEFAULT_DEVICE = "mps"
elif torch.cuda.is_available():
    DEFAULT_DEVICE = "cuda"
elif hasattr(torch, "xpu") and torch.xpu.is_available():
    DEFAULT_DEVICE = "xpu"
else:
    DEFAULT_DEVICE = "cpu"

TORCH_DTYPES = {
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
    "uint8": torch.uint8,
    "uint16": torch.int32,  # TODO: Torch doesn't have `uint16` dtype.
    "uint32": torch.int64,  # TODO: Torch doesn't have `uint32` dtype.
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
    "bfloat16": torch.bfloat16,
    "bool": torch.bool,
    "float8_e4m3fn": torch.float8_e4m3fn,
    "float8_e5m2": torch.float8_e5m2,
    "complex32": torch.complex32,
    "complex64": torch.complex64,
    "complex128": torch.complex128,
}


@contextlib.contextmanager
def device_scope(device_name):
    """Context manager to specify the device for Torch operations.

    Args:
        device_name: String identifying the device (e.g., "cpu", "gpu:0").
    """
    previous_device = global_state.get_global_attribute("torch_device", None)
    current_device = _parse_device_input(device_name)
    global_state.set_global_attribute("torch_device", current_device)
    try:
        yield torch.device(current_device)
    finally:
        global_state.set_global_attribute("torch_device", previous_device)


def get_device():
    """Return the current default device as a string."""
    device = global_state.get_global_attribute("torch_device", None)
    if device is not None:
        return device
    
    distribution = global_state.get_global_attribute("distribution")
    if distribution is not None:
        return distribution.device_mesh.backend_mesh.device_type

    return DEFAULT_DEVICE


def _parse_device_input(device_name):
    """Normalize device input strings."""
    if isinstance(device_name, str):
        device_name = device_name.lower()
        if "gpu" in device_name:
            device_name = device_name.replace("gpu", "cuda")
        if "tpu" in device_name:
            device_name = device_name.replace("tpu", "xla")
        return device_name
    raise ValueError(
        "Invalid value for argument `device_name`. "
        "Expected a string like 'gpu:0' or 'cpu'. "
        f"Received: device_name='{device_name}'"
    )


def to_torch_dtype(dtype):
    """Convert Keras dtype to Torch dtype."""
    standardized_dtype = TORCH_DTYPES.get(standardize_dtype(dtype), None)
    if standardized_dtype is None:
        raise ValueError(f"Unsupported dtype for PyTorch: {dtype}")
    return standardized_dtype


class Variable(KerasVariable):
    """Torch-specific implementation of Keras Variable."""

    def __init__(self, *args, layout=None, **kwargs):
        self._layout = layout
        super().__init__(*args, **kwargs)

    def _initialize_layout(self):
        """Initialize the layout for distributed training."""
        distribution = global_state.get_global_attribute("distribution")
        if self._layout is None and distribution is not None:
            self._layout = distribution.get_variable_layout(self)

    def _initialize(self, value):
        """Initialize the variable value and sharding."""
        self._shape = self._validate_shape(value.shape)
        self._initialize_layout()
        if isinstance(value, torch.nn.Parameter):
            self._value = value
        else:
            value = convert_to_tensor(value, dtype=self._dtype)
            if self._layout is not None:
                from keras.src.backend.torch import distribution_lib
                value = distribution_lib.distribute_variable(value, self._layout)

            requires_grad = self.trainable and torch.is_floating_point(value)
            self._value = torch.nn.Parameter(value, requires_grad=requires_grad)
            
            # Sharding detection for avoidance of redundant .to()
            is_sharded = (
                getattr(value, "device_mesh", None) is not None or 
                getattr(value, "placements", None) is not None or
                (hasattr(value, "data") and (
                    getattr(value.data, "device_mesh", None) is not None or 
                    getattr(value.data, "placements", None) is not None
                ))
            )
            if not is_sharded and self._value.device.type != torch.device(get_device()).type:
                self._value = self._value.to(get_device())
            
            if self._layout is not None and requires_grad:
                self._value.retain_grad()

    def _direct_assign(self, value):
        """Perform direct assignment to the variable."""
        self._initialize_layout()
        if self._layout is not None:
            from keras.src.backend.torch import distribution_lib
            value = distribution_lib.distribute_variable(value, self._layout)

        with torch.no_grad():
            try:
                self.value.copy_(value)
            except Exception as e:
                # Fallback for mixed DTensor/Tensor copy
                if hasattr(value, "to_local"):
                    self.value.copy_(value.to_local())
                else:
                    raise e

    def _convert_to_tensor(self, value, dtype=None):
        return convert_to_tensor(value, dtype=dtype)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        def _unwrap(x):
            return x.value if isinstance(x, Variable) else x

        unwrapped_args = tree.map_structure(_unwrap, args)
        unwrapped_kwargs = tree.map_structure(_unwrap, kwargs or {})

        try:
            return func(*unwrapped_args, **unwrapped_kwargs)
        except (RuntimeError, Exception) as e:
            msg = str(e).lower()
            if any(k in msg for k in ("sharding", "shard", "redistribute", "partial", "dtensor", "flatten", "view", "reshape", "boolean value", "meta tensor")):
                from keras.src.backend.torch.distribution_lib import _maybe_replicate
                unwrapped_args = tree.map_structure(_maybe_replicate, unwrapped_args)
                unwrapped_kwargs = tree.map_structure(_maybe_replicate, unwrapped_kwargs)
                return func(*unwrapped_args, **unwrapped_kwargs)
            raise e

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        try:
            return func(*args, **(kwargs or {}))
        except (RuntimeError, Exception) as e:
            msg = str(e).lower()
            if any(k in msg for k in ("sharding", "shard", "redistribute", "partial", "dtensor", "flatten", "view", "reshape", "boolean value", "meta tensor")):
                from keras.src.backend.torch.distribution_lib import _maybe_replicate
                args = tree.map_structure(_maybe_replicate, args)
                kwargs = tree.map_structure(_maybe_replicate, kwargs or {})
                return func(*args, **kwargs)
            raise e

    def __array__(self, dtype=None):
        value = convert_to_numpy(self.value)
        return value.astype(dtype) if dtype else value

    @property
    def value(self):
        """Return the current value of the variable, handling symbolic and stateless scopes."""
        def maybe_use_symbolic_tensor(value):
            if str(get_device()) == "meta" and str(value.device) != "meta":
                return torch.nn.Parameter(
                    torch.empty(size=self._shape, dtype=to_torch_dtype(self._dtype), device="meta"),
                    requires_grad=self.trainable
                )
            return value

        if in_stateless_scope():
            value = get_stateless_scope().get_current_value(self)
            if value is not None:
                return maybe_use_symbolic_tensor(self._maybe_autocast(value))
        
        if self._value is None:
            value = self._maybe_autocast(self._initializer(self._shape, dtype=self._dtype))
        else:
            value = self._maybe_autocast(self._value)
        return maybe_use_symbolic_tensor(value)

    @property
    def trainable(self):
        return self._trainable

    @trainable.setter
    def trainable(self, value):
        self._trainable = value
        if self._value is not None:
            self._value.requires_grad = value

    def __eq__(self, other):
        try:
            return super().__eq__(other)
        except Exception:
            return False


def _maybe_distribute(res):
    """Automatically distribute a tensor if a distribution is active."""
    from keras.src.backend.torch.distribution_lib import DTensor
    if not isinstance(res, DTensor):
        from keras.src.distribution import distribution_lib
        d = distribution_lib.distribution()
        if d is not None:
            from keras.src.backend.torch.distribution_lib import distribute_tensor
            from keras.src.distribution import TensorLayout
            layout = TensorLayout([None] * len(res.shape), d.device_mesh)
            return distribute_tensor(res, layout)
    return res


def convert_to_tensor(x, dtype=None, sparse=None, ragged=None):
    """Convert input to a Torch tensor."""
    if sparse or ragged:
        raise ValueError(f"{'Sparse' if sparse else 'Ragged'} tensors are not supported in Torch backend.")

    if isinstance(x, Variable) or is_tensor(x):
        if isinstance(x, Variable):
            x = x.value
        device = get_device()
        if x.device != device:
            x = torch.empty_like(x, device=device) if x.is_meta else x.to(device)
        if dtype is not None:
            x = x.to(to_torch_dtype(dtype))
        return _maybe_distribute(x)
    
    if dtype is None:
        if isinstance(x, bool):
            return _maybe_distribute(torch.as_tensor(x, dtype=torch.bool, device=get_device()))
        if isinstance(x, int):
            return _maybe_distribute(torch.as_tensor(x, dtype=torch.int32, device=get_device()))
        if isinstance(x, float):
            return _maybe_distribute(torch.as_tensor(x, dtype=to_torch_dtype(floatx()), device=get_device()))

    if not isinstance(x, (list, tuple)):
        x = np.array(x)
    elif len(x) > 0 and any(isinstance(x1, torch.Tensor) for x1 in x):
        return _maybe_distribute(torch.stack([convert_to_tensor(x1) for x1 in x]))
    
    if isinstance(x, np.ndarray):
        if x.dtype == np.uint32:
            x = x.astype(np.int64)
        if standardize_dtype(x.dtype) == "bfloat16":
            x, dtype = x.astype(np.float32), "bfloat16"
        dtype = dtype or x.dtype
    
    if dtype is None:
        dtype = result_type(*[getattr(item, "dtype", type(item)) for item in tree.flatten(x)])
    
    return _maybe_distribute(torch.as_tensor(x, dtype=to_torch_dtype(dtype), device=get_device()))


def convert_to_numpy(x):
    """Convert a Torch tensor to a NumPy array."""
    def transform(x):
        if is_tensor(x):
            if hasattr(x, "to_local"):
                x = x.to_local()
            if x.requires_grad:
                x = x.detach()
            if x.device != torch.device("cpu"):
                x = x.cpu()
            if x.dtype == torch.bfloat16:
                return np.array(x.to(torch.float32)).astype(ml_dtypes.bfloat16)
        return np.array(x)

    if isinstance(x, (list, tuple)):
        return np.array([transform(e) for e in x])
    return transform(x)


def is_tensor(x):
    """Return True if x is a Torch tensor."""
    return isinstance(x, torch.Tensor)


def shape(x):
    """Return the shape of a tensor as a tuple."""
    return tuple(x.shape)


def cast(x, dtype):
    """Cast a tensor to a specific dtype."""
    target_dtype = to_torch_dtype(dtype)
    if isinstance(x, Variable):
        x = x.value
    if is_tensor(x):
        return x if x.dtype == target_dtype else x.to(target_dtype)
    return convert_to_tensor(x, dtype)


def compute_output_spec(fn, *args, **kwargs):
    """Compute the output specification (shape and dtype) of a function."""
    def convert_keras_tensor_to_torch(x, fill_value=None):
        if isinstance(x, KerasTensor):
            s = [fill_value if e is None else e for e in x.shape]
            return _maybe_distribute(torch.ones(size=s, dtype=TORCH_DTYPES[x.dtype], device=get_device()))
        return x

    def symbolic_call(fn, args, kwargs, fill_value):
        try:
            with device_scope("meta"):
                meta_args, meta_kwargs = tree.map_structure(lambda x: convert_keras_tensor_to_torch(x, fill_value), (args, kwargs))
                return fn(*meta_args, **meta_kwargs)
        except:
            with device_scope(get_device()):
                eager_args, eager_kwargs = tree.map_structure(lambda x: convert_keras_tensor_to_torch(x, fill_value), (args, kwargs))
                return fn(*eager_args, **eager_kwargs)

    with StatelessScope(), SymbolicScope(), torch.no_grad():
        outputs = symbolic_call(fn, args, kwargs, fill_value=83)
        if any(None in x.shape for x in tree.flatten((args, kwargs)) if isinstance(x, KerasTensor)):
            outputs_2 = symbolic_call(fn, args, kwargs, fill_value=89)
            flat_out_1, flat_out_2 = tree.flatten(outputs), tree.flatten(outputs_2)
            flat_out = []
            for x1, x2 in zip(flat_out_1, flat_out_2):
                s = [None if x1.shape[i] != x2.shape[i] else x1.shape[i] for i in range(len(x1.shape))]
                flat_out.append(KerasTensor(s, standardize_dtype(x1.dtype)))
            outputs = tree.pack_sequence_as(outputs, flat_out)
        
        return tree.map_structure(lambda x: KerasTensor(x.shape, standardize_dtype(x.dtype)) if is_tensor(x) else x, outputs)


def cond(pred, true_fn, false_fn):
    """Conditional execution."""
    if get_device() == "meta":
        return true_fn()
    return true_fn() if pred else false_fn()


def vectorized_map(function, elements):
    """Apply function to elements using vectorized execution."""
    return torch.vmap(function)(elements)


def map(f, xs):
    """Apply function to elements sequentially."""
    return scan(lambda _, x: ((), f(x)), (), xs)[1]


def scan(f, init, xs=None, length=None, reverse=False, unroll=1):
    """Iteratively apply a function and accumulate results."""
    if xs is None:
        num_elements, flat_xs = int(length), []
    else:
        flat_xs = [convert_to_tensor(e) for e in tree.flatten(xs)]
        num_elements = int(length) if length is not None else shape(flat_xs[0])[0]

    init_flat = [convert_to_tensor(i) for i in tree.flatten(init)]
    carry = tree.pack_sequence_as(init, init_flat)
    ys = []
    indices = reversed(range(num_elements)) if reverse else range(num_elements)
    
    for i in indices:
        xs_slice = tree.pack_sequence_as(xs, [x[i] for x in flat_xs]) if flat_xs else None
        carry, y = f(carry, xs_slice)
        ys.append(y if y is not None else [torch.zeros_like(i) for i in init_flat])
    
    stacked_y = tree.map_structure(lambda *ys: torch.stack(ys), *(reversed(ys) if reverse else ys))
    return carry, stacked_y


def associative_scan(f, elems, reverse=False, axis=0):
    """Perform an associative scan operation."""
    flat_elems = [convert_to_tensor(e) for e in tree.flatten(elems)]
    if reverse:
        flat_elems = [torch.flip(elem, (axis,)) for elem in flat_elems]

    def _combine(a_flat, b_flat):
        a_flat = [convert_to_tensor(a) for a in a_flat]
        b_flat = [convert_to_tensor(b) for b in b_flat]

        a = tree.pack_sequence_as(elems, a_flat)
        b = tree.pack_sequence_as(elems, b_flat)
        c = f(a, b)
        c_flat = tree.flatten(c)
        return c_flat

    num_elems = int(flat_elems[0].shape[axis])
    if not all(int(elem.shape[axis]) == num_elems for elem in flat_elems[1:]):
        raise ValueError(
            "Array inputs to associative_scan must have the same "
            "first dimension. (saw: {})".format(
                [elem.shape for elem in flat_elems]
            )
        )

    def _interleave(a, b, axis):
        """Given two Tensors of static shape, interleave them along axis."""
        assert (
            a.shape[axis] == b.shape[axis] or a.shape[axis] == b.shape[axis] + 1
        )

        # we want to get a: [a1, a2], b: [b1, b2]
        # to a: [a1, 0, a2, 0], b: [0, b1, 0, b2]
        a_shape = list(a.shape)
        a_shape[axis] = a.shape[axis] * 2 - 1

        b_shape = list(b.shape)
        b_shape[axis] = b.shape[axis] * 2 - 1

        a_dil = torch.zeros(a_shape)
        slice_along_axis(a_dil, 0, None, 2, axis).copy_(a)

        b_dil = torch.zeros(b_shape)
        slice_along_axis(b_dil, 0, None, 2, axis).copy_(b)

        a_pad = [[0, 0] for _ in range(a.dim())]
        a_pad[axis][-1] = 1 if a.shape[axis] == b.shape[axis] else 0
        a_pad = a_pad[::-1]
        a_pad = tree.flatten(a_pad)

        b_pad = [[0, 0] for _ in range(b.dim())]
        b_pad[axis] = [1, 0] if a.shape[axis] == b.shape[axis] else [1, 1]
        b_pad = b_pad[::-1]
        b_pad = tree.flatten(b_pad)

        op = torch.bitwise_or if a.dtype == torch.bool else torch.add
        return op(
            torch.nn.functional.pad(a_dil, a_pad),
            torch.nn.functional.pad(b_dil, b_pad),
        )

    def _scan(elems):
        num_elems = elems[0].shape[axis]
        if num_elems < 2:
            return elems

        reduced_elems = _combine(
            [
                slice_along_axis(elem, 0, -1, step=2, axis=axis)
                for elem in elems
            ],
            [
                slice_along_axis(elem, 1, None, step=2, axis=axis)
                for elem in elems
            ],
        )

        odd_elems = _scan(reduced_elems)
        if num_elems % 2 == 0:
            even_elems = _combine(
                [slice_along_axis(e, 0, -1, axis=axis) for e in odd_elems],
                [
                    slice_along_axis(e, 2, None, step=2, axis=axis)
                    for e in elems
                ],
            )
        else:
            even_elems = _combine(
                odd_elems,
                [
                    slice_along_axis(e, 2, None, step=2, axis=axis)
                    for e in elems
                ],
            )

        even_elems = [
            torch.cat(
                [slice_along_axis(elem, 0, 1, axis=axis), result],
                dim=axis,
            )
            for (elem, result) in zip(elems, even_elems)
        ]
        return list(
            builtins.map(
                functools.partial(_interleave, axis=axis), even_elems, odd_elems
            )
        )

    scans = _scan(elems_flat)
    if reverse:
        scans = [torch.flip(scanned, (axis,)) for scanned in scans]

    return tree.pack_sequence_as(elems, scans)


def scatter(indices, values, shape):
    indices = convert_to_tensor(indices)
    values = convert_to_tensor(values)
    zeros = torch.zeros(shape, dtype=values.dtype, device=get_device())

    index_length = indices.shape[-1]
    value_shape = shape[index_length:]
    indices = torch.reshape(indices, [-1, index_length])
    values = torch.reshape(values, [-1] + list(value_shape))

    for i in range(indices.shape[0]):
        index = indices[i]
        zeros[tuple(index)] += values[i]
    return zeros


def scatter_update(inputs, indices, updates, reduction=None):
    inputs = convert_to_tensor(inputs)
    indices = convert_to_tensor(indices, dtype="int64")
    updates = convert_to_tensor(updates, dtype=inputs.dtype)
    indices = torch.transpose(indices, 0, 1)
    idx = tuple(indices)

    outputs = torch.clone(inputs)
    if reduction is None:
        outputs[idx] = updates
    elif reduction == "add":
        # Use index_put_ with accumulate=True for proper accumulation
        outputs.index_put_(idx, updates, accumulate=True)
    elif reduction == "max":
        # Loop-based approach handles both scalar and slice updates.
        # Associative, so sequential application handles duplicates.
        indices_t = indices.T
        for i in range(indices_t.shape[0]):
            idx = tuple(indices_t[i])
            outputs[idx] = torch.maximum(outputs[idx], updates[i])
    elif reduction == "min":
        indices_t = indices.T
        for i in range(indices_t.shape[0]):
            idx = tuple(indices_t[i])
            outputs[idx] = torch.minimum(outputs[idx], updates[i])
    elif reduction == "mul":
        indices_t = indices.T
        for i in range(indices_t.shape[0]):
            idx = tuple(indices_t[i])
            outputs[idx] = outputs[idx] * updates[i]
    else:
        raise ValueError(f"Unsupported reduction: {reduction}")
    return outputs


def slice(inputs, start_indices, shape):
    shape_dtype = to_torch_dtype("int64")
    inputs = convert_to_tensor(inputs)
    start_indices = convert_to_tensor(start_indices).to(shape_dtype)
    shape = convert_to_tensor(shape).to(shape_dtype)

    python_slice = builtins.slice
    slices = [
        python_slice(int(start_index), int(start_index + length))
        for start_index, length in zip(start_indices, shape)
    ]
    return inputs[slices]


def slice_update(inputs, start_indices, updates):
    shape_dtype = to_torch_dtype("int64")
    inputs = convert_to_tensor(inputs)
    start_indices = convert_to_tensor(start_indices).to(shape_dtype)
    updates = convert_to_tensor(updates)

    python_slice = builtins.slice
    slices = [
        python_slice(int(start_index), int(start_index + update_length))
        for start_index, update_length in zip(start_indices, updates.shape)
    ]
    outputs = torch.clone(inputs)
    outputs[slices] = updates
    return outputs


def switch(index, branches, *operands):
    index = convert_to_tensor(index, "int32")
    index = torch.clamp(index, 0, len(branches) - 1)
    return branches[index](*operands)


def while_loop(
    cond,
    body,
    loop_vars,
    maximum_iterations=None,
):
    current_iter = 0
    iteration_check = (
        lambda iter: maximum_iterations is None or iter < maximum_iterations
    )
    is_tuple = isinstance(loop_vars, (tuple, list))
    loop_vars = tuple(loop_vars) if is_tuple else (loop_vars,)
    loop_vars = tree.map_structure(convert_to_tensor, loop_vars)
    while cond(*loop_vars) and iteration_check(current_iter):
        loop_vars = body(*loop_vars)
        if not isinstance(loop_vars, (list, tuple)):
            loop_vars = (loop_vars,)
        loop_vars = tuple(loop_vars)
        current_iter += 1
    return loop_vars if is_tuple else loop_vars[0]


def fori_loop(lower, upper, body_fun, init_val):
    val = init_val
    for i in range(lower, upper):
        val = body_fun(i, val)
    return val


def stop_gradient(variable):
    if isinstance(variable, Variable):
        variable = variable.value
    # We can't use `.requires_grad_(False)` here since it only
    # works when the tensor is a leaf node in the graph.
    return variable.detach()


def unstack(x, num=None, axis=0):
    return x.unbind(axis)


def random_seed_dtype():
    # uint32 doesn't exist in torch, use int32 instead.
    return "int32"


def remat(f):
    """Implementation of rematerialization.

    Args:
        f: The function or operation to rematerialize.
    Returns:
        A function wrapping f that defines a custom gradient, which
        recomputes f on the backwards pass of a gradient call.
    """

    def wrapped(*args, **kwargs):
        return torch.utils.checkpoint.checkpoint(
            f, *args, use_reentrant=False, **kwargs
        )

    return wrapped


class custom_gradient:
    """Decorator for custom gradients.

    Args:
        forward_fn: Forward pass function.
    """

    def __init__(self, forward_fn):
        self.forward_fn = forward_fn

    def __call__(self, *args, **kwargs):
        return CustomGradientFunction.apply(self.forward_fn, *args, **kwargs)


class CustomGradientFunction(torch.autograd.Function):
    """Enables custom forward & backward passes for gradient computation."""

    @staticmethod
    def forward(ctx, forward_fn, *args, **kwargs):
        """Forward pass computation specification.

        Args:
            ctx: Context object.
            forward_fn: Function to compute forward pass.
            *args: Arguments for the forward pass.
            **kwargs: Keyword arguments for the forward pass.
        """
        ctx.forward_fn = forward_fn
        ctx.save_for_backward(*args)
        try:
            output, ctx.grad_fn = forward_fn(*args, **kwargs)
        except:
            output = forward_fn(*args, **kwargs)
            ctx.grad_fn = lambda *args, **kwargs: torch.full((), float("nan"))
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass computation specification.

        Args:
            ctx: Context object.
            grad_output: Gradient with respect to the output.
        """
        args = ctx.saved_tensors
        grad_fn = ctx.grad_fn
        if grad_fn is None:
            raise ValueError("grad_fn must be provided for custom gradient")
        grads = grad_fn(*args, upstream=grad_output)
        if not isinstance(grads, tuple):
            grads = (grads,)
        return (None,) + grads
