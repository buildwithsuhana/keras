import functools
import logging
import re

from keras import Variable
from keras import device
from keras.src import backend
from keras.src import ops
from keras.src.backend import distribution_lib

logger = logging.getLogger(__name__)


class ShardedWeight:
    """Wrapper for a sharded Keras Variable providing a consistent interface.

    Attributes:
        regularizer: Placeholder for weight regularization logic.
    """

    def __init__(self, tensor_shard, name, trainable=True, device_id=None):
        """Initializes the ShardedWeight.

        Args:
            tensor_shard: The tensor slice belonging to this rank.
            name: The original variable name.
            trainable: Boolean indicating if the weight is trainable.
            device_id: Device identifier string for the variable placement.
        """
        dev_name = device_id if device_id else "UNKNOWN_DEVICE"
        print(
            f"   [DEV: {dev_name}] 🧬 Creating Sharded Variable '{name}' shape {tensor_shard.shape}"
        )

        safe_name = name.replace("/", "_").replace(":", "_")
        with device(dev_name):
            self._variable = Variable(
                initializer=tensor_shard, trainable=trainable, name=safe_name
            )

        # Ensure the variable has a path for optimizer tracking
        self._variable._path = name
        self.path = name
        self.regularizer = None

    @property
    def name(self):
        """Returns variable name."""
        return self._variable.name

    @property
    def trainable(self):
        """Returns trainability."""
        return self._variable.trainable

    @property
    def shape(self):
        """Returns sharded shape."""
        return self._variable.shape

    @property
    def dtype(self):
        """Returns data type."""
        return self._variable.dtype

    @property
    def variable(self):
        """Returns the internal Variable object."""
        return self._variable

    @property
    def value(self):
        """Returns the variable value."""
        return self._variable.value

    def numpy(self):
        """Returns numpy representation."""
        return self._variable.numpy()

    def num_elements(self):
        """Returns total element count."""
        return ops.size(self._variable)

    def __repr__(self):
        return f"<ShardedWeight name='{self.name}' shape={self.shape} trainable={self.trainable}>"


class ParameterShardingStrategy:
    """Handles parameter-level sharding logic and configuration normalization."""

    def __init__(self, device_count, rank):
        """Initializes the strategy.

        Args:
            device_count: Total devices in the model axis.
            rank: Current device rank.
        """
        self.device_count = device_count
        self.rank = rank
        self.sharded_weights = {}
        self.weight_mapping = {}
        self.sharded_weights_by_id = {}
        self.param_path_map = {}
        self._id_to_param_map = {}

    def shard_model_parameters(self, model, config, device_id):
        """Orchestrates the sharding of model parameters.

        Args:
            model: Original model instance.
            config: LayoutMap with sharding rules.
            device_id: Targeted device identifier.

        Returns:
            Tuple of (sharded_model, modified_parameters_set).
        """
        ParameterShardedModel = _define_parameter_sharded_model()
        print(f"🔧 Applying parameter-level sharding to {model.name}")

        # Map build & Auto-Discovery
        self.param_path_map = {w.path: w for w in model.weights}
        for w in model.weights:
            ref_fn = getattr(w, "experimental_ref", None)
            ref = ref_fn() if ref_fn else w
            self._id_to_param_map[id(ref)] = (w.path, w)

        # Preserve the original variable metadata to avoid mutating the model
        # across multiple shard creations for different ranks.
        original_metadata = {}
        for w in model.weights:
            original_metadata[id(w)] = (
                getattr(w, "_shape", None),
                getattr(w, "_ndim", None),
                getattr(w, "_value", None),
            )

        # Normalize configuration keys to string paths
        norm_rules = {}
        for pattern, action in list(config.state_rules.items()):
            if isinstance(pattern, int) and pattern in self._id_to_param_map:
                path, _ = self._id_to_param_map[pattern]
                norm_rules[path] = action
                del config.state_rules[pattern]
        config.state_rules.update(norm_rules)

        modified = set()

        # 1. Shard variables and apply shape-lying hack
        for pattern, action in config.state_rules.items():
            if callable(action):
                for name, param in self._find_matching_parameters(
                    model, pattern
                ):
                    if name in self.sharded_weights:
                        modified.add(name)
                        continue

                    pr_fn = getattr(param, "experimental_ref", None)
                    pid = id(pr_fn()) if pr_fn else id(param)

                    if pid in self.sharded_weights_by_id:
                        self.sharded_weights[name] = self.sharded_weights_by_id[
                            pid
                        ]
                        modified.add(name)
                        continue

                    original_shape = param.shape
                    shard = action(param, self.rank)
                    self.sharded_weights[name] = shard
                    self.sharded_weights_by_id[pid] = shard
                    self.weight_mapping[name] = {
                        "original": original_shape,
                        "sharded": shard.shape,
                    }

                    # Update original variable to sharded state (backend-agnostic hack)
                    shard_tensor = self._convert_to_tensor(shard)

                    def update_shape():
                        from keras.src.backend import backend as backend_fn

                        if backend_fn() == "torch":
                            from keras.src.backend.torch.core import get_device

                            if get_device() == "meta":
                                return
                        param._shape = shard.shape
                        param._ndim = len(shard.shape)

                    update_shape()
                    if hasattr(param, "_value"):
                        param._value = shard_tensor

                    modified.add(name)
                    print(
                        f"   ✅ Sharded {name}: {original_shape} -> {shard.shape}"
                    )

        # 2. Patch layers recursively with output rules (communication ops)
        from keras.src.backend import backend as backend_fn

        patched_count = 0
        is_meta = False
        if backend_fn() == "torch":
            from keras.src.backend.torch.core import get_device

            if get_device() == "meta":
                is_meta = True

        for layer in model._flatten_layers(recursive=True, include_self=True):
            if is_meta:
                continue

            lp = getattr(layer, "path", None) or layer.name
            lp_s = str(lp)

            # Find and apply sharded variables to layer attributes
            for attr_name in dir(layer):
                try:
                    attr = getattr(layer, attr_name)
                    if isinstance(attr, Variable):
                        if attr.path in self.sharded_weights:
                            # Shape-lying for the layer variable itself
                            sharded_val = self.sharded_weights[attr.path]
                            sharded_tensor = self._convert_to_tensor(
                                sharded_val
                            )
                            attr._shape = sharded_val.shape
                            attr._ndim = len(attr._shape)
                            if hasattr(attr, "_value"):
                                attr._value = sharded_tensor
                except Exception:
                    continue

            for pat, rule in config.output_rules.items():
                pat_s = str(pat)
                # Flexible matching: exact, suffix, or regex
                matched = (
                    pat_s == lp_s
                    or lp_s.endswith("/" + pat_s)
                    or pat_s.endswith("/" + lp_s)
                    or (isinstance(pat_s, str) and re.search(pat_s, lp_s))
                )

                if matched:
                    actual_rule = (
                        rule.get(0) if isinstance(rule, dict) else rule
                    )
                    self._patch_layer(layer, actual_rule)
                    patched_count += 1
                    break

        if not is_meta:
            print(f"🎯 Patched {patched_count} layers with communication rules")

        sharded_model = ParameterShardedModel(model, self, config, device_id)

        # Restore the original model variable metadata so subsequent shard
        # creations start from the full original model state.
        for w in model.weights:
            metadata = original_metadata.get(id(w))
            if metadata is not None:
                shape, ndim, value = metadata
                if shape is not None:
                    w._shape = shape
                if ndim is not None:
                    w._ndim = ndim
                if value is not None:
                    w._value = value

        print(f"🎯 Sharding complete: {len(modified)} parameters sharded")
        return sharded_model, modified

    def _patch_layer(self, layer, rule):
        """Patches a layer's call and compute_output_shape methods."""
        if hasattr(layer, "_is_tp_patched"):
            return

        # Patch call
        old_call = layer.call

        @functools.wraps(old_call)
        def sharded_call(*args, **kwargs):
            is_reversible = "ReversibleEmbedding" in layer.__class__.__name__
            is_reverse = is_reversible and (
                getattr(layer, "reverse", False)
                or kwargs.get("reverse", False)
            )

            # Skip communication ops during symbolic trace
            from keras.src.backend import backend

            # Row-sharded embeddings must translate global indices to local
            # indices before lookup and contribute zeros for rows owned by
            # other ranks. This is required by both JAX and Torch backends.
            if backend() != "torch" and not is_reverse:
                is_embedding = (
                    "Embedding" in layer.__class__.__name__
                    or hasattr(layer, "_embeddings")
                )
                if is_embedding:
                    target_var = next(
                        (
                            weight
                            for weight in layer.weights
                            if weight.path.endswith("/embeddings")
                        ),
                        getattr(layer, "_embeddings", None),
                    )
                    mapping = self.weight_mapping.get(
                        getattr(target_var, "path", None)
                    )
                    if mapping and mapping["original"][0] != mapping[
                        "sharded"
                    ][0]:
                        inputs = args[0] if args else kwargs.get("inputs")
                        if inputs is not None:
                            nelem = mapping["original"][0]
                            default_size, remainder = divmod(
                                nelem, self.device_count
                            )
                            start_idx = self.rank * default_size + min(
                                self.rank, remainder
                            )
                            shard_size = default_size + (
                                self.rank < remainder
                            )
                            end_idx = start_idx + shard_size
                            dtype = str(inputs.dtype)

                            if dtype.startswith(("int", "uint")):
                                mask = ops.logical_and(
                                    inputs >= start_idx, inputs < end_idx
                                )
                                local_inputs = ops.where(
                                    mask,
                                    inputs - start_idx,
                                    ops.zeros_like(inputs),
                                )
                                new_args = list(args)
                                if new_args:
                                    new_args[0] = local_inputs
                                    out = old_call(*new_args, **kwargs)
                                else:
                                    new_kwargs = dict(kwargs)
                                    new_kwargs["inputs"] = local_inputs
                                    out = old_call(**new_kwargs)
                            elif "PositionEmbedding" in layer.__class__.__name__:
                                positions = kwargs.get("positions")
                                if args and len(args) > 2:
                                    positions = args[2]
                                if positions is None:
                                    start_index = kwargs.get("start_index", 0)
                                    if args and len(args) > 1:
                                        start_index = args[1]
                                    positions = ops.arange(
                                        start_index,
                                        start_index + inputs.shape[1],
                                    )
                                mask = ops.logical_and(
                                    positions >= start_idx, positions < end_idx
                                )
                                local_positions = ops.where(
                                    mask,
                                    positions - start_idx,
                                    ops.zeros_like(positions),
                                )
                                new_kwargs = dict(kwargs)
                                new_kwargs["positions"] = local_positions
                                out = old_call(inputs, **new_kwargs)
                            else:
                                out = old_call(*args, **kwargs)

                            out = ops.where(
                                ops.expand_dims(mask, -1),
                                out,
                                ops.zeros_like(out),
                            )
                            if callable(rule):
                                return rule(out)
                            if isinstance(rule, str):
                                return self._comm(out, rule)
                            return out

            if backend() == "torch":
                from keras.src.backend.common.symbolic_scope import (
                    get_symbolic_scope,
                )
                from keras.src.backend.torch.core import get_device

                if get_device() == "meta" or get_symbolic_scope() is not None:
                    return old_call(*args, **kwargs)

                is_embedding = (
                    "Embedding" in layer.__class__.__name__
                    or hasattr(layer, "_embeddings")
                ) and not is_reverse

                if is_embedding:
                    target_var = None
                    for w in layer.weights:
                        if w.path.endswith("/embeddings"):
                            target_var = w
                            break
                    if target_var is None:
                        target_var = getattr(layer, "_embeddings", None)

                    weight_path = getattr(target_var, "path", None)
                    mapping = self.weight_mapping.get(weight_path)

                    if (
                        mapping
                        and len(mapping["original"]) > 0
                        and mapping["original"][0] != mapping["sharded"][0]
                    ):
                        import torch

                        inputs = args[0] if args else kwargs.get("inputs")
                        if inputs is not None:
                            if not isinstance(inputs, torch.Tensor):
                                inputs = torch.as_tensor(inputs)

                            # Calculate shard boundaries using np.array_split logic
                            nelem = mapping["original"][0]
                            ncat = self.device_count
                            default_size = nelem // ncat
                            remainder = nelem % ncat

                            if self.rank < remainder:
                                start_idx = self.rank * (default_size + 1)
                                my_shard_size = default_size + 1
                            else:
                                start_idx = (
                                    self.rank * default_size + remainder
                                )
                                my_shard_size = default_size
                            end_idx = start_idx + my_shard_size

                            if not inputs.is_floating_point():
                                # Standard index lookup (Token Embedding)
                                mask = (inputs >= start_idx) & (
                                    inputs < end_idx
                                )
                                local_inputs = torch.where(
                                    mask,
                                    inputs - start_idx,
                                    torch.zeros_like(inputs),
                                )

                                new_args = list(args)
                                if args:
                                    new_args[0] = local_inputs
                                else:
                                    kwargs["inputs"] = local_inputs

                                # Call with local indices
                                out = old_call(*new_args, **kwargs)

                                # Zero out embeddings for indices not owned by this shard
                                out = torch.where(
                                    mask.unsqueeze(-1),
                                    out,
                                    torch.zeros_like(out),
                                )
                            elif (
                                "PositionEmbedding"
                                in layer.__class__.__name__
                            ):
                                # Float inputs (Position Embedding receiving hidden states)
                                # We need to find the sequence positions.
                                start_index = kwargs.get("start_index", 0)
                                if args and len(args) > 1:
                                    start_index = args[1]

                                positions = kwargs.get("positions")
                                if args and len(args) > 2:
                                    positions = args[2]

                                seq_len = inputs.shape[1]
                                if positions is None:
                                    positions = torch.arange(
                                        start_index,
                                        start_index + seq_len,
                                        device=inputs.device,
                                    )

                                # Mask and shift POSITIONS instead of INPUTS
                                mask = (positions >= start_idx) & (
                                    positions < end_idx
                                )
                                local_positions = torch.where(
                                    mask,
                                    positions - start_idx,
                                    torch.zeros_like(positions),
                                )

                                # Update kwargs/args to use local_positions
                                new_kwargs = dict(kwargs)
                                if "positions" in kwargs or (
                                    args and len(args) > 2
                                ):
                                    new_kwargs["positions"] = local_positions
                                else:
                                    # Handle case where positions was not passed
                                    # but we generated it.
                                    # If it's KerasHub PositionEmbedding, it expects it.
                                    new_kwargs["positions"] = local_positions

                                # Call old_call
                                out = old_call(inputs, **new_kwargs)

                                # Zero out output based on mask
                                # Broadcast mask to (batch, seq_len, 1)
                                if mask.dim() == 1:
                                    mask = mask.unsqueeze(0)  # (1, seq_len)

                                out = torch.where(
                                    mask.unsqueeze(-1),
                                    out,
                                    torch.zeros_like(out),
                                )
                            else:
                                # Fallback for other embedding types
                                out = old_call(*args, **kwargs)

                            # Apply the communication rule (usually AllReduce sum)
                            if rule:
                                if callable(rule):
                                    return rule(out)
                                elif isinstance(rule, str):
                                    return self._comm(out, rule)
                            return out

            # Rule 3: For Row Parallel layers, bias should be added AFTER all_reduce
            # to avoid summing the bias N times.
            from keras.src.distribution.tensor_parallel.autoconfig import (
                _reduce_sum,
            )

            if rule == "parallel_dropout":
                # Rule 5: Parallel regions use different seeds
                if hasattr(layer, "seed_generator"):
                    # We slightly shift the seed based on rank to get different masks
                    # This is a simple way to achieve Rule 5 parallel RNG behavior
                    seed_state = layer.seed_generator.state
                    seed_state.assign(seed_state.value + self.rank * 1000)
                return old_call(*args, **kwargs)

            # Special override for ReversibleEmbedding in reverse mode:
            # force all_gather on axis -1
            if is_reverse:
                out = old_call(*args, **kwargs)
                return distribution_lib.all_gather(
                    out, axis=-1, axis_name="model"
                )

            use_bias = getattr(layer, "use_bias", False)
            if (
                use_bias
                and rule == _reduce_sum
                and hasattr(layer, "bias")
                and layer.bias is not None
            ):
                # Temporarily disable bias addition in the original call
                layer.use_bias = False
                try:
                    out = old_call(*args, **kwargs)
                finally:
                    layer.use_bias = True

                # Apply the rule (AllReduce Sum)
                out = rule(out)

                # Add the bias manually after AllReduce
                out = out + layer.bias
                return out

            out = old_call(*args, **kwargs)
            if rule:
                if callable(rule):
                    out = rule(out)
                elif isinstance(rule, str):
                    out = self._comm(out, rule)
            return out

        layer.call = sharded_call

        # Disable input_spec validation for sharded layers
        # as it often conflicts with sharded input shapes.
        def disable_input_spec(l):
            if hasattr(l, "input_spec"):
                l.input_spec = None
            if hasattr(l, "_flatten_layers"):
                for sub_l in l._flatten_layers(
                    recursive=True, include_self=False
                ):
                    if hasattr(sub_l, "input_spec"):
                        sub_l.input_spec = None

        disable_input_spec(layer)

        # Patch compute_output_shape to return full shape if gathering
        old_cos = layer.compute_output_shape

        @functools.wraps(old_cos)
        def sharded_cos(input_shape):
            shape = old_cos(input_shape)
            if isinstance(rule, str) and "gather" in rule:
                parts = rule.split(" ")
                dim = (
                    int(parts[-1])
                    if len(parts) > 1 and parts[-1].lstrip("-").isdigit()
                    else -1
                )
                # Adjust negative axis to positive
                axis = dim if dim >= 0 else len(shape) + dim
                new_shape = list(shape)
                if new_shape[axis] is not None:
                    new_shape[axis] *= self.device_count
                return tuple(new_shape)
            return shape

        layer.compute_output_shape = sharded_cos

        layer._is_tp_patched = True
        print(f"   🔗 Patched layer {layer.name} with communication rule")

    def _get_layer_mlp_type(self, layer):
        from keras.src.distribution.tensor_parallel.autoconfig import (
            analyze_dense_layer,
        )

        return analyze_dense_layer(layer)

    def _comm(self, val, rule):
        """Internal communication wrapper."""
        if rule == "parallel_dropout":
            # Rule 5: Parallel Dropout needs different seeds on different devices.
            # We achieve this by adding the rank to the seed if possible,
            # or by doing nothing if the backend already handles it.
            # In Keras 3.0, we can use a seed_generator or just rely on the rank
            # being part of the global state.
            # For simplicity, we ensure that if we are in a parallel region,
            # we are not syncing seeds.
            return val
        if "sum" in rule or "allreduce" in rule:
            res = distribution_lib.all_reduce(val, op="sum", axis_name="model")
            return res
        if "gather" in rule:
            parts = rule.split(" ")
            dim = (
                int(parts[-1])
                if len(parts) > 1 and parts[-1].lstrip("-").isdigit()
                else -1
            )
            res = distribution_lib.all_gather(val, axis=dim, axis_name="model")
            return res
        return val

    def _convert_to_tensor(self, value):
        """Convert a value to a backend tensor with a stable fallback."""
        try:
            return ops.convert_to_tensor(value)
        except TypeError:
            return backend.core.convert_to_tensor(value)

    def _find_matching_parameters(self, model, pattern):
        """Matches a pattern to model weights."""
        if isinstance(pattern, int):
            return (
                [self._id_to_param_map[pattern]]
                if pattern in self._id_to_param_map
                else []
            )
        if not isinstance(pattern, str):
            return []
        if pattern in self.param_path_map:
            return [(pattern, self.param_path_map[pattern])]
        suffix = "/" + pattern
        return [
            (p, w) for p, w in self.param_path_map.items() if p.endswith(suffix)
        ]


def _define_parameter_sharded_model():
    """Defines the wrapper model class dynamically."""
    from keras.src.models import Model

    class ParameterShardedModel(Model):
        """Wrapper model implementing distributed forward pass logic via weight injection."""

        def __init__(
            self, original_model, sharding_strategy, config, device_id
        ):
            """Initializes the model and caches mappings."""
            super().__init__(name=original_model.name)
            self.original_model = original_model
            self.sharding_strategy = sharding_strategy
            self.config = config
            self._device = device_id

            if not self.original_model.built and self.original_model.inputs:
                self.original_model.build(self.original_model.inputs[0].shape)

            self._build_and_cache_weights()
            # Set built flag to True and full_build to skip redundant fit initialization
            self.built = True
            self._built = True
            print("🚀 ParameterShardedModel created and marked as BUILT")

        def _build_and_cache_weights(self):
            """Merges sharded and original weights into a definitive list."""
            ws, self._var_map = [], {}
            sharded_ids = set(
                self.sharding_strategy.sharded_weights_by_id.keys()
            )

            for name, shard in self.sharding_strategy.sharded_weights.items():
                sharded_var = ShardedWeight(
                    shard, name, device_id=self._device
                ).variable
                ws.append(sharded_var)
                # Map original variable path and ref to our new sharded Variable object
                orig_var = self.sharding_strategy.param_path_map.get(name)
                if orig_var is not None:
                    self._var_map[orig_var.path] = sharded_var
                    if hasattr(orig_var, "experimental_ref"):
                        ref = orig_var.experimental_ref()
                        self._var_map[id(ref)] = sharded_var

            for w in self.original_model.weights:
                if w.path not in self._var_map:
                    # Explicitly move non-sharded weights to the target device
                    with device(self._device):
                        new_v = Variable(
                            initializer=w.value,
                            trainable=w.trainable,
                            name=w.name.replace("/", "_").replace(":", "_")
                            + "_sharded",
                        )
                        new_v._path = w.path
                    ws.append(new_v)
                    self._var_map[w.path] = new_v
                    if hasattr(w, "experimental_ref"):
                        ref = w.experimental_ref()
                        self._var_map[id(ref)] = new_v

            self._weights_list = ws
            self._trainable_weights_list = [v for v in ws if v.trainable]
            self._non_trainable_weights_list = [
                v for v in ws if not v.trainable
            ]

        @property
        def weights(self):
            """Returns model weights (shards for sharded params, original for others)."""
            return self._weights_list

        @property
        def trainable_weights(self):
            return self._trainable_weights_list

        @property
        def non_trainable_weights(self):
            return self._non_trainable_weights_list

        def compute_output_shape(self, input_shape):
            return self.original_model.compute_output_shape(input_shape)

        def compute_output_spec(self, *args, **kwargs):
            if args:
                return self.original_model.compute_output_spec(args[0])
            return self.original_model.compute_output_spec(**kwargs)

        @property
        def trainable_variables(self):
            # Skip sharded weights during symbolic trace
            from keras.src.backend import backend as backend_fn

            if backend_fn() == "torch":
                from keras.src.backend.torch.core import get_device

                if get_device() == "meta":
                    return self.original_model.trainable_variables

            return super().trainable_variables

        @property
        def non_trainable_variables(self):
            # Skip sharded weights during symbolic trace
            from keras.src.backend import backend as backend_fn

            if backend_fn() == "torch":
                from keras.src.backend.torch.core import get_device

                if get_device() == "meta":
                    return self.original_model.non_trainable_variables

            return super().non_trainable_variables

        def call(self, inputs, training=None, mask=None):
            """Forward pass that correctly handles sharded variable state."""
            from keras.src import tree
            from keras.src.backend import is_tensor

            # Skip weight replacement during symbolic trace
            from keras.src.backend import backend

            if backend() == "torch":
                from keras.src.backend.torch.core import get_device

                if get_device() == "meta":
                    return self.original_model(
                        inputs, training=training, mask=mask
                    )

            # Since we've already updated the layers' internal state via shape-lying,
            # we can call the original model directly.
            # The sharded variables are stored in self._weights_list and tracked via
            # the framework through self.trainable_weights and self.non_trainable_weights.

            # Build a temporary mapping of original_model variables to their sharded values
            # for the duration of this call
            original_state = {}

            # Temporarily replace original_model's variables with our sharded versions
            for orig_var in self.original_model.trainable_variables:
                var_id = id(orig_var)
                original_state[var_id] = (
                    getattr(orig_var, "_shape", None),
                    getattr(orig_var, "_ndim", None),
                    getattr(orig_var, "_value", None),
                )
                sharded_var = self._var_map.get(
                    orig_var.path,
                    self._var_map.get(
                        id(orig_var.experimental_ref())
                        if hasattr(orig_var, "experimental_ref")
                        else id(orig_var),
                        None,
                    ),
                )
                if sharded_var is not None:
                    orig_var._shape = sharded_var.shape
                    orig_var._ndim = (
                        len(sharded_var.shape) if sharded_var.shape else 0
                    )
                    if hasattr(orig_var, "_value"):
                        # Use the underlying leaf Parameter object (_value)
                        # instead of .value which might return an autocasted tensor.
                        val = getattr(sharded_var, "_value", sharded_var.value)
                        if (
                            hasattr(sharded_var, "trainable")
                            and sharded_var.trainable
                            and hasattr(val, "requires_grad")
                            and not val.requires_grad
                        ):
                            val.requires_grad_(True)
                        orig_var._value = val

            for orig_var in self.original_model.non_trainable_variables:
                var_id = id(orig_var)
                if (
                    var_id not in original_state
                ):  # Only store if not already stored
                    original_state[var_id] = (
                        getattr(orig_var, "_shape", None),
                        getattr(orig_var, "_ndim", None),
                        getattr(orig_var, "_value", None),
                    )
                sharded_var = self._var_map.get(
                    orig_var.path,
                    self._var_map.get(
                        id(orig_var.experimental_ref())
                        if hasattr(orig_var, "experimental_ref")
                        else id(orig_var),
                        None,
                    ),
                )
                if sharded_var is not None:
                    orig_var._shape = sharded_var.shape
                    orig_var._ndim = (
                        len(sharded_var.shape) if sharded_var.shape else 0
                    )
                    if hasattr(orig_var, "_value"):
                        orig_var._value = getattr(
                            sharded_var, "_value", sharded_var.value
                        )

            try:
                # Call the original model
                outputs = self.original_model.call(
                    inputs, training=training, mask=mask
                )

                # Special handling for sharded outputs (e.g. final Dense layer)
                # If the output tensor is sharded along its last dimension (column-parallel),
                # and it was NOT gathered yet, we must gather it so the loss function
                # sees the full vocabulary/classes.
                
                # Heuristic: if the output shape doesn't match the original model's 
                # expected output shape (if we could know it), we gather.
                # Since we don't easily know the full expected shape here, 
                # we check if any of the "leaf" layers that contributed to the output
                # are sharded on axis 1 (column-parallel) but don't have a gather rule.
                
                def maybe_gather(out):
                    if not is_tensor(out):
                        return out
                    
                    # If the last dim size * device_count matches a typical vocabulary size
                    # or if we can find the layer that produced it.
                    # A more reliable way: check if the output layer was sharded.
                    output_layer = self.original_model.layers[-1]
                    
                    # If it's a ColumnParallel layer, restored shape should have 
                    # full dim at -1.
                    for w in output_layer.weights:
                        mapping = self.sharding_strategy.weight_mapping.get(w.path)
                        if mapping and len(mapping["original"]) > 1:
                            if mapping["original"][-1] != mapping["sharded"][-1]:
                                # It's column-parallel on the last dimension.
                                # Check if the current output tensor has the sharded size.
                                if out.shape[-1] == mapping["sharded"][-1]:
                                    return distribution_lib.all_gather(
                                        out, axis=-1, axis_name="model"
                                    )
                    return out

                outputs = tree.map_structure(maybe_gather, outputs)
            finally:
                # Restore original variable state
                for orig_var in list(
                    self.original_model.trainable_variables
                ) + list(self.original_model.non_trainable_variables):
                    var_id = id(orig_var)
                    if var_id in original_state:
                        shape, ndim, value = original_state[var_id]
                        if shape is not None:
                            orig_var._shape = shape
                        if ndim is not None:
                            orig_var._ndim = ndim
                        if value is not None and hasattr(orig_var, "_value"):
                            orig_var._value = value

            return outputs

        def get_config(self):
            return self.original_model.get_config()

        @classmethod
        def from_config(cls, config, custom_objects=None):
            return cls(**config)

    return ParameterShardedModel


def make_parameter_sharded_model(module, config, rank, device_count, device_id):
    """Factory function to create a parameter-sharded model."""
    strat = ParameterShardingStrategy(device_count, rank)
    return strat.shard_model_parameters(module, config, device_id)
