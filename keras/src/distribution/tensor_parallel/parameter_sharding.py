import logging
import inspect
import re
from typing import Any, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

import numpy as np

from keras.src.backend import distribution_lib
from keras.src import ops
from keras.src import layers
from keras import Variable, device 

from keras.src.distribution.tensor_parallel.tensor_layout import LayoutMap

if TYPE_CHECKING:
    from keras.src.models import Model

logger = logging.getLogger(__name__)


class ShardedWeight:
    """
    A wrapper for a sharded Keras Variable, providing a consistent interface.
    """

    def __init__(self, tensor_shard, name, trainable=True, device_id=None):
        current_dev_name = device_id if device_id else "UNKNOWN_DEVICE"
        print(
            f"   [DEV: {current_dev_name}] 🧬 Creating Sharded Variable "
            f"'{name}' with shape {tensor_shard.shape}"
        )

        with device(current_dev_name):
            self._variable = Variable(
                initializer=tensor_shard, trainable=trainable, name=name
            )
        self.regularizer = None

    @property
    def name(self):
        return self._variable.name

    @property
    def trainable(self):
        return self._variable.trainable

    @property
    def shape(self):
        return self._variable.shape

    @property
    def dtype(self):
        return self._variable.dtype

    @property
    def variable(self):
        return self._variable

    @property
    def value(self):
        return self._variable.value

    def numpy(self):
        return self._variable.numpy()

    def num_elements(self):
        return ops.size(self._variable)

    def __repr__(self):
        return (
            f"<ShardedWeight name='{self.name}' "
            f"shape={self.shape} trainable={self.trainable}>"
        )


class ParameterShardingStrategy:
    """
    Parameter-level sharding strategy that works with any Keras model.
    """

    def __init__(self, device_count: int, rank: int):
        self.device_count = device_count
        self.rank = rank
        self.sharded_weights = {}
        # self.original_weights = {}
        self.weight_mapping = {}
        self.sharded_weights_by_id = {}

    def shard_model_parameters(
        self,
        model,
        config: LayoutMap,
        device_id: Any,
    ) -> Tuple["Model", Set[str]]:
        """
        Shard model parameters without rebuilding the model structure.
        """
        ParameterShardedModel = _define_parameter_sharded_model()

        print(f"🔧 Applying parameter-level sharding to {model.name}")

        # self._store_original_weights(model)
        modified_parameters = set()

        for pattern, action in config.state_rules.items():
            # FIX: Check if action is callable (lambda) instead of isinstance(Split)
            if callable(action):
                matching_params = self._find_matching_parameters(model, pattern)

                for param_name, param in matching_params:
                    try:
                        param_id = id(param.experimental_ref())
                    except AttributeError:
                        param_id = id(param)

                    # Weight tying logic
                    if param_id in self.sharded_weights_by_id:
                        self.sharded_weights[param_name] = self.sharded_weights_by_id[
                            param_id
                        ]
                        
                        # Find existing name for logging/mapping
                        existing_param_name = "unknown"
                        for name, shard in self.sharded_weights.items():
                            if shard is self.sharded_weights_by_id[param_id]:
                                existing_param_name = name
                                break

                        self.weight_mapping[param_name] = self.weight_mapping.get(
                            existing_param_name, {}
                        )
                        modified_parameters.add(param_name)
                        print(
                            f"   🔗 Tied {param_name} to existing shard from {existing_param_name}"
                        )
                        continue

                    # Execute the lambda to get the shard (slice)
                    sharded_tensor = action(param, self.rank)

                    self.sharded_weights[param_name] = sharded_tensor
                    self.sharded_weights_by_id[param_id] = sharded_tensor
                    if self.rank == self.device_count - 1:
                         # We replace the massive tensor with a 1-byte placeholder
                         # Keras Variables are mutable.
                         try:
                             dummy = np.zeros((1,), dtype=param.dtype)
                             param.assign(dummy)
                             print(f"   🗑️  Freed original memory for {param_name}")
                         except:
                             pass

                    self.weight_mapping[param_name] = {
                        "original_shape": param.shape,
                        "sharded_shape": sharded_tensor.shape,
                        "action": action,
                    }

                    modified_parameters.add(param_name)
                    print(
                        f"   ✅ Sharded {param_name}: {param.shape} -> {sharded_tensor.shape}"
                    )

        sharded_model = ParameterShardedModel(
            original_model=model,
            sharding_strategy=self,
            config=config,
            device_id=device_id,
        )

        print(
            f"🎯 Parameter sharding completed: {len(modified_parameters)} parameters sharded"
        )
        return sharded_model, modified_parameters

    def _store_original_weights(self, model):
        """Store original weights for reference."""
        
        def find_weights_recursive(current_layer, prefix=""):
            name = current_layer.name
            full_name = f"{prefix}.{name}" if prefix else name

            if hasattr(current_layer, "weights") and current_layer.weights:
                for weight in current_layer.weights:
                    cleaned_name = weight.name.split("/")[-1].split(":")[0]
                    param_name = f"{full_name}.{cleaned_name}"
                    
                    # FIX: Check for numpy capability before access
                    if hasattr(weight, 'numpy'):
                        self.original_weights[param_name] = weight.numpy()

            if hasattr(current_layer, "layers") and current_layer.layers:
                for sub_layer in current_layer.layers:
                    find_weights_recursive(sub_layer, full_name)

            for attr_name in dir(current_layer):
                if attr_name.startswith("__") and attr_name.endswith("__"):
                    continue
                try:
                    attr = getattr(current_layer, attr_name)
                except Exception:
                    continue
                if isinstance(attr, layers.Layer) and attr is not current_layer:
                    find_weights_recursive(attr, full_name)
                elif isinstance(attr, (list, tuple)):
                    for item in attr:
                        if isinstance(item, layers.Layer):
                            find_weights_recursive(item, full_name)

        # Start recursion
        find_weights_recursive(model, prefix="")

    def _find_matching_parameters(self, model, pattern: str) -> List[Tuple[str, Any]]:
        """
        Find parameters that match the given pattern using smart recursion.
        """
        matching_params = []
        processed_layers = set()

        def search_layer_recursive(current_layer, prefix=""):
            if id(current_layer) in processed_layers:
                return
            processed_layers.add(id(current_layer))

            name = current_layer.name
            full_name = f"{prefix}.{name}" if prefix else name

            if hasattr(current_layer, "weights") and current_layer.weights:
                for weight in current_layer.weights:
                    cleaned_weight_name = weight.name.split("/")[-1].split(":")[0]
                    param_name = f"{full_name}.{cleaned_weight_name}"

                    # FIX: Use fullmatch for strict pattern adherence
                    if re.fullmatch(pattern, param_name):
                        matching_params.append((param_name, weight))

            if hasattr(current_layer, "layers") and current_layer.layers:
                for sub_layer in current_layer.layers:
                    search_layer_recursive(sub_layer, full_name)

            for attr_name in dir(current_layer):
                if attr_name.startswith("__") and attr_name.endswith("__"):
                    continue

                try:
                    attr = getattr(current_layer, attr_name)
                except Exception:
                    continue

                if isinstance(attr, layers.Layer) and attr is not current_layer:
                    search_layer_recursive(attr, full_name)

                elif isinstance(attr, (list, tuple)):
                    for item in attr:
                        if isinstance(item, layers.Layer):
                            search_layer_recursive(item, full_name)

        search_layer_recursive(model, prefix="")
        return matching_params


def _define_parameter_sharded_model():
    """
    Factory function to define and return the ParameterShardedModel class.
    This delays the import of keras.src.models.Model to break circular dependencies.
    """
    from keras.src.models import Model
    from keras.src.models import Functional

    class ParameterShardedModel(Model):
        """
        Wrapper model that handles parameter sharding without rebuilding the structure.
        """

        def __init__(
            self,
            original_model: Model,
            sharding_strategy: ParameterShardingStrategy,
            config: LayoutMap,
            device_id: Any,
        ):
            super().__init__(name=original_model.name)

            self.original_model = original_model
            self.sharding_strategy = sharding_strategy
            self.config = config
            self._device = device_id

            # Build if not built (vital for inputs access)
            if not self.original_model.built and self.original_model.inputs:
                self.original_model.build(self.original_model.inputs[0].shape)

            self._build_and_cache_weights()

            print("🚀 ParameterShardedModel created successfully")

        @property
        def device(self):
            return self._device

        def _build_and_cache_weights(self):
            print("   - Building and caching the definitive weights list...")
            weights_list = []

            sharded_weight_ids = set(
                self.sharding_strategy.sharded_weights_by_id.keys()
            )

            for (
                param_name,
                sharded_tensor,
            ) in self.sharding_strategy.sharded_weights.items():
                weights_list.append(
                    ShardedWeight(
                        sharded_tensor, 
                        param_name, 
                        device_id=self._device
                    )
                )

            unsharded_count = 0
            for weight in self.original_model.weights:
                try:
                    weight_id = id(weight.experimental_ref())
                except AttributeError:
                    weight_id = id(weight)

                if weight_id not in sharded_weight_ids:
                    weights_list.append(weight)
                    unsharded_count += 1

            self._weights_list = weights_list

        @property
        def weights(self):
            return self._weights_list
        
        # --- FIX: Robust Output Spec handling ---
        def compute_output_shape(self, input_shape):
            return self.original_model.compute_output_shape(input_shape)

        def compute_output_spec(self, *args, **kwargs):
            """
            Fix: Accept *args and **kwargs to handle 'mask' or other arguments 
            Keras passes during symbolic execution, but only forward the input_spec 
            to the underlying model.
            """
            # Usually args[0] is the input_spec/inputs
            try:
                return self.original_model.compute_output_spec(args[0])
            except Exception:
                # Fallback for older Keras versions or differing signatures
                return self.original_model.compute_output_spec(*args, **kwargs)

        def call(self, inputs, training=None, mask=None):

            tensor_cache = {}

            # 1. Cache Inputs
            if isinstance(inputs, dict):
                for inp_tensor in self.original_model.inputs:
                    name = inp_tensor.name
                    clean_name = name.split(":")[0]
                    # Prefer exact name key, then cleaned name; store by id
                    if name in inputs:
                        val = inputs[name]
                    elif clean_name in inputs:
                        val = inputs[clean_name]
                    else:
                        val = None

                    if val is not None:
                        tensor_cache[id(inp_tensor)] = val
                        # also store by cleaned name so lookups can match
                        tensor_cache[clean_name] = val
                        tensor_cache[name] = val
            else:
                input_list = ops.convert_to_tensor(inputs)
                if not isinstance(input_list, (list, tuple)):
                    input_list = [input_list]
                
                for i, inp_tensor in enumerate(self.original_model.inputs):
                    if i < len(input_list):
                        val = input_list[i]
                        tensor_cache[id(inp_tensor)] = val
                        try:
                            name = inp_tensor.name
                            clean_name = name.split(":")[0]
                            tensor_cache[clean_name] = val
                            tensor_cache[name] = val
                        except Exception:
                            # best-effort: ignore if symbolic name not available
                            pass

            # 2. Iterate Layers
            for layer in self.original_model.layers:
                if isinstance(layer, layers.InputLayer):
                    continue
                
                # Reconstruct inputs for this layer from the cache
                layer_inputs = []
                for node in layer._inbound_nodes:
                    for symbolic_input_tensor in node.input_tensors:
                        key_id = id(symbolic_input_tensor)
                        if key_id in tensor_cache:
                            layer_inputs.append(tensor_cache[key_id])
                        else:
                            # Fallback: try matching by cleaned symbolic name
                            try:
                                name = symbolic_input_tensor.name
                                clean_name = name.split(":")[0]
                            except Exception:
                                name = None
                                clean_name = None

                            if clean_name and clean_name in tensor_cache:
                                layer_inputs.append(tensor_cache[clean_name])
                            elif name and name in tensor_cache:
                                layer_inputs.append(tensor_cache[name])

                # --- Robust handling for Functional/Model inputs ---
                # Some backbone models (Gemma/Transformers) expect dictionary
                # inputs keyed by the input tensor names (e.g. 'token_ids').
                # Rather than rely on `input_names` length matches, prefer
                # reconstructing a dict using the layer.symbolic `inputs`
                # objects which give precise names we cached earlier.
                if isinstance(layer, (Functional, Model)) and getattr(layer, "inputs", None):
                    mapped_inputs = {}
                    for sym in layer.inputs:
                        try:
                            sym_id = id(sym)
                            name = getattr(sym, "name", None)
                            clean_name = name.split(":")[0] if name else None
                        except Exception:
                            sym_id = None
                            name = None
                            clean_name = None

                        val = None
                        if sym_id is not None and sym_id in tensor_cache:
                            val = tensor_cache[sym_id]
                        elif clean_name and clean_name in tensor_cache:
                            val = tensor_cache[clean_name]
                        elif name and name in tensor_cache:
                            val = tensor_cache[name]

                        if val is not None:
                            # Use the cleaned name as the key when possible
                            key = clean_name or name
                            mapped_inputs[key] = val

                    # If we recovered at least one named input, prefer the dict form.
                    # If we recovered all expected inputs, definitely use it.
                    if mapped_inputs:
                        # If we got all expected inputs, use mapped_inputs directly.
                        if len(mapped_inputs) == len(layer.inputs):
                            layer_inputs = mapped_inputs
                        else:
                            # Partial mapping: fall back to previous positional list
                            # unless the layer clearly expects named kwargs (input_names).
                            if getattr(layer, "input_names", None):
                                # try to align by input_names where available
                                cleaned_names = [n.split(":")[0] for n in layer.input_names]
                                if all(n in mapped_inputs for n in cleaned_names):
                                    layer_inputs = {n: mapped_inputs[n] for n in cleaned_names}

                # Preserve previous single-element behavior
                if isinstance(layer_inputs, (list, tuple)) and len(layer_inputs) == 1:
                    layer_inputs = layer_inputs[0]

                try:
                    # Defensive: if we failed to reconstruct any inputs for this
                    # layer, forward the top-level `inputs` so layers don't
                    # receive an empty list (which causes errors like 'list'
                    # has no dtype). Use DEBUG level to avoid noisy INFO logs
                    # during normal runs.
                    # Only treat as "empty" if we have a container with
                    # zero length. Avoid using truthiness on tensors/arrays
                    # (which raises ambiguous truth-value errors for JAX
                    # arrays). If the reconstructed inputs are None or an
                    # empty list/tuple/dict, forward the top-level inputs.
                    is_empty_container = (
                        layer_inputs is None
                        or (isinstance(layer_inputs, (list, tuple)) and len(layer_inputs) == 0)
                        or (isinstance(layer_inputs, dict) and len(layer_inputs) == 0)
                    )

                    if is_empty_container:
                        logger.debug(
                            "Layer '%s' received empty reconstructed inputs; forwarding top-level inputs",
                            layer.name,
                        )
                        layer_inputs = inputs

                    # Forward any recorded keyword arguments from the original
                    # symbolic node (e.g. `reverse=True`) so that layers that
                    # rely on call-time kwargs receive them during runtime
                    # reconstruction. Node keeps these in `node.arguments.kwargs`.
                    node_kwargs = {}
                    try:
                        if hasattr(node, "arguments") and getattr(node, "arguments") is not None:
                            node_kwargs = getattr(node.arguments, "kwargs", {}) or {}
                    except Exception:
                        node_kwargs = {}

                    # Pass through recorded node kwargs to the layer call.
                    call_kwargs = {"training": training} if training is not None else {}
                    # Merge node kwargs (do not overwrite explicit training)
                    for k, v in node_kwargs.items():
                        if k != "training":
                            call_kwargs[k] = v

                    current_tensor = layer(layer_inputs, **call_kwargs)
                except Exception:
                    # Try a sequence of conservative fallbacks to adapt
                    # the inputs to what the layer implementation expects.
                    # 1) If we have a list/tuple, try building a dict by
                    #    matching declared input names.
                    # 2) If we have a dict, try passing a single tensor
                    #    (first value) in case the layer expects a single
                    #    positional tensor.
                    # 3) If we have a list/tuple, try passing the first
                    #    element as a single positional tensor.

                    tried_call = False

                    # (1) list/tuple -> dict by names
                    if isinstance(layer_inputs, (list, tuple)):
                        cleaned_names = None
                        try:
                            if hasattr(layer, "inputs") and layer.inputs:
                                cleaned_names = [n.name.split(":")[0] for n in layer.inputs]
                        except Exception:
                            cleaned_names = None

                        if not cleaned_names and hasattr(layer, "input_names"):
                            try:
                                cleaned_names = [n.split(":")[0] for n in layer.input_names]
                            except Exception:
                                cleaned_names = None

                        if cleaned_names and len(cleaned_names) == len(layer_inputs):
                            alt_inputs = dict(zip(cleaned_names, layer_inputs))
                            try:
                                logger.debug(
                                    "Retrying layer '%s' with dict inputs: %s",
                                    layer.name,
                                    list(alt_inputs.keys()),
                                )
                                current_tensor = layer(alt_inputs, training=training)
                                tried_call = True
                            except Exception:
                                tried_call = False

                    # (2) dict -> single tensor (first value)
                    if not tried_call and isinstance(layer_inputs, dict):
                        first_val = next(iter(layer_inputs.values()), None)
                        if first_val is not None:
                            try:
                                logger.debug(
                                    "Retrying layer '%s' with single tensor from dict inputs",
                                    layer.name,
                                )
                                current_tensor = layer(first_val, training=training)
                                tried_call = True
                            except Exception:
                                tried_call = False

                    # (3) list/tuple -> single positional tensor (first element)
                    if not tried_call and isinstance(layer_inputs, (list, tuple)) and len(layer_inputs) > 0:
                        try:
                            logger.debug(
                                "Retrying layer '%s' with first element of positional inputs",
                                layer.name,
                            )
                            current_tensor = layer(layer_inputs[0], training=training)
                            tried_call = True
                        except Exception:
                            tried_call = False

                    if not tried_call:
                        # Nothing worked — re-raise to preserve original traceback
                        raise

                # --- NEW DEBUG: immediate post-call dump for token embeddings ---
                # Dump the raw return immediately after the layer call so we can
                # see exactly what the layer returned (type/repr/shape) before any
                # communication or mapping logic runs.
                if "token_embedding" in layer.name or "embeddings" in layer.name:
                    try:
                        rep = repr(current_tensor)
                        rep_snip = rep[:1000] + ("...<truncated>" if len(rep) > 1000 else "")
                        shp = getattr(current_tensor, "shape", None)
                    except Exception as _e:
                        pass

                # 4/5. Apply Communication Rules (AllReduce/Gather) and Cache Output
                layer_path = layer.path
                output_rule = None
                for pattern, rule in self.config.output_rules.items():
                    if re.search(pattern, layer_path):
                        output_rule = rule.get(0)
                        break

                if output_rule:
                    # Debug: if this layer is a token embedding or related,
                    # print the tensor before communication so we can see
                    # whether gather/all_gather is changing its shape.
                    if "token_embedding" in layer.name or "embeddings" in layer.name:
                        try:
                            pre_shp = getattr(current_tensor, "shape", None)
                        except Exception:
                            pre_shp = None

                    current_tensor = self._apply_communication(
                        current_tensor, layer.name, output_rule
                    )

                    if "token_embedding" in layer.name or "embeddings" in layer.name:
                        try:
                            post_shp = getattr(current_tensor, "shape", None)
                        except Exception:
                            post_shp = None

                # Map the runtime outputs back to the symbolic output tensors
                # for every inbound node. Store by id and by several name
                # variants (full name, cleaned name, base name) to be tolerant
                # to id/name mismatches.
                for node_idx, node in enumerate(layer._inbound_nodes):
                    try:
                        outputs = getattr(node, "output_tensors", None)
                    except Exception:
                        outputs = None

                    if not outputs:
                        continue

                    if isinstance(current_tensor, (list, tuple)):
                        for out_idx, (sym, val) in enumerate(zip(outputs, current_tensor)):
                            # Primary mapping by id(symbolic_tensor)
                            tensor_cache[id(sym)] = val
                            # Precise mapping by (layer name, node index, tensor index)
                            try:
                                layer_key = ("layer_node_output", layer.name, node_idx, out_idx)
                                tensor_cache[layer_key] = val
                            except Exception:
                                pass
                            # Best-effort name mappings for debugging only (may collide)
                            try:
                                name = getattr(sym, "name", None)
                                if name:
                                    clean_name = name.split(":")[0]
                                    tensor_cache[clean_name] = val
                                    tensor_cache[name] = val
                            except Exception:
                                pass
                    else:
                        for out_idx, sym in enumerate(outputs):
                            tensor_cache[id(sym)] = current_tensor
                            try:
                                layer_key = ("layer_node_output", layer.name, node_idx, out_idx)
                                tensor_cache[layer_key] = current_tensor
                            except Exception:
                                pass
                            try:
                                name = getattr(sym, "name", None)
                                if name:
                                    clean_name = name.split(":")[0]
                                    tensor_cache[clean_name] = current_tensor
                                    tensor_cache[name] = current_tensor
                            except Exception:
                                pass

            # Return final outputs
            # Print declared output shape/spec for debugging
            try:
                declared_shape = None
                try:
                    declared_shape = self.original_model.compute_output_shape(
                        [inp.shape for inp in self.original_model.inputs]
                    )
                except Exception:
                    try:
                        declared_shape = self.original_model.compute_output_shape(
                            self.original_model.inputs[0].shape
                        )
                    except Exception:
                        declared_shape = None
            except Exception:
                pass

            # Print Keras history for each symbolic output to locate producer layer
            try:
                for symbolic_output in self.original_model.outputs:
                    try:
                        history = getattr(symbolic_output, "_keras_history", None)
                        if history:
                            layer_obj = history[0]
                            layer_name = getattr(layer_obj, "name", str(layer_obj))
                            # Print detailed info about the producing layer class/module/file
                            try:
                                cls = layer_obj.__class__
                                mod = getattr(cls, "__module__", "<no module>")
                                cls_name = getattr(cls, "__name__", str(cls))
                                src_file = inspect.getsourcefile(cls) or "<source file not found>"
                            except Exception:
                                pass
                        else:
                            layer_name = "<no keras history>"
                    except Exception:
                        layer_name = "<history lookup failed>"
            except Exception:
                pass

            final_outputs = []
            for symbolic_output in self.original_model.outputs:
                # Robust lookup: prefer id, then cleaned/full name variants.
                val = None
                try:
                    val = tensor_cache.get(id(symbolic_output), None)
                except Exception:
                    val = None

                if val is None:
                    # Try by name variants
                    try:
                        name = getattr(symbolic_output, "name", None)
                        if name:
                            clean_name = name.split(":")[0]
                            val = tensor_cache.get(clean_name, tensor_cache.get(name, None))
                    except Exception:
                        val = None

                # If still not found, try precise lookup via _keras_history
                if val is None:
                    try:
                        history = getattr(symbolic_output, "_keras_history", None)
                        if history and len(history) >= 3:
                            producing_layer_obj = history[0]
                            node_index = history[1]
                            tensor_index = history[2]
                            layer_key = ("layer_node_output", getattr(producing_layer_obj, "name", None), node_index, tensor_index)
                            val = tensor_cache.get(layer_key, None)
                    except Exception:
                        val = None

                if val is None:
                    # As a last resort, try base name
                    try:
                        name = getattr(symbolic_output, "name", None)
                        if name:
                            base_name = name.split(":")[0].split("/")[-1]
                            val = tensor_cache.get(base_name, None)
                    except Exception:
                        val = None

                if val is None:
                    # Nothing found — raise with helpful context
                    raise RuntimeError(
                        f"Missing runtime value for model output symbolic tensor: {symbolic_output}."
                        " Available tensor_cache keys: " + ",".join([str(k) for k in list(tensor_cache.keys())[:20]])
                    )

                # Print shape/type for debugging unexpected output shapes
                try:
                    shp = getattr(val, "shape", None)
                except Exception:
                    shp = None
                name = getattr(symbolic_output, 'name', symbolic_output)
                # Deep introspection for nested structures and abnormal ranks
                

                final_outputs.append(val)

            if len(final_outputs) == 1:
                return final_outputs[0]
            return final_outputs

        def _apply_communication(self, sharded_output, layer_name, rule_str: str):
            """Applies communication directly using the distributed backend."""
            if "sum" in rule_str or "allreduce" in rule_str:
                logger.debug(f"Applying AllReduce (sum) to {layer_name}")
                return distribution_lib.all_reduce(
                    sharded_output, op="sum", axis_name="model"
                )

            elif "gather" in rule_str:
                try:
                    parts = rule_str.split(" ")
                    if len(parts) > 1:
                        dim = int(parts[-1])
                    else:
                        dim = -1
                except (ValueError, IndexError):
                    dim = -1
                
                logger.debug(f"Applying AllGather (dim={dim}) to {layer_name}")
                return distribution_lib.all_gather(
                    sharded_output, axis=dim, axis_name="model"
                )

            else:
                return sharded_output

        def get_config(self):
            return self.original_model.get_config()

        @classmethod
        def from_config(cls, config, custom_objects=None):
            return cls(**config)

    return ParameterShardedModel


def make_parameter_sharded_model(
    module: "Model",
    config: LayoutMap,
    rank: int,
    device_count: int,
    device_id: Any,
) -> Tuple["Model", Set[str]]:
    """
    Create a parameter-sharded version of a Keras model.
    """
    sharding_strategy = ParameterShardingStrategy(device_count, rank)

    sharded_model, modified_parameters = sharding_strategy.shard_model_parameters(
        module, config, device_id
    )

    return sharded_model, modified_parameters