"""
This file implements the logic for sharding the model's parameters.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

# Assuming these imports work for the JAX backend
from keras.src.backend import distribution_lib
# --- MODIFIED: Import SplitRule ---
from keras.src.distribution.tensor_parallel.tensor_layout import LayoutMap, SplitRule
# NOTE: The refactored split function is inside the tensor_layout module,
#       but we will access it via the 'action' callable passed in the config.

from keras.src import ops
from keras.src.backend import core
from keras.src import layers # Ensure layers is directly imported for helper functions
from keras import Variable, device # Ensure Variable and device are available

logger = logging.getLogger(__name__)


class ShardedWeight:
    """
    A wrapper for a sharded Keras Variable, providing a consistent interface.
    """

    def __init__(self, tensor_shard, name, trainable=True, device_id=None):
        
        # 🌟 FIX: Use the device_id passed to the constructor for Keras Variable placement
        current_dev_name = device_id if device_id else "UNKNOWN_DEVICE"
        print(
            f"   [DEV: {current_dev_name}] 🧬 Creating Sharded Variable "
            f"'{name}' with shape {tensor_shard.shape}"
        )

        with device(current_dev_name):
            self._variable = Variable(
                # --- MODIFIED: Use convert_to_numpy for cleaner device transfer ---
                # This ensures the initializer is a simple array,
                # preventing potential graph issues during init.
                initializer=ops.convert_to_numpy(tensor_shard),
                trainable=trainable, 
                name=name
            )
        self.regularizer = None # Note: This discards original regularizers

    @property
    def name(self):
        """Returns the name of the underlying variable."""
        return self._variable.name

    @property
    def trainable(self):
        """Returns whether the variable is trainable."""
        return self._variable.trainable

    @property
    def shape(self):
        """Returns the shape of the variable."""
        return self._variable.shape

    @property
    def dtype(self):
        """Returns the dtype of the underlying variable."""
        return self._variable.dtype

    @property
    def variable(self):
        """Provides direct access to the underlying tf.Variable."""
        return self._variable

    @property
    def value(self):
        return self._variable.value

    def numpy(self):
        """Returns the value of the variable as a NumPy array."""
        return self.variable.numpy()

    def num_elements(self):
        """Returns the total number of elements in the tensor."""
        return ops.size(self._variable)

    def __repr__(self):
        return (
            f"<ShardedWeight name='{self.name}' "
            f"shape={self.shape} trainable={self.trainable}>"
        )


class ParameterShardingStrategy:
    """
    Parameter-level sharding strategy that works with any Keras model.
    Instead of rebuilding the model, we shard only the weights and handle
    communication during forward/backward passes.
    """

    def __init__(self, device_count: int, rank: int):
        self.device_count = device_count
        self.rank = rank
        self.sharded_weights = {}
        self.original_weights = {}
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

        self._store_original_weights(model)
        modified_parameters = set()

        # --- MODIFIED: Check if the action is a SplitRule instance ---
        for pattern, action in config.state_rules.items():
            if isinstance(action, SplitRule): # Check for SplitRule
                matching_params = self._find_matching_parameters(model, pattern)

                for param_name, param in matching_params:
                    try:
                        param_id = id(param.experimental_ref())
                    except AttributeError:
                        param_id = id(param)

                    if param_id in self.sharded_weights_by_id:
                        # Weight tying logic for shared parameters
                        self.sharded_weights[param_name] = self.sharded_weights_by_id[
                            param_id
                        ]

                        existing_param_name = "unknown"
                        for name, shard in self.sharded_weights.items():
                            if shard is self.sharded_weights_by_id[param_id]:
                                existing_param_name = name
                                break
                        
                        # --- MODIFIED: Store the sharding dim ---
                        self.weight_mapping[param_name] = self.weight_mapping[
                            existing_param_name
                        ]
                        modified_parameters.add(param_name)
                        print(
                            f"   🔗 Tied {param_name} to existing shard from {existing_param_name}"
                        )
                        continue

                    # The 'action' is our SplitRule object, which is callable
                    sharded_tensor = action(param, self.rank)

                    self.sharded_weights[param_name] = sharded_tensor
                    self.sharded_weights_by_id[param_id] = sharded_tensor

                    # --- MODIFIED: Store the sharding dim from the action ---
                    self.weight_mapping[param_name] = {
                        "original_shape": param.shape,
                        "sharded_shape": sharded_tensor.shape,
                        "action": action,
                        "dim": action.dim,  # <-- CRITICAL FOR CHECKPOINTING
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

    def _store_original_weights(self, model: "Model"):
        """Store original weights for reference."""
        # from keras.src import layers # Already imported globally
        def find_weights_recursive(
            current_layer: layers.Layer, prefix: str = ""
        ):
            name = current_layer.name
            full_name = f"{prefix}.{name}" if prefix else name

            if hasattr(current_layer, "weights") and current_layer.weights:
                for weight in current_layer.weights:
                    # Logic to get the cleaned weight name (e.g., 'kernel' or 'bias')
                    cleaned_name = weight.name.split("/")[-1].split(":")[0]
                    param_name = f"{full_name}.{cleaned_name}"
                    # Only store if it's not a symbolic tensor/placeholder, which .numpy() helps confirm
                    if hasattr(weight, 'numpy'):
                         # --- MODIFIED: Store weights as numpy ---
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

        # Start recursion from the base model
        find_weights_recursive(model, prefix="")

    def _find_matching_parameters(
        self, model: "Model", pattern: str
    ) -> List[Tuple[str, Any]]:
        """
        Find parameters that match the given pattern using smart recursion.
        """
        # from keras.src import layers # Already imported globally

        matching_params = []
        processed_layers = set()

        def search_layer_recursive(
            current_layer: layers.Layer, prefix: str = ""
        ):
            if id(current_layer) in processed_layers:
                return
            processed_layers.add(id(current_layer))

            name = current_layer.name
            full_name = f"{prefix}.{name}" if prefix else name

            if hasattr(current_layer, "weights") and current_layer.weights:
                for weight in current_layer.weights:
                    cleaned_weight_name = weight.name.split("/")[-1].split(":")[
                        0
                    ]
                    param_name = f"{full_name}.{cleaned_weight_name}"

                    # 🌟 FIX: Use re.fullmatch for exact match of the pattern to the parameter path
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
    from keras.src.models import Model # Imported here to prevent circular dependency

    class ParameterShardedModel(Model):
        """
        Wrapper model that handles parameter sharding without rebuilding the structure.
        This preserves the original model's functionality while enabling tensor parallelism.
        """

        def __init__(
            self,
            original_model: Model,
            sharding_strategy: ParameterShardingStrategy,
            config: LayoutMap,
            device_id: Any,
        ):
            # Pass original_model properties to the base Model init
            super().__init__(name=original_model.name) 

            self.original_model = original_model
            self.sharding_strategy = sharding_strategy
            self.config = config
            self._device = device_id
            
            # --- MODIFIED: Cache for ShardedWeight objects ---
            self._sharded_weight_cache: Dict[str, ShardedWeight] = {}

            self._build_and_cache_weights()

            # The original model must be built before accessing its inputs
            if not self.original_model.built:
                 try:
                     # Try to build with a single input
                     input_shape = self.original_model.inputs[0].shape
                     self.original_model.build(input_shape)
                     self.build(input_shape)
                 except (AttributeError, IndexError, TypeError, ValueError):
                     logger.warning(
                         f"Could not auto-build original_model {self.original_model.name}. "
                         "Assuming it was built manually."
                     )
                 
            if self.original_model.inputs and not self.built:
                self.build(self.original_model.inputs[0].shape)

            print("🚀 ParameterShardedModel created successfully")

        @property
        def device(self):
            return self._device

        def _build_and_cache_weights(self):
            """
            Builds the list of trainable/non-trainable weights ONCE and caches it.
            """
            print("   - Building and caching the definitive weights list...")
            logger.debug("--- WEIGHT CACHE BUILDER ---")
            weights_list = []

            sharded_weight_ids = set(
                self.sharding_strategy.sharded_weights_by_id.keys()
            )

            for (
                param_name,
                sharded_tensor,
            ) in self.sharding_strategy.sharded_weights.items():
                # --- MODIFIED: Create and cache ShardedWeight objects ---
                sharded_weight_obj = ShardedWeight(
                    sharded_tensor, 
                    param_name, 
                    device_id=self._device  
                )
                weights_list.append(sharded_weight_obj)
                self._sharded_weight_cache[param_name] = sharded_weight_obj

            logger.debug(f"Added {len(weights_list)} sharded weights.")

            unsharded_count = 0
            for weight in self.original_model.weights:
                try:
                    weight_id = id(weight.experimental_ref())
                except AttributeError:
                    weight_id = id(weight)

                if weight_id not in sharded_weight_ids:
                    weights_list.append(weight)
                    unsharded_count += 1

            logger.debug(f"Added {unsharded_count} replicated weights.")

            self._weights_list = weights_list
            logger.debug("--- WEIGHT CACHE BUILD COMPLETE ---")
            logger.debug(f"Total weights in list: {len(self._weights_list)}")

        @property
        def weights(self):
            """
            Override weights property to return the cached list of sharded weights.
            """
            return self._weights_list

        def call(self, inputs, training=None, mask=None):
            # --- DEBUGGER ---
            print("\n[DEBUGGER] --- TRACE: ParameterShardedModel.call EXECUTED ---")
            
            # from keras.src import layers # Already imported globally
            
            tensor_cache = {}
            print(f"[DEBUG] New call to ParameterShardedModel.call. Inputs: {inputs}")

            # Populate cache with initial inputs
            if isinstance(inputs, dict):
                model_input_names = [inp.name.split(":")[0] for inp in self.original_model.inputs]
                for name in model_input_names:
                    if name in inputs:
                        symbolic_inp = next(inp for inp in self.original_model.inputs if inp.name.split(":")[0] == name)
                        tensor_cache[symbolic_inp.name] = inputs[name]
            elif isinstance(inputs, (list, tuple)):
                for i, inp_tensor in enumerate(self.original_model.inputs):
                     tensor_cache[inp_tensor.name] = inputs[i]
            else:
                tensor_cache[self.original_model.inputs[0].name] = inputs

            print(f"[DEBUG] Initial tensor_cache keys: {list(tensor_cache.keys())}")

            # Iterate through the original model's layers
            for layer in self.original_model.layers:
                if isinstance(layer, layers.InputLayer):
                    continue
                
                # ==========================================================
                # --- START: 🌟 CRITICAL FIX 🌟 ---
                # Rebuild inputs in the *exact* structure the layer expects
                # (dict, list, or single tensor)
                # ==========================================================
                
                if not layer._inbound_nodes:
                    layer_call_inputs = []
                else:
                    node = layer._inbound_nodes[0] 
                    
                    # Check how the layer was *originally* called (in the functional model)
                    original_call_inputs = node.call_args[0] if node.call_args else None
                    
                    if isinstance(original_call_inputs, dict):
                        # --- This layer expects a DICTIONARY ---
                        print(f"[DEBUG] Layer {layer.name} expects a DICT. Rebuilding dict...")
                        layer_call_inputs = {}
                        for key, symbolic_tensor in original_call_inputs.items():
                            if symbolic_tensor.name not in tensor_cache:
                                error_msg = (
                                    f"FATAL CACHE MISS (dict): Cannot find "
                                    f"'{symbolic_tensor.name}' for layer '{layer.name}'"
                                )
                                print(error_msg)
                                raise KeyError(error_msg)
                            layer_call_inputs[key] = tensor_cache[symbolic_tensor.name]

                    else:
                        # --- This layer expects a LIST or a single TENSOR ---
                        print(f"[DEBUG] Layer {layer.name} expects a LIST/TENSOR. Rebuilding...")
                        symbolic_inputs = node.input_tensors
                        if not isinstance(symbolic_inputs, (list, tuple)):
                            symbolic_inputs = [symbolic_inputs]
                        
                        layer_call_inputs = []
                        for symbolic_tensor in symbolic_inputs:
                            if symbolic_tensor.name not in tensor_cache:
                                error_msg = (
                                    f"FATAL CACHE MISS (list): Cannot find "
                                    f"'{symbolic_tensor.name}' for layer '{layer.name}'"
                                )
                                print(error_msg)
                                raise KeyError(error_msg)
                            layer_call_inputs.append(tensor_cache[symbolic_tensor.name])
                        
                        if len(layer_call_inputs) == 1:
                            layer_call_inputs = layer_call_inputs[0]

                # ==========================================================
                # --- END: CRITICAL FIX ---
                # ==========================================================

                # Run the layer's forward pass
                # print(f"[DEBUG] Calling layer: {layer.name} with inputs: {layer_call_inputs}")
                current_tensor = layer(layer_call_inputs, training=training)

                # Update cache with the layer's output
                layer_outputs = current_tensor
                if not isinstance(layer_outputs, (list, tuple)):
                    symbolic_outputs = [layer.output]
                    layer_outputs = [layer_outputs]
                else:
                    symbolic_outputs = layer.output
                
                for symbolic_out, concrete_out in zip(symbolic_outputs, layer_outputs):
                    tensor_cache[symbolic_out.name] = concrete_out
                
                # Check for output rule to apply collective communication
                layer_path = layer.path
                output_rule = None
                
                for pattern, rule in self.config.output_rules.items():
                    if re.search(pattern, layer_path):
                        output_rule = rule.get(0)
                        break
                
                if output_rule:
                    first_output_tensor = layer_outputs[0]
                    modified_tensor = self._apply_communication(
                        first_output_tensor, layer.name, output_rule
                    )
                    tensor_cache[symbolic_outputs[0].name] = modified_tensor

            # Return the final model output(s)
            final_outputs = []
            
            print("\n[DEBUG] --- Final Output Lookup ---")
            print(f"[DEBUG] Available keys in tensor_cache: {list(tensor_cache.keys())}")
            print(f"[DEBUG] Original model outputs to find: {[out.name for out in self.original_model.outputs]}")

            for symbolic_output in self.original_model.outputs:
                if symbolic_output.name not in tensor_cache:
                    error_msg = (
                        f"[DEBUG] FATAL: Final output tensor {symbolic_output.name} not in tensor_cache!"
                    )
                    print(error_msg)
                    raise KeyError(error_msg)
                final_outputs.append(tensor_cache[symbolic_output.name])

            # --- DEBUGGER ---
            print(f"[DEBUGGER] --- TRACE: ParameterShardedModel.call RETURNING ---")
            print(f"[DEBUGGER] Final outputs list: {final_outputs}")
            print(f"[DEBUGGER] Number of final outputs: {len(final_outputs)}")

            if len(final_outputs) == 1:
                return final_outputs[0]
            return final_outputs
        
        def _apply_communication(self, sharded_output, layer_name, rule_str: str):
            """Applies communication directly using the distributed backend."""

            if "sum" in rule_str or "allreduce" in rule_str:
                # Row-Parallel Forward pass requires AllReduce/summing of partial results
                logger.debug(
                    f"Applying Row-Parallel Forward (AllReduce) to {layer_name}"
                )
                # Call all_reduce directly
                return core.all_reduce(
                    sharded_output, op="sum", axis_name="model"
                )

            elif "gather" in rule_str:
                # Column-Parallel Forward pass requires AllGather/concatenation of partial results
                try:
                    # Extract dimension from the rule string, e.g., "gather -1"
                    # The rule should contain the dim, e.g., output_rules[f"{full_name}"] = {0: "gather -1"}
                    parts = rule_str.split(" ")
                    if len(parts) > 1:
                        dim = int(parts[-1])
                    else:
                        dim = -1
                except (ValueError, IndexError):
                    dim = -1
                logger.debug(
                    f"Applying Column-Parallel Forward (AllGather dim={dim}) to {layer_name}"
                )
                # Call all_gather directly
                return distribution_lib.all_gather(
                    sharded_output, axis=dim, axis_name="model"
                )

            else:
                return sharded_output

        def get_config(self):
            """Get model configuration."""
            return self.original_model.get_config()

        @classmethod
        def from_config(cls, config, custom_objects=None):
            """Create model from config."""
            # NOTE: This implementation is preserved from the original code.
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