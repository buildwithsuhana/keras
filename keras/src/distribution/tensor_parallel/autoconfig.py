import logging
from typing import Sequence, Set
import re
import keras
from keras.src.distribution.tensor_parallel.config import ConfigKeras
from keras.src.distribution.tensor_parallel.state_action_keras import SplitKeras

# --- 1. Define logger at the top ---
logger = logging.getLogger(__name__)

# --- 2. Correctly import KerasNLP layers ---
try:
    from keras_nlp import layers as knlp_layers
    from keras_nlp import models as knlp_models
    
    KERASNLP_BACKBONES = (knlp_models.GemmaBackbone, knlp_models.GPT2Backbone, knlp_models.OPTBackbone)
    
    # A tuple of all MHA types we want to check
    KERASNLP_MHA_TUPLE = (knlp_layers.CachedMultiHeadAttention)
    
    # --- FIX: We also need to explicitly check for PositionEmbedding ---
    KERASNLP_EMBEDDINGS = (knlp_layers.PositionEmbedding, )

    logger.info("Successfully imported KerasNLP layers for sharding.")

except ImportError:
    KERASNLP_BACKBONES = ()
    KERASNLP_MHA_TUPLE = () # Empty tuple
    KERASNLP_EMBEDDINGS = () # Empty tuple # <-- FIX
    logger.warning("KerasNLP not found. Sharding will only apply to stock Keras layers.")
# --- END IMPORTS ---


def analyze_dense_layer_directly(layer, module, prefix: str) -> str:
    """
    Analyzes a Dense layer by its kernel shape to determine if it's an
    up-projection, down-projection, or generic.
    """
    from keras import layers, Model
    if not isinstance(layer, layers.Dense) or not hasattr(layer, 'kernel'):
        return 'generic_dense'

    try:
        # Kernel shape is always (input_dim, output_dim)
        kernel_shape = layer.kernel.shape
        if len(kernel_shape) != 2:
            return 'generic_dense'
            
        input_dim = kernel_shape[0]
        output_dim = kernel_shape[1]
        
    except Exception:
        # Layer might not be built, fallback to units
        if hasattr(layer, 'units'):
            output_dim = layer.units
        else:
            return 'generic_dense' # Cannot determine
        
        # Fallback for input_dim (less reliable, but better than nothing)
        if hasattr(layer, 'input_shape') and layer.input_shape and len(layer.input_shape) > 1:
             input_dim = layer.input_shape[-1]
        else:
             return 'generic_dense' # Cannot determine

    if not input_dim or not output_dim:
        return 'generic_dense'

    # Use a 1.5x threshold to classify
    expansion_threshold = 1.5
    is_expansion = output_dim > input_dim * expansion_threshold
    is_contraction = input_dim > output_dim * expansion_threshold

    if is_expansion:
        return 'up_projection'
    elif is_contraction:
        return 'down_projection'
    else:
        return 'generic_dense'


def get_default_config_keras(module, device_ids: Sequence[str]) -> ConfigKeras:
    """
    Generates a smart, recursive sharding configuration for a Keras model.
    """
    from keras import layers, Model
    world_size = len(device_ids)
    state_rules = {}
    output_rules = {}
    processed_layers = set()

    processed_weights_ids = set()

    def _find_and_shard_layers(current_layer: layers.Layer, prefix: str = ""):
        """
        Recursively find and apply sharding rules to all nested layers.
        """
        
        if id(current_layer) in processed_layers:
            return
        processed_layers.add(id(current_layer))

        name = current_layer.name
        full_name = f"{prefix}.{name}" if prefix else name
        
        # --- We have deleted the MHA block ---
        # We now shard the Dense/EinsumDense layers *inside* it

        if isinstance(current_layer, layers.Dense):
            # This handles all Dense layers, including MLPs
            mlp_type = analyze_dense_layer_directly(current_layer, module, full_name)
            
            if mlp_type == 'up_projection':
                state_rules[f"^{full_name}.kernel$"] = SplitKeras(world_size, 1, "column")
                if current_layer.use_bias:
                    state_rules[f"^{full_name}.bias$"] = SplitKeras(world_size, 0, "column")
                output_rules[f"^{full_name}$"] = {0: "gather"}
                logger.info(f"Applied Column-wise sharding to MLP up-projection {full_name}")
            
            elif mlp_type == 'down_projection':
                state_rules[f"^{full_name}.kernel$"] = SplitKeras(world_size, 0, "row")
                output_rules[f"^{full_name}$"] = {0: "allreduce"}
                logger.info(f"Applied Row-wise sharding to MLP down-projection {full_name}")
            
            else: # Generic Dense
                state_rules[f"^{full_name}.kernel$"] = SplitKeras(world_size, 1, "column")
                if current_layer.use_bias:
                    state_rules[f"^{full_name}.bias$"] = SplitKeras(world_size, 0, "column") 
                output_rules[f"^{full_name}$"] = {0: "gather -1"}
                logger.info(f"Applied Generic Column-wise sharding to {full_name}")
                
            return # This is a leaf layer, stop recursion.

        elif isinstance(current_layer, layers.EinsumDense):
            # --- FIX 1: Make EinsumDense logic smart ---
            # It needs to know the difference between Column (query/key/value)
            # and Row (attention_output) sharding.
            
            if "attention_output" in full_name:
                # This is Row-Parallel
                state_rules[f"^{full_name}.kernel$"] = SplitKeras(world_size, 0, "row")
                if hasattr(current_layer, 'bias') and current_layer.bias is not None:
                    # Bias is replicated, no rule needed
                    pass
                output_rules[f"^{full_name}$"] = {0: "allreduce"}
                logger.info(f"Applied EinsumDense Row-wise sharding to {full_name}")
            else:
                # This is Column-Parallel (query, key, value)
                state_rules[f"^{full_name}.kernel$"] = SplitKeras(world_size, 1, "column")
                if hasattr(current_layer, 'bias') and current_layer.bias is not None:
                    state_rules[f"^{full_name}.bias$"] = SplitKeras(world_size, 0, "column")
                output_rules[f"^{full_name}$"] = {0: "gather -1"}
                logger.info(f"Applied EinsumDense Column-wise sharding to {full_name}")
            # --- END FIX 1 ---
            return 

        # --- FIX 2: Correct Embedding Logic ---
        elif isinstance(current_layer, (layers.Embedding,) + KERASNLP_EMBEDDINGS):
            # Check if this is a *container* (like TokenAndPositionEmbedding)
            if hasattr(current_layer, 'token_embedding') or hasattr(current_layer, 'position_embedding'):
                # If it's a container, DON'T shard it.
                # Let the recursion continue to find the *real* layers inside.
                logger.debug(f"{full_name} is an Embedding container, recursing...")
            
            else:
                # This is a *simple* Embedding layer. Find its weight and shard it.
                weight_name = None
                if hasattr(current_layer, 'embeddings'):
                    weight_name = 'embeddings'  # For TokenEmbedding
                elif hasattr(current_layer, 'position_embeddings'):
                    weight_name = 'position_embeddings' # For PositionEmbedding
                
                if weight_name:
                    # --- FINAL FIX: Make the regex rule more robust ---
                    state_rules[f"^{full_name}\..*{weight_name}$"] = SplitKeras(world_size, 1, "column")
                    output_rules[f"^{full_name}$"] = {0: "no_comm"} 
                    logger.info(f"Applied Embedding sharding to {full_name} (target: {weight_name})")
                else:
                    logger.warning(f"Could not find a shardable weight for Embedding layer {full_name}")
                
                return # This is a true leaf layer, stop recursion.
        # --- END FIX 2 ---

        elif isinstance(current_layer, (layers.LayerNormalization, layers.BatchNormalization, layers.GroupNormalization)):
            return # Stop recursion

        # --- RECURSIVE STEP ---
        
        # --- FIX: This recursion logic was buggy (if/else) ---
        # --- This new version (no 'else') correctly finds all layers ---
        
        # 1. Recurse into .layers if it exists
        if hasattr(current_layer, 'layers') and current_layer.layers:
            for sub_layer in current_layer.layers:
                _find_and_shard_layers(sub_layer, full_name)
        
        # 2. ALSO recurse into attributes (like .token_embedding)
        for attr_name in dir(current_layer):
            if attr_name.startswith('__') and attr_name.endswith('__'):
                continue 
            try:
                attr = getattr(current_layer, attr_name)
            except Exception:
                continue 

            if isinstance(attr, layers.Layer) and attr is not current_layer:
                _find_and_shard_layers(attr, full_name)
            elif isinstance(attr, (list, tuple)):
                for item in attr:
                    if isinstance(item, layers.Layer):
                        _find_and_shard_layers(item, full_name)

    _find_and_shard_layers(module, prefix="") 
    
    return ConfigKeras(
        state_rules=state_rules,
        output_rules=output_rules
    )