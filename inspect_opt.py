import os
import sys
import subprocess

def inspect():
    import keras
    import keras_hub
    model = keras_hub.models.OPTBackbone.from_preset("opt_125m_en")
    
    found = False
    for layer in model.layers:
        if "layer_normalization" in layer.name:
            print(f"{os.environ['KERAS_BACKEND']} {layer.name} epsilon: {layer.epsilon}")
            found = True
        if hasattr(layer, "layers"):
            for sub in layer.layers:
                if "layer_normalization" in sub.name:
                    print(f"{os.environ['KERAS_BACKEND']} {sub.name} epsilon: {sub.epsilon}")
                    found = True
    if not found:
        # Check transformer decoder layers specifically
        for layer in model.layers:
            if hasattr(layer, "_self_attention_layer"):
                # TransformerDecoder
                print(f"{os.environ['KERAS_BACKEND']} {layer.name} SA LN epsilon: {layer._self_attention_layer_norm.epsilon}")
                print(f"{os.environ['KERAS_BACKEND']} {layer.name} FF LN epsilon: {layer._feedforward_layer_norm.epsilon}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        os.environ["KERAS_BACKEND"] = sys.argv[1]
        inspect()
    else:
        subprocess.run([sys.executable, sys.argv[0], "jax"])
        subprocess.run([sys.executable, sys.argv[0], "torch"])