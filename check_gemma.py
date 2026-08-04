import os

os.environ["KERAS_BACKEND"] = "torch"
import keras_hub

try:
    model = keras_hub.models.GemmaBackbone.from_preset("gemma_2b_en")
    for v in model.weights:
        print(v.path)
except Exception as e:
    print(e)
