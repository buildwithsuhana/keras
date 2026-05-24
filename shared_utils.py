import numpy as np
import keras
import keras_hub
import tensorflow as tf

def get_layout_map(mesh):
    layout_map = keras.distribution.LayoutMap(mesh)
    # Sharding: Embeddings, MHA (Heads), and MLP (Intermediate)
    layout_map[".*token_embedding/embeddings"] = ("model", None)
    layout_map[".*position_embedding/embeddings"] = (None, "model")
    layout_map[".*self_attention/(query|key|value)/kernel"] = (None, "model", None)
    layout_map[".*self_attention/attention_output/kernel"] = ("model", None)
    layout_map[".*feedforward_intermediate_dense/kernel"] = (None, "model")
    layout_map[".*feedforward_output_dense/kernel"] = ("model", None)
    # Biases and LayerNorm
    layout_map[".*self_attention/(query|key|value)/bias"] = ("model", None)
    layout_map[".*feedforward_intermediate_dense/bias"] = ("model",)
    layout_map[".*layer_norm/(gamma|beta)"] = (None,)
    return layout_map

def get_data(num_samples=200):
    np.random.seed(42)
    x = {
        "token_ids": np.random.randint(0, 50272, (num_samples, 32)).astype("int32"),
        "padding_mask": np.ones((num_samples, 32), dtype="int32")
    }
    y = np.random.normal(size=(num_samples, 32, 768)).astype("float32")
    return x, y

def train_model(distribution, x, y, epochs=10, steps=5, batch_size=4):
    with distribution.scope():
        model = keras_hub.models.OPTBackbone.from_preset("opt_125m_en", dropout=0.0)
        model.compile(optimizer=keras.optimizers.Adam(1e-5), loss="mse", jit_compile=False)
        
        # Use tf.data.Dataset for native auto-sharding in multi-worker settings
        dataset = tf.data.Dataset.from_tensor_slices((x, y))
        dataset = dataset.batch(batch_size).repeat()
        
        history = model.fit(dataset, epochs=epochs, steps_per_epoch=steps, verbose=1)
        return float(history.history["loss"][-1])
