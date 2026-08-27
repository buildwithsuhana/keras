import os
import numpy as np
import keras
import keras_hub
import tensorflow as tf

def get_model(vocab_size=50272):
    keras.backend.set_floatx("float32")
    model = keras_hub.models.OPTCausalLM.from_preset(
        "opt_125m_en",
        load_weights=False,
        preprocessor=None,
    )
    # Initialize weights deterministically for comparison
    return model

def get_dataset():
    # Use synthetic data for stability and speed in this check
    # identical seeds ensure same data across backends
    np.random.seed(42)
    vocab_size = 50272
    seq_length = 128
    num_samples = 16
    
    token_ids = np.random.randint(0, vocab_size, (num_samples, seq_length + 1), dtype="int32")
    
    def split_input_target(chunk):
        input_text = chunk[:, :-1]
        target_text = chunk[:, 1:]
        return {
            "token_ids": input_text,
            "padding_mask": np.ones_like(input_text, dtype="bool"),
        }, target_text

    x, y = split_input_target(token_ids)
    return x, y

def save_results(backend, initial_loss, fit_loss, final_loss):
    import json
    results_file = f"results_{backend}.json"
    data = {
        "backend": backend,
        "initial_loss": float(initial_loss),
        "fit_loss": float(fit_loss),
        "final_loss": float(final_loss)
    }
    with open(results_file, "w") as f:
        json.dump(data, f, indent=4)
    print(f"Results for {backend} saved to {results_file}")
