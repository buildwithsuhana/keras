import os
import sys
import numpy as np
import json
import time

def get_data(num_samples=40, seq_len=32, embed_dim=768, vocab_size=50272):
    np.random.seed(42)
    x = {
        "token_ids": np.random.randint(0, vocab_size, (num_samples, seq_len)).astype("int32"),
        "padding_mask": np.ones((num_samples, seq_len), dtype="int32")
    }
    y = np.random.normal(size=(num_samples, seq_len, embed_dim)).astype("float32")
    return x, y

def run(backend):
    os.environ["KERAS_BACKEND"] = backend
    import keras
    import keras_hub
    
    if backend == "torch":
        import torch
        # Use MATH kernel for consistency across environments
        from torch.nn.attention import sdpa_kernel
        cm = sdpa_kernel(torch.nn.attention.SDPBackend.MATH)
    else:
        from contextlib import nullcontext
        cm = nullcontext()

    keras.utils.set_random_seed(42)
    
    model = keras_hub.models.OPTBackbone.from_preset("opt_125m_en", dropout=0.0)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-5), loss="mse")
    
    x, y = get_data()
    
    with cm:
        # Warmup
        model.fit(x, y, batch_size=4, epochs=1, steps_per_epoch=1, verbose=1, shuffle=False)
        
        start_time = time.time()
        epochs = 5
        history = model.fit(x, y, batch_size=4, epochs=epochs, steps_per_epoch=1, verbose=1, shuffle=False)
        end_time = time.time()
    
    training_time = end_time - start_time
    step_1_loss = float(history.history["loss"][0])
    final_loss = float(history.history["loss"][-1])
    # Report actual throughput (samples/sec)
    throughput = (4 * epochs) / training_time
    perplexity = float(np.exp(final_loss))
    
    results = {
        "step_1_loss": step_1_loss,
        "final_loss": final_loss,
        "perplexity": perplexity,
        "throughput": throughput,
        "training_time": training_time,
    }
    
    with open(f"results_simple_{backend}.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python simple_experiment.py <backend>")
        sys.exit(1)
    run(sys.argv[1])
