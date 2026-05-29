import os
import sys
import numpy as np
import json
import time

# Suppress framework noise
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

def get_data(num_samples=500, seq_len=32, embed_dim=768, vocab_size=50272):
    np.random.seed(42)
    x = {
        "token_ids": np.random.randint(0, vocab_size, (num_samples, seq_len)).astype("int32"),
        "padding_mask": np.ones((num_samples, seq_len), dtype="int32")
    }
    y = np.random.normal(size=(num_samples, seq_len, embed_dim)).astype("float32")
    return x, y

def run(backend):
    os.environ["KERAS_BACKEND"] = backend
    
    # Configure threading strategies intelligently for a single-device process baseline
    if backend == "jax":
        # Allow multi-threading to naturally accelerate JAX matrix operations on CPU
        os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=true"
    elif backend == "torch":
        import torch
        # Suppress dynamo recompilation warnings from spamming stdout
        import torch._dynamo
        torch._dynamo.config.suppress_errors = True

    import keras
    import keras_hub
    
    # Force precision settings cleanly 
    keras.backend.set_floatx("float32")
    
    if backend == "torch":
        import torch
        from torch.nn.attention import sdpa_kernel
        cm = sdpa_kernel(torch.nn.attention.SDPBackend.MATH)
    else:
        from contextlib import nullcontext
        cm = nullcontext()

    keras.utils.set_random_seed(42)
    
    model = keras_hub.models.OPTBackbone.from_preset("opt_125m_en", dropout=0.0)
    # Enable JIT compilation for tracking compiled execution paths
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-5, epsilon=1e-7), loss="mse", jit_compile=True)
    
    x, y = get_data()
    
    batch_size = 16
    epochs = 5
    
    with cm:
        # Warmup (Pushes graph compilation out of timed metrics block)
        model.fit(x, y, batch_size=batch_size, epochs=1, steps_per_epoch=1, verbose=1, shuffle=False)
        
        start_time = time.time()
        history = model.fit(x, y, batch_size=batch_size, epochs=epochs, steps_per_epoch=1, verbose=1, shuffle=False)
        end_time = time.time()
    
    training_time = end_time - start_time
    step_1_loss = float(history.history["loss"][0])
    final_loss = float(history.history["loss"][-1])
    
    total_samples = batch_size * 1 * epochs
    throughput = total_samples / training_time
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