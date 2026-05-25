import json

def compare():
    try:
        with open("results_jax.json") as f: jax = json.load(f)
        with open("results_torch.json") as f: torch = json.load(f)
    except Exception as e:
        print(f"Error loading results: {e}")
        return

    metrics = [
        "step_1_loss", "step_5_loss", "throughput", 
        "training_time", "compilation_time", "peak_memory"
    ]

    print(f"{'Metric':<20} | {'JAX':<15} | {'Torch':<15}")
    print("-" * 55)
    for m in metrics:
        print(f"{m:<20} | {jax.get(m, 0):<15.4f} | {torch.get(m, 0):<15.4f}")

    print("\nPer-Device VRAM:")
    jax_dev = jax.get("per_device_memory", {})
    torch_dev = torch.get("per_device_memory", {})
    for dev in sorted(set(list(jax_dev.keys()) + list(torch_dev.keys()))):
        print(f"{dev:<20} | {jax_dev.get(dev, 0):>7.1f} MB | {torch_dev.get(dev, 0):>7.1f} MB")

if __name__ == "__main__":
    compare()
