import json
import matplotlib.pyplot as plt
import os

def plot_comparison(strategy="dp"):
    jax_path = f"history_jax_{strategy}.json"
    torch_path = f"history_torch_{strategy}.json"
    
    if not os.path.exists(jax_path) or not os.path.exists(torch_path):
        print("Missing history files. Run the training first.")
        return

    with open(jax_path, "r") as f:
        jax_history = json.load(f)
    with open(torch_path, "r") as f:
        torch_history = json.load(f)

    plt.figure(figsize=(10, 6))
    plt.plot(jax_history["loss"], label="JAX Loss", marker='o')
    plt.plot(torch_history["loss"], label="Torch Loss", marker='x', linestyle='--')
    plt.title(f"Loss Comparison: JAX vs Torch ({strategy.upper()})")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid(True)
    
    output_png = f"loss_comparison_{strategy}.png"
    plt.savefig(output_png)
    print(f"Graph saved to {output_png}")
    
    # Print numerical values for quick check
    print("\nEpoch | JAX Loss   | Torch Loss | Diff")
    print("-" * 45)
    for i, (l1, l2) in enumerate(zip(jax_history["loss"], torch_history["loss"])):
        print(f"{i+1:5d} | {l1:.8f} | {l2:.8f} | {abs(l1-l2):.2e}")

if __name__ == "__main__":
    plot_comparison("dp")
