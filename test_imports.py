import os

# Set backend to torch
os.environ["KERAS_BACKEND"] = "torch"

try:
    print("Successfully imported core")
except Exception as e:
    print(f"Failed to import core: {e}")
    import traceback

    traceback.print_exc()
