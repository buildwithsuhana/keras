import numpy as np

import keras


def to_playground(model, **kwargs):
    """
    Instantly transforms a Keras model into an interactive Web UI demo.
    Automatically infers input and output widgets based on model shape.
    """
    try:
        import gradio as gr
    except ImportError:
        raise ImportError(
            "Gradio is required for to_playground(). \n"
            "Please install it via: pip install gradio"
        )

    # 1. Inspect Input Shape to Determine Input Widget
    # Standard Keras shapes are (Batch_Size, Dimensions...)
    input_shape = model.input_shape

    # Handle multi-input models (fallback to first input)
    if isinstance(input_shape, list):
        input_shape = input_shape[0]

    inputs = None

    # (Batch, Height, Width, Channels) -> Image Model
    if len(input_shape) == 4:
        h, w, c = input_shape[1], input_shape[2], input_shape[3]
        if c == 1:
            # Grayscale (e.g., Handwriting/MNIST)
            inputs = gr.Sketchpad(
                crop=False, canvas_size=(h, w), label=f"Draw here ({h}x{w})"
            )
        else:
            # Color Image (e.g., ResNet/MobileNet)
            inputs = gr.Image(
                type="numpy", tool="editor", label=f"Upload Image ({h}x{w}x{c})"
            )

    # (Batch, Features) -> Tabular or Text
    elif len(input_shape) == 2:
        # We can implement further logic to detect Text vs Tabular
        inputs = gr.Textbox(lines=2, placeholder="Enter value or text here...")

    else:
        # Fallback to simple generic components
        inputs = gr.Dataframe()

    # 2. Inspect Output Shape to Determine Output Widget
    output_shape = model.output_shape
    if isinstance(output_shape, list):
        output_shape = output_shape[0]

    outputs = None

    # Check if Classification (Multiple output neurons)
    if output_shape[-1] > 1:
        outputs = gr.Label(num_top_classes=3, label="Top Predictions")
    else:
        outputs = gr.Textbox(label="Model Output")

    # 3. Define the Prediction Bridge
    def predict_bridge(input_data):
        # Apply standard preprocessing depending on input type
        if isinstance(inputs, gr.Sketchpad):
            # Digits/Handwriting often need reshaping back to standard array
            input_resized = np.array(input_data).reshape(1, h, w, c)
            input_resized = input_resized / 255.0  # standard norm
        else:
            input_resized = input_data

        # Run Keras Prediction
        predictions = model.predict(input_resized)

        # Format Classification nicely
        if output_shape[-1] > 1:
            class_names = [f"Class {i}" for i in range(output_shape[-1])]
            return {
                class_names[i]: float(predictions[0][i])
                for i in range(len(class_names))
            }

        return predictions[0]

    # 4. Launch the Playground
    title = f"Keras Interactive Playground - {model.name}"
    desc = f"Input Shape: {input_shape} | Output Shape: {output_shape}"

    app = gr.Interface(
        fn=predict_bridge,
        inputs=inputs,
        outputs=outputs,
        title=title,
        description=desc,
        **kwargs,
    )

    app.launch(inline=True, share=True)
    return app


# --- Example Usage ---
if __name__ == "__main__":
    print("Creating a sample Keras model...")
    # Create a simple generic CNN model (e.g., like for MNIST)
    sample_model = keras.Sequential(
        [
            keras.layers.Input(shape=(28, 28, 1)),
            keras.layers.Conv2D(32, 3, activation="relu"),
            keras.layers.Flatten(),
            keras.layers.Dense(10, activation="softmax"),
        ],
        name="Sample_Digit_Classifier",
    )

    # Compiling with dummy optimizer to build it
    sample_model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy"
    )

    print("Launching Keras Playground prototype...")
    print("NOTE: You will need 'gradio' installed for this to work.")
    print("Run: pip install gradio")

    try:
        to_playground(sample_model)
    except ImportError as e:
        print(f"\n[Error] Could not start playground: {e}")
