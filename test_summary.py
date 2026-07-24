import keras

# 1. Funky Model Name (Contains LaTeX/Markdown/HTML traps)
inputs = keras.Input(shape=(10,), name="input_1")

# 2. Layer with a Pipe '|' (Markdown Table Trap!)
x = keras.layers.Dense(5, name="dense | hidden_layer")(inputs)

# 3. Layer with Angle Brackets '<>' (HTML Tag Trap!)
x = keras.layers.Dense(10, name="dense_<activation>_out")(x)

# 4. Lambda Layer (Standard framework artifacts often look like this)
y = keras.layers.Lambda(lambda z: z, name="<lambda>_identity")(x)

model = keras.Model(inputs=inputs, outputs=y, name="model_v1.0_<alpha> | prod")


# === 🟢 Mode 1: Standard Summary (Available Before & After) ===
print("=== Standard Summary (Text/ASCII) ===")
# This is what you got before, and still get by default.
model.summary()


print("\n=== Markdown Summary ===")
model.summary(format="markdown")

print("\n=== HTML Summary ===")
model.summary(format="html")

print("\n=== LaTeX Summary ===")
model.summary(format="latex")
