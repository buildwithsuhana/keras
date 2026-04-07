
import numpy as np
import openvino as ov
import openvino.opset15 as ov_opset

def test_dft_float64():
    core = ov.Core()
    
    # Create input: (2, 4, 3, 2) where last dim is real/imag
    input_data = np.random.random((2, 4, 3, 2)).astype(np.float64)
    input_node = ov_opset.constant(input_data, ov.Type.f64)
    
    axes = ov_opset.constant([1, 2], ov.Type.i32)
    
    try:
        dft_node = ov_opset.dft(input_node, axes)
        model = ov.Model(results=[dft_node.output(0)], parameters=[])
        compiled_model = core.compile_model(model, "CPU")
        print("Successfully compiled DFT with float64 on CPU")
        
        # Run inference
        result = compiled_model({0: input_data})[0]
        print("Result dtype:", result.dtype)
        
    except Exception as e:
        print("Failed to run DFT with float64 on CPU:", e)

if __name__ == "__main__":
    test_dft_float64()
