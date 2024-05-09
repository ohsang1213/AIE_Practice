import numpy as np
import struct
import sys
import tflite_runtime.interpreter as tflite

_EDGETPU_SHARED_LIB = "libedgetpu.so.1"

def read_Mnist(filename):
    input_vec = []
    with open(filename, "rb") as file:
        magic_number, number_of_images, n_rows, n_cols = struct.unpack(
            ">IIII", file.read(16)
        )
        for _ in range(1):
            for _ in range(n_rows):
                row = []
                for _ in range(n_cols):
                    pixel = struct.unpack("B", file.read(1))[0]
                    row.append(pixel)
                input_vec.append(row)
    return np.array(input_vec)


def main(model_file):
    image = read_Mnist(mnist_input)

    # Print the input image
    print("Input MNIST Image")
    for i in range(28):
        for j in range(28):
            print(f"{image[i][j]:3d} ", end="")
        print()

    delegates = [tflite.load_delegate(_EDGETPU_SHARED_LIB)]
    interpreter = tflite.Interpreter(
        model_path=model_file, experimental_delegates=delegates
    )

    # Initialize the TF interpreter
    interpreter.allocate_tensors()

    # Normalize the input data
    image = image / 255.0
    image = np.expand_dims(image, axis=-1)

    input_details = interpreter.get_input_details()[0]
    params = input_details["quantization_parameters"]
    scale = params["scales"]
    zero_point = params["zero_points"]
    
    # TODO: scale 및 zero_point 관련 예외 처리 로직 수정 필요
    if abs(scale) > 1e-5 and abs(zero_point) > 1e-5:
        image = image / scale + zero_point

    tensor_index = input_details["index"]
    interpreter.tensor(tensor_index)()[0][:, :] = image

    # Run inference
    interpreter.invoke()

    # Get the output tensor
    output_details = interpreter.get_output_details()[0]
    scale, zero_point = output_details["quantization"]

    output_data = interpreter.get_tensor(output_details["index"])
    
    # TODO: scale 및 zero_point 관련 예외 처리 로직 수정 필요
    if abs(scale) > 1e-5 and abs(zero_point) > 1e-5:
        output_data = scale * (output_data.astype(np.int64) - zero_point)

    # Print the output predictions
    for i in range(10):
        print(f"label : {i} {output_data[0][i] * 100:.3f}%")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <tflite_model> <mnist_input>")
        sys.exit(1)
    model_file = sys.argv[1]
    mnist_input = sys.argv[2]
    main(model_file)