import numpy as np
import struct
import sys
import tflite_runtime.interpreter as tflite

_EDGETPU_SHARED_LIB = "libedgetpu.so.1"

def readMnist(filename):
    input_vec = []
    with open(filename, "rb") as file:
        magic_number, number_of_images, n_rows, n_cols = struct.unpack(
            ">IIII", file.read(16)
        )
        for _ in range(n_rows):
            row = []
            for _ in range(n_cols):
                pixel = struct.unpack("B", file.read(1))[0]
                row.append(pixel)
            input_vec.append(row)
    return np.array(input_vec, dtype=np.float32)


def main(model_file):
    # Load mnist input image
    image = readMnist(mnist_input)
    print("Input MNIST Image")
    for i in range(28):
        for j in range(28):
            print(f"{int(image[i][j]):3d} ", end="")
        print()

    # Create Edge TPU delegate
    delegates = [tflite.load_delegate(_EDGETPU_SHARED_LIB)]

    # Make the interpreter with the target model and the delegate
    interpreter = tflite.Interpreter(
        model_path=model_file, experimental_delegates=delegates
    )

    # Allocate tensor buffers
    interpreter.allocate_tensors()

    # Normalize the input data
    image = image / 255.0
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = np.expand_dims(image, axis=-1)  # Add channel dimension

    # Fill input buffers
    input_details = interpreter.get_input_details()[0]
    interpreter.set_tensor(input_details["index"], image)

    # Run inference
    interpreter.invoke()

    # Get the output tensor
    output_details = interpreter.get_output_details()[0]
    output_data = interpreter.get_tensor(output_details["index"])

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
