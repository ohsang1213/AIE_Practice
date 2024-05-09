import numpy as np
import struct
from pycoral.adapters import common
from pycoral.adapters import classify
from pycoral.utils.edgetpu import make_interpreter
import sys

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

    # Initialize the TF interpreter
    interpreter = make_interpreter(model_file)
    interpreter.allocate_tensors()

    # Normalize the input data
    image = image / 255.0
    image = np.expand_dims(image, axis=-1)

    params = common.input_details(interpreter, "quantization_parameters")
    scale = params["scales"]
    zero_point = params["zero_points"]
    if abs(scale) > 1e-5 and abs(zero_point) > 1e-5:
        image = image / scale + zero_point
    common.set_input(interpreter, image)

    # Run inference
    interpreter.invoke()

    # Get the output tensor
    classes = classify.get_classes(interpreter)

    # Print the output predictions
    for c in sorted(classes, key=lambda x: x.id):
        print(f"label : {c.id} {c.score * 100:.3f}%")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <tflite_model>")
        sys.exit(1)
    model_file = sys.argv[1]
    mnist_input = sys.argv[2]
    main(model_file)
