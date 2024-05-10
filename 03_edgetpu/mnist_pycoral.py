import numpy as np
import struct
import sys
from pycoral.adapters import common
from pycoral.adapters import classify
from pycoral.utils.edgetpu import make_interpreter

def read_mnist(filename):
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
    return np.array(input_vec, dtype=np.float32)


def main(model_file):
    # Load mnist input image
    image = read_mnist(mnist_input)
    print("Input MNIST Image")
    for i in range(28):
        for j in range(28):
            print(f"{int(image[i][j]):3d} ", end="")
        print()

    # Make the interpreter with the target model
    interpreter = make_interpreter(model_file)

    # Allocate tensor buffers
    interpreter.allocate_tensors()

    # Normalize the input data
    image = image / 255.0
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = np.expand_dims(image, axis=-1)  # Add channel dimension

    # Fill input buffers
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
