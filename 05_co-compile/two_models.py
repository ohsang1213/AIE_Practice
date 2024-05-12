import argparse
import time
import numpy as np
from PIL import Image
from pycoral.adapters import classify
from pycoral.adapters import common
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter

def transform(interpreter, image, args):
  # Image data must go through two transforms before running inference:
  # 1. normalization: f = (input - mean) / std
  # 2. quantization: q = f / scale + zero_point
  # The following code combines the two steps as such:
  # q = (input - mean) / (std * scale) + zero_point
  # However, if std * scale equals 1, and mean - zero_point equals 0, the input
  # does not need any preprocessing (but in practice, even if the results are
  # very close to 1 and 0, it is probably okay to skip preprocessing for better
  # efficiency; we use 1e-5 below instead of absolute zero).
  params = common.input_details(interpreter, 'quantization_parameters')
  scale = params['scales']
  zero_point = params['zero_points']
  mean = args.input_mean
  std = args.input_std
  if abs(scale * std - 1) < 1e-5 and abs(mean - zero_point) < 1e-5:
    # Input data does not require preprocessing.
    common.set_input(interpreter, image)
  else:
    # Input data requires preprocessing
    normalized_input = (np.asarray(image) - mean) / (std * scale) + zero_point
    np.clip(normalized_input, 0, 255, out=normalized_input)
    common.set_input(interpreter, normalized_input.astype(np.uint8))

def main():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      '-m1', '--model1', required=True, help='File path of .tflite file.')
  parser.add_argument(
      '-m2', '--model2', required=True, help='File path of .tflite file.')
  parser.add_argument(
      '-i', '--input', required=True, help='Image to be classified.')
  parser.add_argument(
      '-l', '--labels', help='File path of labels file.', default='../all_models/imagenet_labels.txt')
  parser.add_argument(
      '-k', '--top_k', type=int, default=1,
      help='Max number of classification results')
  parser.add_argument(
      '-t', '--threshold', type=float, default=0.0,
      help='Classification score threshold')
  parser.add_argument(
      '-c', '--count', type=int, default=5,
      help='Number of times to run inference')
  parser.add_argument(
      '-a', '--input_mean', type=float, default=128.0,
      help='Mean value for input normalization')
  parser.add_argument(
      '-s', '--input_std', type=float, default=128.0,
      help='STD value for input normalization')
  args = parser.parse_args()

  labels = read_label_file(args.labels) if args.labels else {}
  interpreter1 = make_interpreter(*args.model1.split('@'))
  interpreter1.allocate_tensors()
  interpreter2 = make_interpreter(*args.model2.split('@'))
  interpreter2.allocate_tensors()

  # Model must be uint8 quantized
  if common.input_details(interpreter1, 'dtype') != np.uint8:
    raise ValueError('Only support uint8 input type.')
  if common.input_details(interpreter2, 'dtype') != np.uint8:
    raise ValueError('Only support uint8 input type.')

  size1 = common.input_size(interpreter1)
  image1 = Image.open(args.input).convert('RGB').resize(size1, Image.LANCZOS)
  size2 = common.input_size(interpreter2)
  image2 = Image.open(args.input).convert('RGB').resize(size2, Image.LANCZOS)

  transform(interpreter1, image1, args)
  transform(interpreter2, image2, args)

  interpreters = [interpreter1, interpreter2]

  # Run inference
  print('------MODEL A\'s INFERENCE TIME------')
  for _ in range(args.count):
    start = time.perf_counter()
    interpreters[0].invoke()
    inference_time = time.perf_counter() - start
    classes = classify.get_classes(interpreters[0], args.top_k, args.threshold)
    print('%.1fms' % (inference_time * 1000))
  for c in classes:
    print('Result - %s: %.5f' % (labels.get(c.id, c.id), c.score))
  print('------------------------------------')

  print('------MODEL B\'s INFERENCE TIME------')
  for _ in range(args.count):
    start = time.perf_counter()
    interpreters[1].invoke()
    inference_time = time.perf_counter() - start
    classes = classify.get_classes(interpreters[1], args.top_k, args.threshold)
    print('%.1fms' % (inference_time * 1000))
  for c in classes:
    print('Result - %s: %.5f' % (labels.get(c.id, c.id), c.score))
  print('------------------------------------')

  print('----MODEL A & B\'s INFERENCE TIME----')
  print('MODEL A  MODEL B')
  total_start = time.perf_counter()
  for _ in range(args.count):
    for i in range(2):
        start = time.perf_counter()
        interpreters[i].invoke()
        inference_time = time.perf_counter() - start
        classes = classify.get_classes(interpreters[i], args.top_k, args.threshold)
        if i == 0:
            print('%.1fms' % (inference_time * 1000))
        else:
            print('         %.1fms' % (inference_time * 1000))
  total_time = time.perf_counter() - total_start
  print('Total time: %.1fms' % (total_time * 1000))
  print('------------------------------------')

if __name__ == '__main__':
  main()