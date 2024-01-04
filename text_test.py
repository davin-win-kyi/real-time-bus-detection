import matplotlib.pyplot as plt

import keras_ocr

# keras-ocr will automatically download pretrained
# weights for the detector and recognizer.
print("Loading model... ")
pipeline = keras_ocr.pipeline.Pipeline()
print("Model loaded")

# Test an image
print("Testing model... ")
read_image = keras_ocr.tools.read("C:\\Users\\davin\\PycharmProjects\\real-world-alt-text_test\\real-time_bus_detection\\cropped_imgs\\test_image#2.png")
prediction_groups = pipeline.recognize([read_image])
print("Model tested")

print(prediction_groups)