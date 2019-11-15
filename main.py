# Reference:
# https://towardsdatascience.com/visualizing-intermediate-activation-in-convolutional-neural-networks-with-keras-260b36d60d0

import sys

import numpy as np
import PIL.ImageOps
from keras import models
from keras.models import load_model
from keras.preprocessing import image
from matplotlib import pyplot as plt

from my_model import load_our_model

# Load the training model
model = load_our_model()

# Load image from command line
try:
    img = image.load_img(sys.argv[1], color_mode="grayscale", target_size=(28, 28))
except:
    if len(sys.argv) < 2:
        print("Usage: python3 main.py [[image]]")
    else:
        print("Cannot open %s" % sys.argv[1])
    exit()

# Test our network with a hand-drawn image
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255.

# # Changed cmap to "gray." This lets us present the image in the right color.
# plt.imshow(np.squeeze(img_tensor[0]), cmap="gray")
# plt.show()

print(img_tensor.shape)

# Prediction
inverted_image = PIL.ImageOps.invert(img)
x = image.img_to_array(inverted_image)
x = np.expand_dims(x, axis=0)
images = np.vstack([x])
classes = model.predict_classes(images, batch_size=10)
print("Predicted class is:", classes)

# Try to visualize the CNN models
# Extracts the outputs of the top 12 layers
layer_outputs = [layer.output for layer in model.layers[:4]]
# Creates a model that will return these outputs, given the model input
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
# Returns a list of five Numpy arrays: one array per layer activation
activations = activation_model.predict(img_tensor)

# Display a feature map for each layer in the neural network
IMAGES_PER_ROW = 8
for layer, activation in zip(model.layers[:4], activations):
    # print(activation.shape)
    n_features = activation.shape[-1]  # Number of features in the feature map
    size = activation.shape[1]  # The feature map has shape (1, size, size, n_features).
    n_cols = n_features // IMAGES_PER_ROW  # Tiles the activation channels in this matrix
    # print("size: %d, n_cols: %d, IMAGES_PER_ROW: %d" % (size, n_cols, IMAGES_PER_ROW))
    display_grid = np.zeros((size * n_cols, IMAGES_PER_ROW * size))
    for col in range(n_cols):  # Tiles each filter into a big horizontal grid
        for row in range(IMAGES_PER_ROW):
            channel_image = activation[0, :, :, col * IMAGES_PER_ROW + row]
            channel_image -= channel_image.mean()  # Post-processes the feature to make it visually palatable
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[col * size: (col + 1) * size, row * size: (row + 1) * size] = channel_image
    scale = 1. / size
    plt.figure(figsize=(scale * display_grid.shape[1], scale * display_grid.shape[0]))
    plt.title(layer.name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')
plt.show()
