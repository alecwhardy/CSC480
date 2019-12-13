# Reference:
# https://towardsdatascience.com/visualizing-intermediate-activation-in-convolutional-neural-networks-with-keras-260b36d60d0

import sys

import keras
import numpy as np
import PIL.ImageOps

from my_model import load_our_model

# Load the training model
from visualize_layers import visualize_cnn_layers

model = load_our_model()

# Load image from command line
try:
    img = keras.preprocessing.image.load_img(sys.argv[1], color_mode="grayscale", target_size=(28, 28))
except:
    if len(sys.argv) < 2:
        print("Usage: python3 main.py [[image]]")
    else:
        print("Cannot open %s" % sys.argv[1])
    exit()




# Test our network with a hand-drawn image
img_tensor = keras.preprocessing.image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255.

# # Changed cmap to "gray." This lets us present the image in the right color.
# plt.imshow(np.squeeze(img_tensor[0]), cmap="gray")
# plt.show()

print(img_tensor.shape)

# Pre-process
inverted_image = PIL.ImageOps.invert(img)
x = keras.preprocessing.image.img_to_array(inverted_image)
x = np.expand_dims(x, axis=0)
images = np.vstack([x])

# Make a prediction
classes = model.predict_classes(images, batch_size=10)
predicted_digit = classes[0]

# Output the prediction
print("Predicted digit is:", predicted_digit)


visualize_cnn_layers(model, x)
