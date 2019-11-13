# Project from
# https://machinelearningmastery.com/handwritten-digit-recognition-using-convolutional-neural-networks-python-keras/
# Visualization of CNN
# https://towardsdatascience.com/visualizing-intermediate-activation-in-convolutional-neural-networks-with-keras-260b36d60d0

# Larger CNN for the MNIST Dataset
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras.utils.vis_utils import plot_model
from keras import models
from keras.callbacks import ModelCheckpoint
from keras.preprocessing import image

from matplotlib import pyplot as plt
import numpy as np

# constants

NUM_EPOCHS = 1
BATCH_SIZE = 200

# Get GraphVis to work properly
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

# load data from MNIST
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# reshape to be [samples][width][height][channels]
X_train = X_train.reshape((X_train.shape[0], 28, 28, 1)).astype('float32')
X_test = X_test.reshape((X_test.shape[0], 28, 28, 1)).astype('float32')

# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

# Generate model
model = Sequential()

# First CNN Layer
model.add(Conv2D(30, (5, 5), input_shape=(28, 28, 1), activation='relu'))
model.add(MaxPooling2D())
# Second CNN Layer
model.add(Conv2D(15, (3, 3), activation='relu'))
model.add(MaxPooling2D())

# Dropout prevents overfitting
model.add(Dropout(0.2))

# Flatten merges results and puts them through a Fully connected NN
model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))


# Callback function to display intermediate progress
checkpointer = ModelCheckpoint(filepath="curWeights.hdf5",
                               monitor='val_accuracy',
                               verbose=1,
                               save_best_only=True)

# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Generate plot of model
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

# Fit the model
model_generation = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=NUM_EPOCHS,
                             batch_size=BATCH_SIZE, callbacks=[checkpointer])

# Plot performance (from https://towardsdatascience.com/visualizing-intermediate-activation-in-convolutional-neural-
# networks-with-keras-260b36d60d0)
acc = model_generation.history['accuracy']            # Change from 'acc'
val_acc = model_generation.history['val_accuracy']    # Change from 'val_acc'
loss = model_generation.history['loss']
val_loss = model_generation.history['val_loss']
epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("CNN Error: %.2f%%" % (100-scores[1]*100))

# Test our network with a hand-drawn image
img_path = 'notthree.png'

img = image.load_img(img_path, target_size=(28, 28))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255.

plt.imshow(img_tensor[0])
plt.show()

print(img_tensor.shape)

# Prediction
# TODO:  FIX IMAGE SHAPE DIMMENSIONS.  MAY NEED TO FLATTEN SOMEHOW
# See https://stackoverflow.com/questions/49057149/expected-conv2d-1-input-to-have-shape-28-28-1-but-got-array-with-shape-1-2
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
images = np.vstack([x])
classes = model.predict_classes(images, batch_size=10)
print("Predicted class is:", classes)

# Try to visualize the CNN models
# Extracts the outputs of the top 12 layers
layer_outputs = [layer.output for layer in model.layers[:12]]
# Creates a model that will return these outputs, given the model input
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
# Returns a list of five Numpy arrays: one array per layer activation
activations = activation_model.predict(img_tensor)

layer_names = []
for layer in model.layers[:12]:
    layer_names.append(layer.name)  # Names of the layers, so you can have them as part of your plot

images_per_row = 16

for layer_name, layer_activation in zip(layer_names, activations):  # Displays the feature maps
    n_features = layer_activation.shape[-1]  # Number of features in the feature map
    size = layer_activation.shape[1]  # The feature map has shape (1, size, size, n_features).
    n_cols = n_features // images_per_row  # Tiles the activation channels in this matrix
    display_grid = np.zeros((size * n_cols, images_per_row * size))
    for col in range(n_cols):  # Tiles each filter into a big horizontal grid
        for row in range(images_per_row):
            channel_image = layer_activation[0,
                            :, :,
                            col * images_per_row + row]
            channel_image -= channel_image.mean()  # Post-processes the feature to make it visually palatable
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[col * size: (col + 1) * size,  # Displays the grid
            row * size: (row + 1) * size] = channel_image
    scale = 1. / size
    plt.figure(figsize=(scale * display_grid.shape[1],
                        scale * display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')

