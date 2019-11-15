import os
import sys

import numpy as np
import PIL.ImageOps
from keras import models
from keras.callbacks import ModelCheckpoint
from keras.datasets import mnist
from keras.engine.saving import load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.models import Sequential, load_model
from keras.utils import np_utils
from keras.utils.vis_utils import plot_model
from matplotlib import pyplot as plt

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/' # Get GraphViz to work properly

# Constants used during training
EPOCHS = 10
BATCH_SIZE = 200

# File to save model state
MODEL_FILEPATH = "model-state.hdf5"

# Reference
# https://machinelearningmastery.com/handwritten-digit-recognition-using-convolutional-neural-networks-python-keras/

def create_model():
    # Create a training model. We are using a "sequential" training model with layers of neurons.
    # Each layer will be defined and configured below.
    model = Sequential()

    # First CNN Layer
    model.add(Conv2D(30, (5, 5), input_shape=(28, 28, 1), activation='relu'))
    model.add(MaxPooling2D())

    # Second CNN Layer
    model.add(Conv2D(15, (3, 3), activation='relu'))
    model.add(MaxPooling2D())

    # Dropout prevents over-fitting
    model.add(Dropout(0.2))

    # Flatten merges results and puts them through a Fully connected NN
    model.add(Flatten())

    # Narrow down to a layer of size 128
    model.add(Dense(128, activation='relu'))
    # Narrow down to a layer of size 50
    model.add(Dense(50, activation='relu'))
    # Narrow down to the final layer, size 10 (each one represents a possible digit from 0-9)
    model.add(Dense(10, activation='softmax'))

    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


# Train the model using the MNIST dataset
def train_model(model, epochs, batch_size):
    # Load data from MNIST
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # reshape to be [samples][width][height][channels], normalized
    x_train = x_train.reshape((x_train.shape[0], 28, 28, 1)).astype('float32') / 255
    x_test = x_test.reshape((x_test.shape[0], 28, 28, 1)).astype('float32') / 255

    # one hot encode outputs
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)

    # Train the model
    training_history = model.fit(
        x_train, y_train,
        validation_data=(x_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[
            # During training, display intermediate progress
            # Also, generate a file that stores the state of the model so we don't have to retrain each time
            ModelCheckpoint(
                filepath=MODEL_FILEPATH,
                monitor='val_accuracy',
                verbose=1,
                save_best_only=True
            )
        ]
    )

    # Now that we've trained the model, calculate how well the training data fits the model
    scores = model.evaluate(x_test, y_test, verbose=0)
    print("CNN Error: %.2f%%" % (100 - scores[1] * 100))

    return training_history


# Reference:
# Plot performance
# https://towardsdatascience.com/visualizing-intermediate-activation-in-convolutional-neural-networks-with-keras-260b36d60d0

def plot_training_performance(training_history):
    history = training_history.history
    acc = history['accuracy']  # Change from 'acc'
    val_acc = history['val_accuracy']  # Change from 'val_acc'
    loss = history['loss']
    val_loss = history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, 'bo', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()



def generate_and_train_model():
    # Generate the model
    model = create_model()

    # Generate a diagram of the model and save to an image file (requires Graphviz)
    try:
        plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    except:
        pass

    # Train the model
    training_history = train_model(model, epochs=EPOCHS, batch_size=BATCH_SIZE)
    
    # Show a plot of how it was trained
    plot_training_performance(training_history)

    return model


# Load the trained model from the save file.
# If it doesn't exist, create a new model, train it, and save to file.
def load_our_model():
    try:
        model = load_model(MODEL_FILEPATH)
    except:
        model = generate_and_train_model()
    return model


# Run this file directly to force generate and train a new model
if __name__ == "__main__":
    generate_and_train_model()
