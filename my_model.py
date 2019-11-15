from keras.callbacks import ModelCheckpoint
from keras.datasets import mnist
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.utils import np_utils
from keras.utils.vis_utils import plot_model
from matplotlib import pyplot as plt

# Reference
# https://machinelearningmastery.com/handwritten-digit-recognition-using-convolutional-neural-networks-python-keras/

def generate_model(num_epochs, batch_size):
    # load data from MNIST
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # reshape to be [samples][width][height][channels], normalized
    X_train = X_train.reshape((X_train.shape[0], 28, 28, 1)).astype('float32') / 255
    X_test = X_test.reshape((X_test.shape[0], 28, 28, 1)).astype('float32') / 255

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

    # Dropout prevents over-fitting
    model.add(Dropout(0.2))

    # Flatten merges results and puts them through a Fully connected NN
    model.add(Flatten())

    model.add(Dense(128, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    checkpoint = ModelCheckpoint(filepath="curWeights.hdf5",
                                   monitor='val_accuracy',
                                   verbose=1,
                                   save_best_only=True)

    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Generate plot of model
    plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

    # Fit the model
    model_generation = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=num_epochs,
                                 batch_size=batch_size, callbacks=[checkpoint])
    model.save("model.h5")

    # Final evaluation of the model
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("CNN Error: %.2f%%" % (100 - scores[1] * 100))
    return model, model_generation


def plot_fit_performance(model_gen):
    # Reference:
    # Plot performance (from https://towardsdatascience.com/visualizing-intermediate-activation-in-convolutional-neural-
    # networks-with-keras-260b36d60d0)
    acc = model_gen.history['accuracy']  # Change from 'acc'
    val_acc = model_gen.history['val_accuracy']  # Change from 'val_acc'
    loss = model_gen.history['loss']
    val_loss = model_gen.history['val_loss']
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
