import keras
import numpy as np
import PIL.ImageOps
from matplotlib import pyplot as plt
import cv2

IMAGES_PER_ROW = 8

# Create a "feature map" visualization of each 2D layer of the convolutional neural network.
def visualize_cnn_layers(model, img):
    # print(model.layers)
    layers = model.layers

    # Create a "dummy model" that will simply return these outputs, given the model input
    activation_model = keras.models.Model(
        inputs=model.input,
        outputs=[layer.output for layer in layers]
    )

    # Pass an image to this dummy model to create a list of predictions
    # Returns a list of Numpy arrays, each holding prediction values for a given layer in the model
    activations = activation_model.predict(img)

    # Display a feature map for each 2D layer in the neural network
    for layer, layer_activation in zip(layers[:4], activations[:4]): # In our case, only the first 4 layers of the model are visualizable this way
        n_features = layer_activation.shape[-1]  # Number of features in the feature map
        size = layer_activation.shape[1]  # The feature map has shape (1, size, size, n_features).
        n_cols = n_features // IMAGES_PER_ROW  # Tiles the activation channels in this matrix
        # print("size: %d, n_cols: %d, IMAGES_PER_ROW: %d" % (size, n_cols, IMAGES_PER_ROW))

        # Tiles each filter into a big horizontal grid
        display_grid = np.zeros((size * n_cols, IMAGES_PER_ROW * size))
        for col in range(n_cols):
            for row in range(IMAGES_PER_ROW):
                channel_image = layer_activation[0, :, :, col * IMAGES_PER_ROW + row]

                # Post-process the feature to make it visually palatable
                channel_image -= channel_image.mean()
                if channel_image.std() != 0:
                    channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')

                display_grid[col * size: (col + 1) * size, row * size: (row + 1) * size] = channel_image
        
        # TODO: Connect the hand-drawing to the layer visualization, just for kicks!
        # Display using opencv
        # cv2.namedWindow(layer.name)
        # cv2.imshow(layer.name, display_grid)
        
        # Display using pyplot
        scale = 1. / size
        plt.figure(figsize=(scale * display_grid.shape[1], scale * display_grid.shape[0]))
        plt.title(layer.name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')

    plt.show()
    

    # Print the activation values of the final dense layer (each one represents a possible digit from 0-9)
    print("Activation values of final dense layer:")
    for digit, value in enumerate(activations[-1][0]):
        print("%d: %f" % (digit, value))