import keras
import numpy as np
import PIL.ImageOps
from matplotlib import pyplot as plt

IMAGES_PER_ROW = 8

# Create a "feature map" visualization of each 2D layer of the convolutional neural network.
def visualize_cnn_layers(model, img):
    print(model.layers)

    # Only the first 4 layers of the model are visualizable this way
    layers = model.layers[:4]
    
    # Create a "dummy model" that will simply return these outputs, given the model input
    activation_model = keras.models.Model(
        inputs=model.input,
        outputs=[layer.output for layer in layers]
    )

    # Pass an image to this dummy model to create a list of predictions
    # Returns a list of Numpy arrays, each holding prediction values for a given layer in the model
    activations = activation_model.predict(img)

    # Display a feature map for each layer in the neural network
    for layer, layer_activation in zip(layers, activations):
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
                channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')

                display_grid[col * size: (col + 1) * size, row * size: (row + 1) * size] = channel_image
        
        # Display using pyplot
        scale = 1. / size
        plt.figure(figsize=(scale * display_grid.shape[1], scale * display_grid.shape[0]))
        plt.title(layer.name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')
    
    plt.show()