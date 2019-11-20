#
# USE ESC KEY TO EXIT
# USE 'r' KEY TO RESET DRAWING

import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing import image

from my_model import load_our_model
from visualize_layers import visualize_cnn_layers

model = load_our_model()

CANVAS_SIZE = 280   # MAKE SURE THIS IS A MULTIPLE OF 28

# Takes an image and uses the model to predict the digit
def process_and_predict(img):
    # Resize
    img_cv = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)

    # Display the resized image
    cv2.imshow("2. Resized image sent to model for prediction", img_cv)

    # Preprocess - invert and normalize
    img_cv = img_cv.astype('float32')
    img_cv = img_cv.reshape(1, 28, 28, 1)
    img_cv = 255-img_cv
    img_cv /= 255

    # Let the model predict the digit
    classes = model.predict_classes(img_cv, batch_size=10)
    predicted_digit = classes[0]

    # Output the prediction
    print("Predicted digit is:", predicted_digit, end='\r')
    
    # Display the prediction on the GUI
    predicted_digit_img = np.full((50, 50, 1), 255, np.uint8)
    cv2.putText(predicted_digit_img, str(predicted_digit), (14, 34), 2, 1, 0)
    cv2.imshow("3. Predicted digit", predicted_digit_img)

    # TODO: Connect the hand-drawing to the layer visualization, just for kicks!
    # visualize_cnn_layers(model, img_cv)



def handdraw():
    mouse_is_down = False # true if mouse is pressed
    ix, iy = -1, -1

    # Start with blank white canvas
    canvas = np.full((CANVAS_SIZE, CANVAS_SIZE, 1), 255, np.uint8)

    # mouse callback function
    def on_mouse_event(event, x, y, flags, param):
        nonlocal mouse_is_down, canvas

        if event == cv2.EVENT_LBUTTONDOWN:
            mouse_is_down = True
            ix, iy = x, y

        elif event == cv2.EVENT_MOUSEMOVE:
            if mouse_is_down:
                # Draw a circle
                cv2.circle(canvas, (x, y), 7, (0, 0, 0), -1)

        elif event == cv2.EVENT_LBUTTONUP:
            mouse_is_down = False
            cv2.circle(canvas, (x, y), 7, (0, 0, 0), -1)

        # Display the canvas
        cv2.imshow('1. Draw a digit here!', canvas)

        # Make prediction
        process_and_predict(canvas)

    # Create a new window and set up mouse events
    cv2.namedWindow('1. Draw a digit here!')
    cv2.setMouseCallback('1. Draw a digit here!', on_mouse_event)

    # Display the canvas
    cv2.imshow('1. Draw a digit here!', canvas)

    # Render loop
    while True:
        k = cv2.waitKey(1) & 0xFF
        if k == ord('r'):
            # "r" key - Reset to blank white canvas
            canvas = np.full((CANVAS_SIZE, CANVAS_SIZE, 1), 255, np.uint8)
        elif k == 27:
            # "Esc" key - quit
            break

    cv2.destroyAllWindows()


handdraw()
