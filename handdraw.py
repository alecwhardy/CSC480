#
# USE ESC KEY TO EXIT
# USE 'r' KEY TO RESET DRAWING

import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing import image

from my_model import *

# Constants
GENERATE_NEW_MODEL = False
NUM_EPOCHS = 10
BATCH_SIZE = 200

if GENERATE_NEW_MODEL:
    model, model_gen = generate_model(NUM_EPOCHS, BATCH_SIZE)
    plot_fit_performance(model_gen)
else:
    model = load_model("model.h5")


# Takes an image and uses the model to predict the digit
def resize_and_predict(img):
    # Resize
    img_cv = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)

    # Display
    cv2.imshow("After resizing", img_cv)

    # Convert and normalize
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
    cv2.putText(predicted_digit_img, str(predicted_digit), (15, 32), 2, 1, 0)
    cv2.imshow("Predicted digit", predicted_digit_img)



def handdraw():
    mouse_is_down = False # true if mouse is pressed
    ix, iy = -1, -1

    # Start with blank white canvas
    canvas = np.full((140, 140, 1), 255, np.uint8)

    # mouse callback function
    def draw_circle(event, x, y, flags, param):
        nonlocal mouse_is_down

        if event == cv2.EVENT_LBUTTONDOWN:
            mouse_is_down = True
            ix, iy = x, y

        elif event == cv2.EVENT_MOUSEMOVE:
            if mouse_is_down:
                cv2.circle(canvas, (x, y), 7, (0, 0, 0), -1)

        elif event == cv2.EVENT_LBUTTONUP:
            mouse_is_down = False
            cv2.circle(canvas, (x, y), 7, (0, 0, 0), -1)

    # Create a new window and set up mouse events
    cv2.namedWindow('Draw a digit here!')
    cv2.setMouseCallback('Draw a digit here!', draw_circle)

    # Render loop
    while True:
        # Display the canvas
        cv2.imshow('Draw a digit here!', canvas)

        k = cv2.waitKey(1) & 0xFF
        if k == ord('r'):
            # "r" key - Reset to blank white canvas
            canvas = np.full((140, 140, 1), 255, np.uint8)
        elif k == 27:
            # "Esc" key - quit
            break

        # Make prediction
        resize_and_predict(canvas)

    cv2.destroyAllWindows()


handdraw()
