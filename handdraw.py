#
# USE ESC KEY TO EXIT
# USE 'r' KEY TO RESET DRAWING

from my_model import *
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import cv2

# constants
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
    img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
    
    # Display
    cv2.imshow("After resizing", img)
    
    # Convert to tensor
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.

    # Let the model predict the digit
    classes = model.predict_classes(img_tensor, batch_size=10)
    
    print("Predicted digit is:", classes[0], end='\r')



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
    
    # Create a new window and setup the callback
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