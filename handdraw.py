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


mouse_down = False # true if mouse is pressed
ix, iy = -1, -1


# mouse callback function
def draw_circle(event,x,y,flags,param):
    global ix, iy, mouse_down, mode

    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_down = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if mouse_down:
            cv2.circle(img, (x, y), 5, (0, 0, 0), -1)

    elif event == cv2.EVENT_LBUTTONUP:
        mouse_down = False
        cv2.circle(img, (x, y), 5, (0, 0, 0), -1)


img = np.full((140, 140, 1), 255, np.uint8)
cv2.namedWindow('image')
cv2.setMouseCallback('image', draw_circle)

while True:
    cv2.imshow('image', img)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('r'):
        img = np.full((140, 140, 1), 255, np.uint8)
    elif k == 27:
        break

    # Prediction magic here
    img_cv = cv2.resize(img, (28, 28))
    img_tensor = image.img_to_array(img_cv)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.
    cv2.imshow("resized", img_cv)

    classes = model.predict_classes(img_tensor, batch_size=10)
    print("Predicted digit is:", classes[0])
    # End prediction magic

cv2.destroyAllWindows()