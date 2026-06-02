import cv2
import numpy as np

# 1. Define the Callback Function
# This function MUST accept these 5 specific parameters, even if you don't use them all.
def draw_circle(event, x, y, flags, param):
    
    # If the left mouse button is double-clicked or clicked down
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(img, (x, y), 50, (0, 255, 0), -1) # -1 thickness means fill the shape
        
    # If the right mouse button is clicked down
    elif event == cv2.EVENT_RBUTTONDOWN:
        cv2.circle(img, (x, y), 50, (255, 0, 0), -1)

# 2. Connect the function to a window
cv2.namedWindow(winname='my_drawing')
cv2.setMouseCallback('my_drawing', draw_circle)

# 3. Show the image and keep it updating
img = np.zeros((512, 512, 3), np.uint8)

while True:
    cv2.imshow('my_drawing', img)
    
    # Wait 1 ms. If the user presses 'q' (ASCII 113), break the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()


