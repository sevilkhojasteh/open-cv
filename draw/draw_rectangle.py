import cv2
import numpy as np

# --- 1. GLOBALS (State Variables) ---
drawing = False # True if mouse is pressed
ix, iy = -1, -1 # Initial X and Y coordinates

# --- 2. THE CALLBACK FUNCTION ---
def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing

    # Step A: User clicks the mouse down
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        # Record the exact starting coordinates
        ix, iy = x, y 

    # Step B: User drags the mouse
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            # We don't draw anything permanent yet!
            # (In advanced tools, this is where you'd draw a temporary "preview" box)
            pass

    # Step C: User releases the mouse button
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        # Draw the final rectangle from the starting point (ix, iy) to the current point (x, y)
        cv2.rectangle(img, pt1=(ix, iy), pt2=(x, y), color=(0, 255, 0), thickness=2)

# --- 3. THE UI SETUP ---
img = np.zeros((512, 512, 3), np.uint8)
cv2.namedWindow(winname='my_drawing')
cv2.setMouseCallback('my_drawing', draw_rectangle)

# --- 4. THE RENDER LOOP ---
while True:
    cv2.imshow('my_drawing', img)
    
    # Wait 1ms, break if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()