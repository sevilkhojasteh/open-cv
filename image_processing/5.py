import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread("/Users/sevilkhojasteh/Documents/to_learn/YOLOv8/data/images/B.webp")

kernel = np.ones((5, 5), dtype=np.uint8)
# 1. EROSION: This is like a "shrinking" operation. It erodes away the boundaries of the foreground object. The kernel slides over the image, and if any pixel under the kernel is 0 (Black), the central pixel is set to 0. This can help remove small white noise and detach connected objects.
eroded_img = cv2.erode(img, kernel, iterations=1)
plt.subplot(2, 2, 1) # Create a subplot with 2 rows and 2 columns, and set the first plot as active
plt.imshow(eroded_img) # Display the eroded image
plt.show() # Show the plot
# 2. DILATION: This is the opposite of erosion. It "grows" the foreground object. The kernel slides over the image, and if any pixel under the kernel is 255 (White), the central pixel is set to 255. This can help fill in small black holes and connect nearby objects.
dilated_img = cv2.dilate(img, kernel, iterations=1)
plt.subplot(2, 2, 2) # Set the second plot as active
plt.imshow(dilated_img) # Display the dilated image
plt.show() # Show the plot
# 3. OPENING: This is erosion followed by dilation. It is useful for removing small objects from the foreground (like noise) while keeping the shape of larger objects intact.
opened_img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
plt.subplot(2, 2, 3) # Set the third plot as active
plt.imshow(opened_img) # Display the opened image
plt.show() # Show the plot
# 4. CLOSING: This is dilation followed by erosion. It is useful for closing small holes in the foreground objects, or small black points on the object.
closed_img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
plt.subplot(2, 2, 4) # Set the fourth plot as active
plt.imshow(closed_img) # Display the closed image
plt.show() # Show the plot  
