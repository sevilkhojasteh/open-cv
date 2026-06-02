import cv2
import numpy as np
import matplotlib.pyplot as plt

img_gray = cv2.imread('/Users/sevilkhojasteh/Documents/to_learn/YOLOv8/data/images/company_logo.webp', cv2.IMREAD_GRAYSCALE)

# Method 1: global thresholding
ret, thresh_global = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
plt.imshow(thresh_global, cmap='gray')
plt.title('Global Thresholding (v=127)')
plt.show()

"""
1. Global Thresholding (The Absolute Judge)
cv2.threshold(img, 127, ...) looks at the entire image with one strict rule:

Is the pixel darker than 127? Turn it Black (0).

Is the pixel brighter than 127? Turn it White (255).

If you have a logo with dark text on a white background, it stays exactly like that. The white background stays white, and the dark text becomes pure black.
"""

# Method 2: Adaptive Mean Thresholding
thresh_adaptive_mean = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
plt.imshow(thresh_adaptive_mean, cmap='gray')
plt.title('Adaptive Mean Thresholding')
plt.show()

"""
2. Adaptive Thresholding (The Local Neighborhood)cv2.adaptiveThreshold does not use a single number for the whole image. Instead, it slides an $11 \times 11$ pixel box over your image.
For every single pixel, it calculates a custom threshold based on the average brightness of the 100 pixels immediately surrounding it, minus the constant $C$ (which you set to 2).
The Math That Causes the "Opposite" Effect:Let's look at what happens inside the solid areas of your logo.Inside a solid white background: The $11 \times 11$ neighborhood is pure white, so the average is $255$. 
The threshold formula is $255 - 2 = 253$. Because the center pixel ($255$) is greater than $253$, OpenCV turns it White.Inside a solid black letter: The $11 \times 11$ neighborhood is pure black, so the average is $0$. 
The threshold formula is $0 - 2 = -2$. Because the center pixel ($0$) is mathematically greater than $-2$, OpenCV turns it White!This is the "Aha!" moment: Adaptive thresholding turns the massive solid black shapes of your logo entirely white! 
The only places that become black are the exact edges of the logo, because that is the only place where the $11 \times 11$ box contains a mix of black and white pixels, dragging the average up high enough to force the dark pixels below the threshold.
"""

"""
When to use which?
Use Global Thresholding for digital images, logos, screenshots, or any image that already has perfect, even lighting.

Use Adaptive Thresholding for real-world photos taken with a camera (like a picture of a receipt, or a serial number stamped on a curved piece of metal), where half the object is in a dark shadow and the other half is under a bright glare.
"""