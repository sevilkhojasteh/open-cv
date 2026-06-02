"""
Blending and Pasting (Adding a Logo to a Feed)
There are two ways to combine images. addWeighted is for translucent overlays (like a watermark). Masks are for pasting a solid shape without its background.

Method A: Translucent Blending
Both images must be the exact same size.

"""
import cv2
import matplotlib.pyplot as plt
import numpy as np

# img1 = cv2.imread('/Users/sevilkhojasteh/Documents/to_learn/YOLOv8/data/images/nennieinszweidrei-cat-5628953_1920.jpg')
# img2 = cv2.imread('/Users/sevilkhojasteh/Documents/to_learn/YOLOv8/data/images/water_drop.webp')
# height = img1.shape[0]
# width = img1.shape[1]
# img2 = cv2.resize(img2, (width, height))
# print(img1.shape) # Output: (height, width, channels)
# print(img2.shape) # Output: (height, width, channels)
# # img1 gets 70% opacity, img2 gets 30% opacity. 0 is the gamma (leave at 0).
# blended = cv2.addWeighted(img1, 0.7, img2, 0.3, 0)
# plt.imshow(blended) # Display the blended image using Matplotlib
# plt.show() # Show the plot
"""
Method B: Solid Pasting (The Bitwise Mask)
Pasting a small logo onto the bottom-right corner of a larger image.
"""
main_img = cv2.imread('/Users/sevilkhojasteh/Documents/to_learn/YOLOv8/data/images/nennieinszweidrei-cat-5628953_1920.jpg')
logo = cv2.imread('/Users/sevilkhojasteh/Documents/to_learn/YOLOv8/data/images/company_logo.webp')
# Get dimensions of the logo
rows, cols, channels = logo.shape
# print(logo.shape) # Output: (height, width, channels)
# 1. Define the Region of Interest (ROI) on the main image (Top-Left corner here)
roi = main_img[0:rows, 0:cols]
my_roi = main_img.copy() # Make a copy of the ROI for visualization
my_roi[0:rows, 0:cols] = logo # Paste the logo directly onto the ROI (This is just for visualization, not the final result)
plt.imshow(my_roi) # Display the ROI
plt.show() # Show the plot
"""
This technique is way easier than trying to calculate the exact pixel locations of the logo on the main image. You just say "I want to paste the logo at the top-left corner, so my ROI is the same size as the logo, starting at (0,0)." If you wanted to paste it in the bottom-right corner, you would calculate the starting point as (main_img_width - logo_width, main_img_height - logo_height) and define your ROI accordingly.
"""

# print(roi.shape) # Output: (height, width, channels) - Should match the logo's shape
# 2. Create a mask of the logo (Convert to grayscale, then threshold)
logo_gray = cv2.cvtColor(logo, cv2.COLOR_BGR2GRAY)
# Make the background of the logo black, and the logo itself white
ret, mask = cv2.threshold(logo_gray, 240, 255, cv2.THRESH_BINARY_INV) 
# print(mask.shape) # Output: (height, width) - Single channel (grayscale)
# plt.imshow(mask, cmap='gray') # Display the binary mask
# plt.show() # Show the plot
# 3. Cut out the hole in the ROI
main_bg = cv2.bitwise_and(roi, roi, mask=cv2.bitwise_not(mask))
plt.imshow(main_bg) # Display the ROI with the hole
plt.show() # Show the plot
# 4. Extract the logo pixels
logo_fg = cv2.bitwise_and(logo, logo, mask=mask)
plt.imshow(logo_fg) # Display the foreground logo
plt.show() # Show the plot
# 5. Add them together and put them back in the main image
dst = cv2.add(main_bg, logo_fg)
main_img[0:rows, 0:cols] = dst
plt.imshow(main_img) # Display the main image with the logo
plt.show() # Show the plot


"""
What is a "Mask"?
In computer vision, a mask is a strictly binary image. It is a 2D grid where every single pixel is exactly one of two values:

255 (Pure White): Means "True", "Yes", or "Keep this part".

0 (Pure Black): Means "False", "No", or "Ignore this part".

A mask cannot have colors, and it cannot have gray shadows. It is purely an ON/OFF switch for every pixel.
"""

"""
The Pipeline: Grayscale $\rightarrow$ Threshold $\rightarrow$ MaskWhen you load a logo image (like a company logo on a white background),
 it is a 3D RGB array. It has colors, shadows, and anti-aliasing (smooth edges). You cannot use an RGB image as a mask because OpenCV's bitwise functions 
 don't know what to do with "red" or "blue" logic.To create a mask from a logo, you have to strip away the color and force it into an ON/OFF state.
 Step 1: Grayscale (cv2.cvtColor)You flatten the 3D colored image into a 2D image. Now, instead of Red, Green, and Blue, every pixel just has a single brightness value from 0 (Black) to 255 (White).
 Step 2: Threshold (cv2.threshold)Grayscale still has smooth gray shadows. A mask cannot have gray. Thresholding draws a hard line in the sand. You pick a number (like 240). 
 You tell OpenCV: "If a pixel's brightness is higher than 240, crush it to pure White (255). If it is lower, crush it to pure Black (0)."The Result (The Mask)You now have a perfect binary silhouette of your logo.
"""