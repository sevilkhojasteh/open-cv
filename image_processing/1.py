"""
Color Mappings: Why RGB is Terrible for CV
You already know RGB and BGR. But in traditional CV, isolating an object by its RGB color is a nightmare. Why? Because of shadows. If a worker steps into a shadow, their yellow helmet becomes dark brown. The Red, Green, and Blue values all change drastically.

The Solution: HSV (Hue, Saturation, Value)

Hue: The actual color (Yellow, Red, Blue).

Saturation: How "deep" the color is (Pastel vs. Neon).

Value (Lightness): How bright it is (Shadow vs. Direct Sun).

If you convert to HSV, a yellow helmet in the sun and a yellow helmet in the shade have the exact same Hue. Only the 'Value' changes.

Interview Pro-Tip: If asked, "How would you detect a high-vis orange vest without a neural network?" The answer is: "Convert the image from BGR to HSV, and create a mask targeting the specific Hue range of high-vis orange. This makes the detection robust to changing lighting conditions."
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread("/Users/sevilkhojasteh/Documents/to_learn/YOLOv8/data/images/nennieinszweidrei-cat-5628953_1920.jpg")

# Convert BGR to HSV
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# 2. Define the color range for Yellow in HSV
# Hue for yellow is around 20-30. Saturation and Value go from 100 to 255 to ignore dark shadows/white glare.
lower_yellow = np.array([20, 100, 100])
upper_yellow = np.array([30, 255, 255])

# 3. Create the Mask (Binary Image)
# This turns every pixel inside the yellow range PURE WHITE, and everything else PURE BLACK.
mask = cv2.inRange(hsv_img, lower_yellow, upper_yellow)

# 4. Apply the Mask to the original image using Bitwise AND
# "Keep the original image pixels, but ONLY where the mask is white"
yellow_helmet_only = cv2.bitwise_and(img, img, mask=mask)

cv2.imshow('Mask', mask)
cv2.imshow('Result', yellow_helmet_only)
cv2.waitKey(0)
cv2.destroyAllWindows()
# trying to display the mask and result using Matplotlib