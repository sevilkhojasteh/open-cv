import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread("/Users/sevilkhojasteh/Documents/to_learn/YOLOv8/data/images/young-male-construction-worker-wearing-hardhat-side-view.webp")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert from BGR to RGB for correct color display in Matplotlib
# 1. Standard Average Blur (Kernel Size = (5, 5))
blur_avg = cv2.blur(img, ksize=(5, 5))
plt.imshow(blur_avg) # Display the average blurred image
plt.title("Average Blur (5x5 Kernel)")
plt.axis("off") # Hide axes
plt.show() # Show the plot

# 2. Gaussian Blur (Kernel Size = (5, 5), SigmaX = 0)
blur_gaussian = cv2.GaussianBlur(img, ksize=(5, 5), sigmaX=0)
plt.imshow(blur_gaussian) # Display the Gaussian blurred image
plt.title("Gaussian Blur (5x5 Kernel, SigmaX=0)")
plt.axis("off") # Hide axes
plt.show() # Show the plot

# 3. Median Blur (Kernel Size = 5)
blur_median = cv2.medianBlur(img, ksize=5)
plt.imshow(blur_median) # Display the median blurred image
plt.title("Median Blur (Kernel Size = 5)")
plt.axis("off") # Hide axes
plt.show() # Show the plotsq