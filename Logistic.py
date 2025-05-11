import cv2
import numpy as np
import matplotlib.pyplot as plt

# Logistic map function
def logistic_map(x0, r, iterations):
    locations = []
    x = x0
    for _ in range(iterations):
        x = r * x * (1 - x)
        locations.append(x)
    return locations

# Load medical image
image_path = 'original.png'  # Replace with your medical image file path
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    raise ValueError("Image file not found or invalid format.")

height, width = image.shape

# Logistic map parameters
x0 = 0.1  # Initial value (should be between 0 and 1)
r = 3.9    # Control parameter for chaotic behavior
num_locations = 500  # Number of embedding locations to generate

# Generate logistic map sequence
chaotic_values = logistic_map(x0, r, num_locations)

# Map chaotic values to pixel coordinates within the image dimensions
coordinates = [
    (int(val * height) % height, int(val * width) % width)
    for val in chaotic_values
]

# Visualize the locations on the image
import cv2
for y, x in coordinates:
    cv2.circle(image, (x, y), 1, (255, 255, 255), -1)

# Display the image with watermark locations
cv2.imshow('Watermark Locations', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the resulting image
cv2.imwrite('medical_image_with_locations.png', image)

print(f"Pseudo-random watermark locations generated and visualized on 'medical_image.png'.")

