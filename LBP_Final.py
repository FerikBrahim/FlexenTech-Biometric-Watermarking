import cv2
import os
import numpy as np

def compute_lbp(binary_image, radius=1, neighbors=8):
    """
    Compute a naive Local Binary Pattern (LBP) for each pixel on a binary image.
    :param binary_image: 2D numpy array (uint8) of the thresholded, binary fingerprint image.
    :param radius: Radius around the center pixel for sampling neighbors.
    :param neighbors: Number of neighbor points used in LBP.
    :return: 2D numpy array (uint8) of the LBP-coded image.
    """
    height, width = binary_image.shape
    lbp_image = np.zeros((height, width), dtype=np.uint8)

    for y in range(radius, height - radius):
        for x in range(radius, width - radius):
            center_intensity = binary_image[y, x]
            code = 0
            for n in range(neighbors):
                angle = 2.0 * np.pi * n / neighbors
                offset_x = int(round(x + radius * np.cos(angle)))
                offset_y = int(round(y - radius * np.sin(angle)))
                neighbor_intensity = binary_image[offset_y, offset_x]
                bit_val = 1 if neighbor_intensity >= center_intensity else 0
                code |= (bit_val << n)
            lbp_image[y, x] = code

    return lbp_image

def main():
    # Read the fingerprint image in grayscale
    watermark_path = "104_5.png"  # Replace with your actual file
    fingerprint_gray = cv2.imread(watermark_path, cv2.IMREAD_GRAYSCALE)
    
    if fingerprint_gray is None:
        raise ValueError("Could not open or find the fingerprint image.")

    # Convert grayscale image to binary (thresholding)
    thresh_val = 128
    _, fingerprint_binary = cv2.threshold(fingerprint_gray, thresh_val, 255, cv2.THRESH_BINARY)

    # -- Correct approach: read file sizes from disk for actual KB values --
    # For demonstration, let's assume the user has the actual files with known sizes:
    # e.g. 77.80 KB original, 3.87 KB after LBP.
    # We'll read the file size from disk. If the user wants exact forced values, set them manually.

    original_file_size_bytes = os.path.getsize(watermark_path)
    original_size_kb = original_file_size_bytes / 1024
    print(f"Original watermark size: {original_size_kb:.2f} KB")

    # Compute the LBP-coded image on the binary version
    lbp_image = compute_lbp(fingerprint_binary, radius=1, neighbors=8)

    # Save the LBP image
    lbp_output_path = "lbp_fingerprint.png"
    cv2.imwrite(lbp_output_path, lbp_image)

    # Check the newly saved file size on disk to get the final watermark size
    final_file_size_bytes = os.path.getsize(lbp_output_path)
    final_size_kb = final_file_size_bytes / 1024
    print(f"Final watermark size (LBP features): {final_size_kb:.2f} KB")

    print(f"LBP-coded fingerprint saved to {lbp_output_path}")

if __name__ == "__main__":
    main()
