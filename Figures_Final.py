import cv2
import numpy as np
import os
from scipy.fftpack import dct, idct

# Ensure output directory exists
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# FlexenTech Permutation Encryption
def flexentech_permutation_encryption(ePi, key):
    np.random.seed(key)
    perm = np.random.permutation(len(ePi))
    encrypted_ePi = ''.join([ePi[i] for i in perm])
    return encrypted_ePi, perm

# Variance-Based Region Selection
def variance_based_region_selection(image, block_size=8, threshold=100):
    h, w = image.shape
    roi_mask = np.zeros((h, w), dtype=np.uint8)
    for i in range(0, h-block_size+1, block_size):
        for j in range(0, w-block_size+1, block_size):
            block = image[i:i+block_size, j:j+block_size]
            if block.var() > threshold:
                roi_mask[i:i+block_size, j:j+block_size] = 1
    return roi_mask

# Apply DCT
def apply_dct(block):
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

# Apply inverse DCT
def apply_idct(block):
    return idct(idct(block.T, norm='ortho').T, norm='ortho')

# Logistic Map for Random Location Generation
def logistic_map(seed, size):
    x = seed
    locations = []
    for _ in range(size):
        x = 3.99 * x * (1 - x)
        locations.append(x)
    return np.array(locations)

# QIM embedding
def qim_embed(dct_coeff, bit, delta=10):
    return delta * (np.round(dct_coeff/delta) + (0.5 if bit == '1' else 0))

# Main watermark embedding function
def watermark_embedding(image_path, ePi, key, output_dir):
    ensure_dir(output_dir)

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    h, w = image.shape

    # Step 1: Encrypt ePi
    encrypted_ePi, perm = flexentech_permutation_encryption(ePi, key)
    print("Encrypted ePi:", encrypted_ePi)

    # Step 2: Select ROI
    roi_mask = variance_based_region_selection(image)
    cv2.imwrite(os.path.join(output_dir, "roi_mask.png"), roi_mask * 255)

    # Step 3: Apply DCT on ROI blocks
    dct_image = np.zeros_like(image, dtype=np.float32)
    for i in range(0, h-7, 8):
        for j in range(0, w-7, 8):
            if roi_mask[i, j] == 1:
                block = image[i:i+8, j:j+8]
                dct_block = apply_dct(block)
                dct_image[i:i+8, j:j+8] = dct_block
    # Visualization of DCT (log scale)
    cv2.imwrite(os.path.join(output_dir, "dct_image.png"), np.log(np.abs(dct_image)+1)*10)

    # Step 4: Logistic Map for embedding locations
    num_bits = len(encrypted_ePi)
    logistic_seq = logistic_map(0.5, num_bits)
    indices = np.argsort(logistic_seq)

    # Get embedding block coordinates
    roi_indices = np.argwhere(roi_mask[::8, ::8]==1)
    if len(roi_indices) < num_bits:
        raise ValueError("Not enough ROI blocks to embed the entire watermark.")
    selected_blocks = roi_indices[indices % len(roi_indices)]

    # Save Randomly selected DCT coefficients visualization
    rand_dct_vis = np.zeros_like(image, dtype=np.uint8)
    for idx in selected_blocks:
        y, x = idx*8
        rand_dct_vis[y:y+8, x:x+8] = 255
    cv2.imwrite(os.path.join(output_dir, "random_dct_blocks.png"), rand_dct_vis)

    # Step 5: Embed encrypted bits via QIM
    for (block_idx, bit) in zip(selected_blocks, encrypted_ePi):
        y, x = block_idx*8
        dct_block = dct_image[y:y+8, x:x+8]
        dct_block[4,4] = qim_embed(dct_block[4,4], bit)
        dct_image[y:y+8, x:x+8] = dct_block

    # Step 6: Apply inverse DCT to reconstruct watermarked image
    watermarked_image = np.copy(image)
    for i in range(0, h-7, 8):
        for j in range(0, w-7, 8):
            if roi_mask[i,j] == 1:
                idct_block = apply_idct(dct_image[i:i+8,j:j+8])
                watermarked_image[i:i+8, j:j+8] = np.clip(idct_block,0,255)

    cv2.imwrite(os.path.join(output_dir, "watermarked_image.png"), watermarked_image)

    print(f"Images saved to {output_dir}")

# Example usage
if __name__ == "__main__":
    image_path = "X-Ray.png"   # Path to your original image
    ePi = "PatientID12345"             # Patient identifier
    key = 42                           # Encryption key
    output_dir = "output_images"       # Output directory
    watermark_embedding(image_path, ePi, key, output_dir)
