import numpy as np
import cv2
import random

from Cryptodome.Cipher import AES
from Cryptodome.Util.Padding import unpad

def select_high_variance_blocks(image, block_size=8):
    """
    Select regions with high variance (high contrast) for watermark extraction.
    Divide the image into blocks and select those with the highest variance.
    """
    blocks = []
    height, width = image.shape
    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            block = image[i:i + block_size, j:j + block_size]
            block_variance = np.var(block)
            blocks.append((block, (i, j), block_variance))

    # Sort blocks by variance in descending order
    blocks.sort(key=lambda x: x[2], reverse=True)
    return blocks[:len(blocks)//4]  # Select top 25% blocks

def apply_dct(block):
    """
    Apply DCT on a block (8x8) and return the DCT coefficients.
    """
    from scipy.fft import dct
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

def decrypt_patient_id(encrypted_id, iv):
    """
    Decrypt the patient ID using AES decryption.
    """
    key = b'Sixteen byte key'
    cipher = AES.new(key, AES.MODE_CBC, iv=iv)
    decrypted_id = unpad(cipher.decrypt(encrypted_id), AES.block_size)
    return decrypted_id.decode()

def extract_watermark(watermarked_image, watermark_key, block_size=8, watermark_length=128):
    """
    Extract the watermark (encrypted patient ID) from the watermarked image.
    
    Args:
        watermarked_image (numpy.ndarray): Grayscale watermarked image
        watermark_key (int): Key used during watermark embedding
        block_size (int, optional): Size of image blocks. Defaults to 8.
        watermark_length (int, optional): Length of watermark in bits. Defaults to 128.
    
    Returns:
        str: Extracted watermark (encrypted patient ID)
    """
    # Select high variance blocks
    blocks = select_high_variance_blocks(watermarked_image, block_size)
    
    # Extract binary watermark
    binary_watermark = []
    random.seed(watermark_key)  # Ensure the same sequence of blocks
    for block, (i, j), _ in blocks:
        # Apply DCT on the block
        dct_block = apply_dct(block)

        # Extract watermark bit from the low-frequency DCT coefficient
        if len(binary_watermark) < watermark_length:
            extracted_bit = 1 if dct_block[0, 0] > 5 else 0  # Threshold to detect watermark
            binary_watermark.append(str(extracted_bit))

    # Ensure we have extracted enough bits
    if len(binary_watermark) < watermark_length:
        raise ValueError(f"Could not extract full watermark. Extracted only {len(binary_watermark)} bits.")

    # Convert binary watermark to encrypted patient ID
    binary_watermark = ''.join(binary_watermark)
    extracted_id = bytes(int(binary_watermark[i:i + 8], 2) for i in range(0, len(binary_watermark), 8))
    
    return extracted_id

def extract_watermark_from_image(image_path, watermark_key):
    """
    Extract watermark from a given image file.
    
    Args:
        image_path (str): Path to the watermarked image
        watermark_key (int): Key used during watermark embedding
    
    Returns:
        str: Decrypted patient ID
    """
    # Read the watermarked image in grayscale
    watermarked_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if watermarked_image is None:
        raise ValueError(f"Could not read image from {image_path}")
    
    # Extract the encrypted patient ID
    encrypted_id = extract_watermark(watermarked_image, watermark_key)
    
    # Generate the initialization vector (IV) 
    # Note: In a real-world scenario, you'd need a secure way to transmit or store the IV
    iv = b'\x00' * 16  # Placeholder IV, replace with actual IV used during encryption
    
    # Decrypt the patient ID
    decrypted_id = decrypt_patient_id(encrypted_id, iv)
    
    return decrypted_id

def main():
    # Example usage
    try:
        # Path to the watermarked image
        watermarked_image_path = 'watermarked_image.png'
        
        # The same watermark key used during embedding
        watermark_key = 12345  # Replace with the actual key used
        
        # Extract and decrypt the watermark
        patient_id = extract_watermark_from_image(watermarked_image_path, watermark_key)
        
        print("Extracted Patient ID:", patient_id)
    
    except Exception as e:
        print(f"Error extracting watermark: {e}")

if __name__ == "__main__":
    main()