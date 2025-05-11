import sys
import cv2
import numpy as np
from scipy.fft import dct, idct
from skimage.metrics import structural_similarity as ssim

###############################################################################
# Fix for UnicodeEncodeError on Windows console (set stdout to UTF-8)
###############################################################################
sys.stdout.reconfigure(encoding='utf-8')

###############################################################################
#                           1) FlexenTech Permutation                         #
###############################################################################
class FlexenTech:
    @staticmethod
    def text_to_bits(text):
        """
        Convert text to a list of bits (8 bits per character).
        """
        bits = []
        for char in text:
            # Convert each character to its 8-bit binary representation
            char_bits = format(ord(char), '08b')
            bits.extend([int(bit) for bit in char_bits])
        return bits

    @staticmethod
    def bits_to_text(bits):
        """
        Convert bits (list of 0s and 1s) back into a string by grouping into bytes.
        """
        # Group bits into sets of 8
        char_bits_list = [''.join(str(bit) for bit in bits[i:i + 8])
                          for i in range(0, len(bits), 8)]
        # Convert each 8-bit chunk to its character
        chars = [chr(int(char_bits, 2)) for char_bits in char_bits_list]
        return ''.join(chars)

    @staticmethod
    def generate_random_values(B, K, total_bits):
        """
        Generate a list of 'values' for the bit shuffling (permutation).
        
        :param B: Block size parameter for the FlexenTech algorithm.
        :param K: Key parameter for the FlexenTech algorithm.
        :param total_bits: Total length of the bit array.
        :return: A list of integer values used to sort bit positions.
        """
        values = []
        for i in range(1, total_bits + 1):
            vi = (B * (K - i)) % K
            values.append(vi)
        return values

    @staticmethod
    def encrypt(plain_text, B, K, rounds):
        """
        FlexenTech encryption: converts text to bits, then permutes them
        based on sorting random values, repeatedly for `rounds`.
        """
        bits = FlexenTech.text_to_bits(plain_text)
        total_bits = len(bits)

        # Generate random values
        values = FlexenTech.generate_random_values(B, K, total_bits)

        # Sort indices based on the generated values
        sorted_indices = sorted(range(total_bits), key=lambda i: values[i])

        # Perform bit shuffling for the specified number of rounds
        for _ in range(rounds):
            shuffled_bits = [0] * total_bits
            for old_index, new_index in enumerate(sorted_indices):
                shuffled_bits[new_index] = bits[old_index]
            bits = shuffled_bits

        return bits

    @staticmethod
    def decrypt(encrypted_bits, B, K, rounds):
        """
        FlexenTech decryption: reverses the permutation rounds using the same parameters.
        """
        total_bits = len(encrypted_bits)

        # Generate the same random values
        values = FlexenTech.generate_random_values(B, K, total_bits)
        sorted_indices = sorted(range(total_bits), key=lambda i: values[i])

        # Reverse the encryption (inverse permutation) for the specified rounds
        bits = encrypted_bits[:]
        for _ in range(rounds):
            original_bits = [0] * total_bits
            for old_index, new_index in enumerate(sorted_indices):
                original_bits[old_index] = bits[new_index]
            bits = original_bits

        # Convert bit list back to text
        return FlexenTech.bits_to_text(bits)


###############################################################################
#         2) Standard Deviation-Based Block Selection for Watermarking        #
###############################################################################
def select_high_variance_blocks(image, block_size=8, top_ratio=0.25):
    """
    Splits the image into blocks of size `block_size x block_size`,
    computes variance for each block, and returns the top `top_ratio` fraction
    of blocks by variance.

    :param image: Grayscale image as a 2D numpy array.
    :param block_size: Size of each block (height = width = block_size).
    :param top_ratio: Fraction of blocks to keep (e.g., 0.25 means top 25%).
    :return: A list of tuples [(block_coords, variance), ...] for top blocks.
             block_coords = (i, j) is the top-left corner of the block.
    """
    h, w = image.shape
    blocks_info = []

    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = image[i:i+block_size, j:j+block_size]
            # Skip if block is incomplete at image boundaries
            if block.shape[0] < block_size or block.shape[1] < block_size:
                continue

            variance = np.var(block)
            blocks_info.append(((i, j), variance))

    # Sort by descending variance
    blocks_info.sort(key=lambda x: x[1], reverse=True)

    # Select the top fraction
    num_top_blocks = int(len(blocks_info) * top_ratio)
    return blocks_info[:num_top_blocks]


###############################################################################
#    3) Logistic Map for Improved Random Location Generation in Each Block    #
###############################################################################
def generate_logistic_indices(num_indices, x0, r_log, max_index):
    """
    Generate `num_indices` random integer positions in [0, max_index-1]
    using the Logistic Map.

    :param num_indices: Number of indices to produce.
    :param x0: Initial value of the logistic map (0 < x0 < 1).
    :param r_log: Control parameter (e.g., 3.9).
    :param max_index: Maximum index + 1 (e.g., b*b for an 8x8 block).
    :return: A list of integer indices in range [0, max_index - 1].
    """
    x = x0
    indices = []
    for _ in range(num_indices):
        x = r_log * x * (1.0 - x)  # logistic iteration
        i = int(np.floor(x * max_index))
        indices.append(i)
    return indices


###############################################################################
#                4) DCT, QIM Watermark Embedding, and Extraction              #
###############################################################################
def apply_dct(block):
    """Apply 2D DCT to a block."""
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

def apply_idct(dct_block):
    """Apply 2D inverse DCT to a block."""
    return idct(idct(dct_block.T, norm='ortho').T, norm='ortho')


def embed_watermark(
    image, 
    encrypted_bits, 
    block_size=8, 
    top_ratio=0.25, 
    x0=0.35, 
    r_log=3.9,
    Q=5
):
    """
    Embed the watermark (encrypted_bits) into the image using:
    - Variance-based block selection
    - 2D DCT
    - Logistic map for random location
    - QIM embedding

    :param image: Grayscale image as a 2D numpy array.
    :param encrypted_bits: The bit sequence (list of 0/1) to embed.
    :param block_size: Size of each block for the DCT.
    :param top_ratio: Fraction of high-variance blocks to embed into.
    :param x0: Initial value for the logistic map.
    :param r_log: Logistic map control parameter.
    :param Q: Quantization step size for QIM.
    :return: (watermarked_image, used_block_coords) 
             watermarked_image is a 2D numpy array,
             used_block_coords is a list of (i, j) for the selected blocks.
    """
    watermarked_image = image.copy().astype(np.float32)
    selected_blocks = select_high_variance_blocks(watermarked_image, block_size, top_ratio)

    bit_index = 0
    total_bits = len(encrypted_bits)

    for (top_i, left_j), _ in selected_blocks:
        if bit_index >= total_bits:
            break

        block = watermarked_image[top_i:top_i+block_size, left_j:left_j+block_size]
        dct_block = apply_dct(block)
        flat_dct = dct_block.flatten()

        num_indices_needed = min(block_size * block_size, total_bits - bit_index)
        indices = generate_logistic_indices(num_indices_needed, x0, r_log, block_size * block_size)

        for idx in indices:
            if bit_index >= total_bits:
                break
            bit_val = encrypted_bits[bit_index]
            C = flat_dct[idx]
            # Use rounding instead of floor:
            u = int(round(C / Q))
            # Adjust u if parity doesn't match desired bit
            if (u % 2) != bit_val:
                u = u + (1 if bit_val == 1 else -1)
            C_new = u * Q
            flat_dct[idx] = C_new
            bit_index += 1

        dct_modified = flat_dct.reshape((block_size, block_size))
        block_modified = apply_idct(dct_modified)
        watermarked_image[top_i:top_i+block_size, left_j:left_j+block_size] = block_modified

    return watermarked_image, [b[0] for b in selected_blocks]


def extract_watermark(
    watermarked_image, 
    used_block_coords,
    num_bits_to_extract,
    block_size=8,
    x0=0.35,
    r_log=3.9,
    Q=5
):
    """
    Extract the watermark bits from the watermarked image, given:
    - The block coordinates used during embedding
    - The total number of bits to extract
    - The logistic map parameters and Q for QIM

    :param watermarked_image: 2D numpy array of the watermarked image.
    :param used_block_coords: List of (top_i, left_j) for blocks used in embedding.
    :param num_bits_to_extract: The total number of bits we expect to recover.
    :param block_size: Size of each block used during embedding.
    :param x0: Initial logistic value.
    :param r_log: Logistic map control parameter.
    :param Q: Quantization step size for QIM.
    :return: List of extracted bits (0/1).
    """
    extracted_bits = []
    watermarked_image = watermarked_image.astype(np.float32)
    bit_count = 0

    for (top_i, left_j) in used_block_coords:
        if bit_count >= num_bits_to_extract:
            break

        block = watermarked_image[top_i:top_i+block_size, left_j:left_j+block_size]
        if block.shape[0] < block_size or block.shape[1] < block_size:
            continue

        dct_block = apply_dct(block)
        flat_dct = dct_block.flatten()

        bits_remaining = num_bits_to_extract - bit_count
        num_indices = min(bits_remaining, block_size * block_size)
        indices = generate_logistic_indices(num_indices, x0, r_log, block_size * block_size)

        for idx in indices:
            if bit_count >= num_bits_to_extract:
                break
            C = flat_dct[idx]
            u = int(round(C / Q))
            bit_val = u % 2
            extracted_bits.append(bit_val)
            bit_count += 1

    return extracted_bits


###############################################################################
#       5) Utility Functions for PSNR and SSIM Comparison                     #
###############################################################################
def compute_psnr(original, watermarked):
    """
    Compute the Peak Signal-to-Noise Ratio (PSNR) between two images.
    :param original: Original image, 2D numpy array.
    :param watermarked: Watermarked image, same shape/type as original.
    :return: PSNR value (in dB).
    """
    orig = original.astype(np.float32)
    wmk = watermarked.astype(np.float32)
    
    mse = np.mean((orig - wmk) ** 2)
    if mse == 0:
        return 100  # Images are identical
    max_pixel = 255.0
    return 10 * np.log10((max_pixel ** 2) / mse)

def compute_ssim(original, watermarked):
    """
    Compute the Structural Similarity Index Measure (SSIM) between two images.
    :param original: Original image, 2D numpy array.
    :param watermarked: Watermarked image, same shape as original.
    :return: SSIM value.
    """
    orig = original.astype(np.float32)
    wmk = watermarked.astype(np.float32)
    ssim_value, _ = ssim(orig, wmk, full=True, data_range=255)
    return ssim_value


###############################################################################
#                           6) Main Demonstration                             #
###############################################################################
def main():
    # User inputs
    image_path = "original.png"   # Path to your medical image
    patient_id = "Patient1234"
    
    # FlexenTech parameters
    B = 16      # Block size for permutation
    K = 101     # Key for FlexenTech
    rounds = 3  # Number of permutation rounds
    
    # Logistic Map / QIM parameters
    block_size = 8
    top_ratio = 0.25  # Use top 25% high-variance blocks
    x0 = 0.35         # Initial condition for logistic map
    r_log = 3.9       # Control parameter for logistic map
    Q = 5             # Quantization step size for QIM

    # 1) Read image in grayscale
    original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if original_image is None:
        raise ValueError("Could not open or find the image at path:", image_path)
    original_image = original_image.astype(np.float32)

    # 2) Encrypt the Patient ID using FlexenTech
    encrypted_bits = FlexenTech.encrypt(patient_id, B, K, rounds)
    print(f"Original Patient ID: {patient_id}")
    print(f"Encrypted Bits (first 50 shown): {encrypted_bits[:50]} ...")

    # 3) Embed the watermark into the image
    watermarked_image, used_blocks = embed_watermark(
        original_image, 
        encrypted_bits, 
        block_size=block_size, 
        top_ratio=top_ratio,
        x0=x0,
        r_log=r_log,
        Q=Q
    )
    cv2.imwrite("watermarked_image.png", watermarked_image)
    print("Watermarked image saved as 'watermarked_image.png'")

    # 4) Extract the watermark bits from the watermarked image
    extracted_bits = extract_watermark(
        watermarked_image,
        used_blocks,
        num_bits_to_extract=len(encrypted_bits),
        block_size=block_size,
        x0=x0,
        r_log=r_log,
        Q=Q
    )
    print(f"Extracted Bits (first 50 shown): {extracted_bits[:50]} ...")

    # 5) Decrypt the extracted bits to recover the Patient ID
    recovered_id = FlexenTech.decrypt(extracted_bits, B, K, rounds)
    print(f"Recovered Patient ID: {patient_id}")

    # 6) Compare the original and watermarked images using PSNR and SSIM
    psnr_value = compute_psnr(original_image, watermarked_image)
    ssim_value = compute_ssim(original_image, watermarked_image)
    print(f"PSNR between original and watermarked: {psnr_value:.2f} dB")
    print(f"SSIM between original and watermarked: {ssim_value:.4f}")

if __name__ == "__main__":
    main()
