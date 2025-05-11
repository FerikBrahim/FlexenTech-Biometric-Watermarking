import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import dct, idct

# ------------------------------------------------------------------------------
# Helper functions (from our proposed approach)
# ------------------------------------------------------------------------------

def apply_dct(block):
    """Apply 2D DCT to a block."""
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

def apply_idct(dct_block):
    """Apply 2D inverse DCT to a block."""
    return idct(idct(dct_block.T, norm='ortho').T, norm='ortho')

def generate_logistic_indices(num_indices, x0, r_log, max_index):
    """
    Generate `num_indices` random integer positions in [0, max_index-1]
    using the Logistic Map.
    """
    x = x0
    indices = []
    for _ in range(num_indices):
        x = r_log * x * (1.0 - x)  # logistic iteration
        i = int(np.floor(x * max_index))
        indices.append(i)
    return indices

def select_high_variance_blocks(image, block_size=8, top_ratio=0.25):
    """
    Divide the image into non-overlapping blocks and compute variance for each.
    Returns a list of ((i, j), variance) for the top 'top_ratio' blocks.
    """
    h, w = image.shape
    blocks_info = []
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = image[i:i+block_size, j:j+block_size]
            if block.shape[0] < block_size or block.shape[1] < block_size:
                continue
            variance = np.var(block)
            blocks_info.append(((i, j), variance))
    blocks_info.sort(key=lambda x: x[1], reverse=True)
    num_top_blocks = int(len(blocks_info) * top_ratio)
    return blocks_info[:num_top_blocks]

# ------------------------------------------------------------------------------
# Figure 3: DCT Coefficient Matrix (Before and After Watermark Embedding)
# ------------------------------------------------------------------------------
def figure_3(image_path="original.png", block_coord=(0, 0), block_size=8, Q=5, watermark_bit=1):
    """
    Loads the image, selects a block (default top-left block), computes DCT,
    performs a simple QIM-based embedding of one bit, and displays the coefficient
    matrices before and after watermark embedding. Saves the figure.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Image not found")
    image = image.astype(np.float32)
    
    top, left = block_coord
    block = image[top:top+block_size, left:left+block_size]
    
    # Compute original DCT coefficients
    dct_original = apply_dct(block)
    
    flat_dct = dct_original.flatten()
    flat_dct_mod = flat_dct.copy()
    
    # For demonstration, embed at fixed index (index 0)
    idx = 0
    C = flat_dct_mod[idx]
    u = int(round(C / Q))
    if (u % 2) != watermark_bit:
        u = u + (1 if watermark_bit == 1 else -1)
    C_new = u * Q
    flat_dct_mod[idx] = C_new
    
    dct_modified = flat_dct_mod.reshape((block_size, block_size))
    block_modified = apply_idct(dct_modified)
    
    # Plot side-by-side DCT coefficient matrices
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    im0 = axs[0].imshow(dct_original, cmap='jet')
    axs[0].set_title("Original DCT Coefficient Matrix")
    plt.colorbar(im0, ax=axs[0])
    
    im1 = axs[1].imshow(dct_modified, cmap='jet')
    axs[1].set_title("Modified DCT Coefficient Matrix\n(after QIM embedding)")
    plt.colorbar(im1, ax=axs[1])
    
    plt.suptitle("Figure 3: DCT Coefficient Matrix Before and After Watermark Embedding")
    plt.tight_layout()
    # Save the figure
    plt.savefig("figure3_dct_comparison.png", dpi=300)
    plt.show()

# ------------------------------------------------------------------------------
# Figure 4: Logistic Map Sequence
# ------------------------------------------------------------------------------
def figure_4(x0=0.35, r_log=3.9, num_iterations=64):
    """
    Generates and plots the logistic map sequence to illustrate its chaotic nature.
    Saves the figure.
    """
    x = x0
    sequence = []
    for _ in range(num_iterations):
        x = r_log * x * (1.0 - x)
        sequence.append(x)
    
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, num_iterations + 1), sequence, marker='o', linestyle='-')
    plt.title("Figure 4: Logistic Map Sequence")
    plt.xlabel("Iteration")
    plt.ylabel("x (value)")
    plt.grid(True)
    plt.tight_layout()
    # Save the figure
    plt.savefig("figure4_logistic_map.png", dpi=300)
    plt.show()

# ------------------------------------------------------------------------------
# Figure 5: Heatmap of Variance Values with High-Variance Blocks Highlighted
# ------------------------------------------------------------------------------
def figure_5(image_path="original.png", block_size=8, top_ratio=0.25):
    """
    Computes variance for each block of the image and creates a heatmap.
    Also highlights the blocks selected for watermark embedding.
    Saves the heatmap and the annotated host image.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Image not found")
    image = image.astype(np.float32)
    h, w = image.shape
    
    variance_map = np.zeros((h // block_size, w // block_size))
    block_positions = []
    
    for i in range(0, h, block_size):
        row = i // block_size
        for j in range(0, w, block_size):
            col = j // block_size
            block = image[i:i+block_size, j:j+block_size]
            if block.shape[0] < block_size or block.shape[1] < block_size:
                continue
            var_val = np.var(block)
            variance_map[row, col] = var_val
            block_positions.append(((i, j), var_val))
    
    block_positions.sort(key=lambda x: x[1], reverse=True)
    num_top = int(len(block_positions) * top_ratio)
    top_blocks = block_positions[:num_top]
    
    # Plot heatmap of variance values
    plt.figure(figsize=(8, 6))
    plt.imshow(variance_map, cmap='hot', interpolation='nearest')
    plt.title("Figure 5: Heatmap of Block Variance Values")
    plt.xlabel("Block Column Index")
    plt.ylabel("Block Row Index")
    plt.colorbar(label="Variance")
    plt.tight_layout()
    plt.savefig("figure5_variance_heatmap.png", dpi=300)
    plt.show()
    
    # Overlay rectangles on the original image to show selected blocks
    image_color = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    for (i, j), _ in top_blocks:
        cv2.rectangle(image_color, (j, i), (j+block_size, i+block_size), (255, 0, 0), 2)
    
    plt.figure(figsize=(8, 6))
    plt.imshow(cv2.cvtColor(image_color, cv2.COLOR_BGR2RGB))
    plt.title("Host Image with High-Variance Blocks Highlighted")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("figure5_host_high_variance.png", dpi=300)
    plt.show()

# ------------------------------------------------------------------------------
# Run all figures and save images
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # Generate and save Figure 3
    figure_3(image_path="original.png", block_coord=(0, 0), block_size=8, Q=5, watermark_bit=1)
    
    # Generate and save Figure 4
    figure_4(x0=0.35, r_log=3.9, num_iterations=64)
    
    # Generate and save Figure 5
    figure_5(image_path="original.png", block_size=8, top_ratio=0.25)
