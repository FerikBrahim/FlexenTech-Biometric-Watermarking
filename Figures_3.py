import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import dct

def apply_dct(block):
    """Apply 2D DCT to an image block."""
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

def select_high_variance_blocks(image, block_size=8, top_ratio=0.25):
    """
    Divide the image into non-overlapping blocks and compute variance for each block.
    Return a list of tuples containing block coordinates and the block's image.
    
    :param image: Input grayscale image as a 2D numpy array.
    :param block_size: Size of each block (height and width).
    :param top_ratio: Fraction of blocks to select (e.g., 0.25 for top 25%).
    :return: List of (coordinate, block) for top high-variance blocks.
    """
    h, w = image.shape
    blocks = []
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = image[i:i+block_size, j:j+block_size]
            if block.shape[0] != block_size or block.shape[1] != block_size:
                continue  # Skip incomplete blocks at edges
            variance = np.var(block)
            blocks.append(((i, j), block, variance))
    
    # Sort blocks based on variance (descending order)
    blocks_sorted = sorted(blocks, key=lambda x: x[2], reverse=True)
    num_top = int(len(blocks_sorted) * top_ratio)
    
    # Return only coordinates and block image (drop variance)
    selected = [(coord, blk) for coord, blk, var in blocks_sorted[:num_top]]
    return selected

def display_dct_of_selected_blocks(image_path="original.png", block_size=8, top_ratio=0.25):
    """
    Load the host image, divide it into blocks, select top high-variance blocks,
    compute DCT of each selected block, and display them in a montage.
    """
    # Load the host image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Image not found at the given path")
    image = image.astype(np.float32)
    
    # Select top high-variance blocks
    selected_blocks = select_high_variance_blocks(image, block_size, top_ratio)
    
    # Compute DCT for each selected block
    dct_blocks = []
    for (i, j), blk in selected_blocks:
        dct_blk = apply_dct(blk)
        dct_blocks.append(dct_blk)
    
    # Determine montage grid size (approximate square)
    num_blocks = len(dct_blocks)
    cols = int(np.ceil(np.sqrt(num_blocks)))
    rows = int(np.ceil(num_blocks / cols))
    
    # Create a figure with subplots for each DCT block
    fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3))
    axes = axes.flatten()  # Flatten in case of multi-dim array of axes
    
    for idx, ax in enumerate(axes):
        if idx < num_blocks:
            im = ax.imshow(dct_blocks[idx], cmap='jet')
            ax.set_title(f"Block {idx+1}")
            ax.axis('off')
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        else:
            ax.axis('off')
    
    fig.suptitle("DCT Coefficient Matrices of Top High-Variance Blocks", fontsize=16)
    plt.tight_layout()
    plt.show()
    
    # Optionally, save the figure
    fig.savefig("dct_selected_blocks.png", dpi=300)

if __name__ == "__main__":
    display_dct_of_selected_blocks(image_path="original.png", block_size=8, top_ratio=0.25)
