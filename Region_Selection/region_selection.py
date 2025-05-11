import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import Tuple, List
import os

class RegionSelector:
    def __init__(self, block_size: int = 8, threshold_percentile: int = 75):
        """
        Initialize the Region Selector.
        
        Args:
            block_size: Size of blocks to divide the image (default: 8x8)
            threshold_percentile: Percentile threshold for selecting high std regions (default: 75)
        """
        self.block_size = block_size
        self.threshold_percentile = threshold_percentile
        
    def compute_block_std(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute standard deviation for each block in the image.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Tuple of (std_map, block_positions)
        """
        height, width = image.shape[:2]
        std_map = np.zeros((height // self.block_size, width // self.block_size))
        block_positions = []
        
        for i in range(0, height - self.block_size + 1, self.block_size):
            for j in range(0, width - self.block_size + 1, self.block_size):
                block = image[i:i + self.block_size, j:j + self.block_size]
                std_value = np.std(block)
                std_map[i // self.block_size, j // self.block_size] = std_value
                block_positions.append((i, j))
                
        return std_map, np.array(block_positions)
    
    def select_regions(self, image: np.ndarray) -> Tuple[List[Tuple[int, int]], np.ndarray]:
        """
        Select regions based on standard deviation threshold.
        
        Args:
            image: Input image
            
        Returns:
            Tuple of (selected_positions, visualization)
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image
            
        # Compute std map
        std_map, block_positions = self.compute_block_std(gray_image)
        
        # Select threshold
        threshold = np.percentile(std_map, self.threshold_percentile)
        
        # Find positions above threshold
        selected_indices = np.where(std_map.flatten() >= threshold)[0]
        selected_positions = block_positions[selected_indices]
        
        # Create visualization
        visualization = self.create_visualization(image, selected_positions)
        
        return selected_positions, visualization
    
    def create_visualization(self, image: np.ndarray, selected_positions: np.ndarray) -> np.ndarray:
        """
        Create visualization of selected regions.
        
        Args:
            image: Original image
            selected_positions: List of selected block positions
            
        Returns:
            Visualization image
        """
        visualization = image.copy()
        if len(visualization.shape) == 2:
            visualization = cv2.cvtColor(visualization, cv2.COLOR_GRAY2BGR)
            
        # Draw rectangles around selected regions
        for pos in selected_positions:
            i, j = pos
            cv2.rectangle(visualization, 
                         (j, i), 
                         (j + self.block_size, i + self.block_size),
                         (0, 255, 0), 
                         1)
            
        return visualization

def display_results(original: np.ndarray, visualization: np.ndarray):
    """
    Display original image and visualization side by side.
    
    Args:
        original: Original image
        visualization: Visualization with selected regions
    """
    plt.figure(figsize=(15, 7))
    
    plt.subplot(121)
    plt.title('Original Image')
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB) if len(original.shape) == 3 else original, 
               cmap='gray')
    plt.axis('off')
    
    plt.subplot(122)
    plt.title('Selected Regions (Green Boxes)')
    plt.imshow(cv2.cvtColor(visualization, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def main():
    """
    Main function demonstrating the usage of RegionSelector.
    """
    # Initialize selector
    selector = RegionSelector(block_size=16, threshold_percentile=85)
    
    # Read image
    image = cv2.imread('data_2.png')
    if image is None:
        # Create sample image if no file exists
        image = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
    
    # Select regions and get visualization
    selected_positions, visualization = selector.select_regions(image)
    
    # Display results
    display_results(image, visualization)
    
    print(f"Number of selected regions: {len(selected_positions)}")

if __name__ == "__main__":
    main()