import numpy as np
import cv2 # OpenCV is now required
from typing import List, Tuple

from .var import calculate_variance

class PatchExtractor:
    """
    A class to extract the top K most variant patches from a full-resolution image.
    
    V2: Includes a pre-processing step to find the breast tissue bounding box,
    making the process more efficient and robust by ignoring background space.
    """
    def __init__(self, patch_size: int = 128, stride: int = 64, k: int = 5):
        if not all(isinstance(i, int) and i > 0 for i in [patch_size, stride, k]):
            raise ValueError("patch_size, stride, and k must be positive integers.")
        self.patch_size = patch_size
        self.stride = stride
        self.k = k
        print(f"PatchExtractor initialized with: size={self.patch_size}, stride={self.stride}, k={self.k}")

    def _find_tissue_bounding_box(self, image: np.ndarray) -> Tuple[int, int, int, int]:
        """
        Finds the bounding box of the largest contour, assumed to be the breast tissue.
        
        Args:
            image (np.ndarray): The full grayscale image (uint8 or uint16).
        
        Returns:
            A tuple (x, y, w, h) representing the bounding box.
        """
        # 1. Threshold the image to create a binary mask.
        # We use a low threshold to make sure we capture all faint tissue.
        # OTSU's method can also be effective here.
        _, binary_mask = cv2.threshold(image, 25, 255, cv2.THRESH_BINARY)
        
        # Ensure mask is uint8, as findContours requires it
        if binary_mask.dtype != np.uint8:
            binary_mask = cv2.convertScaleAbs(binary_mask, alpha=(255.0/binary_mask.max()))

        # 2. Find all contours in the binary mask.
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            # If no contours are found, return the whole image area
            return (0, 0, image.shape[1], image.shape[0])

        # 3. Find the contour with the largest area.
        largest_contour = max(contours, key=cv2.contourArea)

        # 4. Get the bounding rectangle for the largest contour.
        return cv2.boundingRect(largest_contour)

    def _pad_patches(self, patch_list: List[np.ndarray]) -> np.ndarray:
        """Pads the list of patches with black images to ensure length K."""
        # This function remains the same as before
        num_found = len(patch_list)
        if self.k - num_found > 0:
            black_patch = np.zeros((self.patch_size, self.patch_size), dtype=np.float32)
            for _ in range(self.k - num_found):
                patch_list.append(black_patch)
        return np.array(patch_list, dtype=np.float32)

    def extract(self, full_image: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """
        Extracts the top K patches from within the breast tissue area.
        
        Returns:
            A tuple containing:
            - np.ndarray: The array of top K patches.
            - Tuple[int, int, int, int]: The bounding box (x,y,w,h) that was used.
        """
        if full_image.ndim != 2:
            raise ValueError("Input image must be a 2D grayscale NumPy array.")

        # NEW STEP 1: Find the bounding box of the breast tissue
        bx, by, bw, bh = self._find_tissue_bounding_box(full_image)
        
        # Crop the image to the bounding box for efficient processing
        tissue_image = full_image[by:by+bh, bx:bx+bw]

        # Normalize the tissue-only image
        if tissue_image.dtype == np.uint16:
            img_norm = tissue_image.astype(np.float32) / 65535.0
        else:
            img_norm = tissue_image.astype(np.float32) / 255.0

        height, width = img_norm.shape
        if height < self.patch_size or width < self.patch_size:
            print("Warning: Tissue area is smaller than patch size. Padding.")
            return self._pad_patches([]), (bx, by, bw, bh)

        scored_patches = []
        for y in range(0, height - self.patch_size + 1, self.stride):
            for x in range(0, width - self.patch_size + 1, self.stride):
                patch = img_norm[y:y+self.patch_size, x:x+self.patch_size]
                variance = calculate_variance(patch)
                scored_patches.append((variance, patch))
        
        if not scored_patches:
            print("Warning: No patches generated from the tissue area.")
            return self._pad_patches([]), (bx, by, bw, bh)

        scored_patches.sort(key=lambda item: item[0], reverse=True)
        top_patches = [patch for variance, patch in scored_patches[:self.k]]

        return self._pad_patches(top_patches), (bx, by, bw, bh)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import os

    print("\n--- Running PatchExtractor V2 (with Tissue Detection) ---")
    
    # --- UPDATE THIS PATH ---
    IMAGE_FILE_PATH = "/home/beckett/Documents/k0d1ng/pinn-bcd/kaggle/data/cbis-ddsm-breast-cancer-image-dataset/jpeg/1.3.6.1.4.1.9590.100.1.2.1171337510424515733644328272589553771/1-009.jpg" 
    
    if not os.path.exists(IMAGE_FILE_PATH):
        print(f"\n❌ ERROR: Please update 'IMAGE_FILE_PATH' in {__file__}")
    else:
        print(f"Loading image file: {IMAGE_FILE_PATH}")
        real_image = cv2.imread(IMAGE_FILE_PATH, cv2.IMREAD_UNCHANGED)

        # Initialize with a larger patch size for better visualization
        extractor = PatchExtractor(patch_size=256, stride=64, k=5)
    
        print("\nExtracting top 5 patches...")
        top_k_patches, bbox = extractor.extract(real_image)
        
        # --- Visualization ---
        print("Displaying results...")
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('PatchExtractor V2 Results (Tissue Detection)', fontsize=20)

        # Draw the bounding box on the original image for visualization
        x, y, w, h = bbox
        img_with_bbox = cv2.cvtColor(real_image, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(img_with_bbox, (x, y), (x+w, y+h), (0, 255, 0), 10) # Draw a thick green box
        
        axes[0, 0].imshow(img_with_bbox)
        axes[0, 0].set_title(f'Original Image with Bounding Box\nShape: {real_image.shape}')
        axes[0, 0].axis('off')

        for i, patch in enumerate(top_k_patches):
            ax = axes.flatten()[i+1]
            ax.imshow(patch, cmap='gray')
            ax.set_title(f'Top Patch #{i+1}\nVariance: {np.var(patch):.4f}')
            ax.axis('off')
            
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig('patch_extractor_v2_results.png', dpi=300)

        print("\n✅ Real image test completed.")