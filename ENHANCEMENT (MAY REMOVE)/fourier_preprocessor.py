"""
Fourier Transform-Based Image Preprocessing (Module 3)
=======================================================

This module uses Fourier Transform for frequency domain filtering
to improve edge detection and reduce noise.

Frequency domain filtering is a classical signal processing technique
that helps remove noise patterns and enhance edges.
"""

import cv2
import numpy as np

class FourierPreprocessor:
    """
    Image preprocessing using Fourier Transform (Module 3).
    Implements high-pass and low-pass filtering in frequency domain.
    """
    
    def __init__(self):
        """Initialize Fourier preprocessor with default parameters."""
        # High-pass filter radius (removes low frequencies = smooth areas)
        self.hp_radius = 30
        
        # Low-pass filter radius (removes high frequencies = noise)
        self.lp_radius = 100
        
    def create_ideal_highpass_filter(self, shape, radius):
        """
        Create ideal high-pass filter in frequency domain.
        Removes low frequencies (smooth areas), keeps edges.
        
        Args:
            shape: Image shape (rows, cols)
            radius: Cutoff radius for filter
            
        Returns:
            High-pass filter mask
        """
        rows, cols = shape
        crow, ccol = rows // 2, cols // 2  # Center
        
        # Create distance matrix from center
        y, x = np.ogrid[:rows, :cols]
        distance = np.sqrt((x - ccol)**2 + (y - crow)**2)
        
        # High-pass: Keep frequencies > radius
        hp_filter = np.ones((rows, cols), dtype=np.float32)
        hp_filter[distance <= radius] = 0
        
        return hp_filter
    
    def create_ideal_lowpass_filter(self, shape, radius):
        """
        Create ideal low-pass filter in frequency domain.
        Removes high frequencies (noise), keeps smooth areas.
        
        Args:
            shape: Image shape (rows, cols)
            radius: Cutoff radius for filter
            
        Returns:
            Low-pass filter mask
        """
        rows, cols = shape
        crow, ccol = rows // 2, cols // 2
        
        # Create distance matrix
        y, x = np.ogrid[:rows, :cols]
        distance = np.sqrt((x - ccol)**2 + (y - crow)**2)
        
        # Low-pass: Keep frequencies < radius
        lp_filter = np.zeros((rows, cols), dtype=np.float32)
        lp_filter[distance <= radius] = 1
        
        return lp_filter
    
    def apply_highpass_filter(self, image, radius=None):
        """
        Apply high-pass filter to enhance edges.
        
        Args:
            image: Grayscale input image
            radius: Optional cutoff radius (uses default if None)
            
        Returns:
            Edge-enhanced image
        """
        if radius is None:
            radius = self.hp_radius
        
        # Convert to float
        img_float = np.float32(image)
        
        # Forward FFT
        dft = cv2.dft(img_float, flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        
        # Create and apply high-pass filter
        hp_filter = self.create_ideal_highpass_filter(image.shape, radius)
        hp_filter = np.stack([hp_filter, hp_filter], axis=-1)  # Two channels for complex
        
        filtered = dft_shift * hp_filter
        
        # Inverse FFT
        f_ishift = np.fft.ifftshift(filtered)
        img_back = cv2.idft(f_ishift)
        img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
        
        # Normalize to 0-255
        cv2.normalize(img_back, img_back, 0, 255, cv2.NORM_MINMAX)
        
        return np.uint8(img_back)
    
    def apply_lowpass_filter(self, image, radius=None):
        """
        Apply low-pass filter to reduce noise.
        
        Args:
            image: Grayscale input image
            radius: Optional cutoff radius
            
        Returns:
            Noise-reduced image
        """
        if radius is None:
            radius = self.lp_radius
        
        img_float = np.float32(image)
        
        # Forward FFT
        dft = cv2.dft(img_float, flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        
        # Create and apply low-pass filter
        lp_filter = self.create_ideal_lowpass_filter(image.shape, radius)
        lp_filter = np.stack([lp_filter, lp_filter], axis=-1)
        
        filtered = dft_shift * lp_filter
        
        # Inverse FFT
        f_ishift = np.fft.ifftshift(filtered)
        img_back = cv2.idft(f_ishift)
        img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
        
        cv2.normalize(img_back, img_back, 0, 255, cv2.NORM_MINMAX)
        
        return np.uint8(img_back)
    
    def bandpass_filter(self, image, low_radius=None, high_radius=None):
        """
        Apply bandpass filter (combination of high-pass and low-pass).
        Removes both low frequencies (smooth areas) and very high frequencies (noise).
        
        Args:
            image: Grayscale input image
            low_radius: Inner radius (high-pass)
            high_radius: Outer radius (low-pass)
            
        Returns:
            Band-pass filtered image
        """
        if low_radius is None:
            low_radius = self.hp_radius
        if high_radius is None:
            high_radius = self.lp_radius
        
        img_float = np.float32(image)
        
        # Forward FFT
        dft = cv2.dft(img_float, flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        
        # Create bandpass filter
        rows, cols = image.shape
        crow, ccol = rows // 2, cols // 2
        
        y, x = np.ogrid[:rows, :cols]
        distance = np.sqrt((x - ccol)**2 + (y - crow)**2)
        
        # Keep frequencies between low_radius and high_radius
        bp_filter = np.zeros((rows, cols), dtype=np.float32)
        bp_filter[(distance > low_radius) & (distance < high_radius)] = 1
        bp_filter = np.stack([bp_filter, bp_filter], axis=-1)
        
        filtered = dft_shift * bp_filter
        
        # Inverse FFT
        f_ishift = np.fft.ifftshift(filtered)
        img_back = cv2.idft(f_ishift)
        img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
        
        cv2.normalize(img_back, img_back, 0, 255, cv2.NORM_MINMAX)
        
        return np.uint8(img_back)
    
    def enhanced_edge_detection(self, frame):
        """
        Perform edge detection with Fourier preprocessing.
        
        Pipeline:
        1. Convert to grayscale
        2. Apply bandpass filter (remove noise and smooth areas)
        3. Apply Canny edge detection
        4. Morphological operations to clean up
        
        Args:
            frame: Input BGR frame
            
        Returns:
            tuple: (edges, preprocessed_gray)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply bandpass filter in Fourier domain
        filtered = self.bandpass_filter(gray, low_radius=20, high_radius=120)
        
        # Additional spatial domain smoothing
        filtered = cv2.bilateralFilter(filtered, 9, 75, 75)
        
        # Canny edge detection on filtered image
        edges = cv2.Canny(filtered, 50, 150)
        
        # Morphological closing to connect broken edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        return edges, filtered


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

def demo_fourier_preprocessing():
    """
    Demonstration of Fourier Transform-based preprocessing.
    
    Shows comparison of:
    1. Raw edge detection (no filtering)
    2. Spatial domain filtering (Gaussian)
    3. Frequency domain filtering (Fourier)
    
    Benefits of Fourier filtering:
    - More selective frequency removal
    - Better noise reduction
    - Preserves important edges
    - Removes periodic noise patterns
    """
    print("=" * 60)
    print("Fourier Transform Preprocessing Demo (Module 3)")
    print("=" * 60)
    
    preprocessor = FourierPreprocessor()
    cap = cv2.VideoCapture(0)
    
    print("\n✓ Fourier preprocessor initialized")
    print("✓ Camera opened")
    print("\nPress 'Q' to quit")
    print("\nWatch how Fourier filtering improves edge detection!\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Method 1: Raw edge detection (no preprocessing)
        edges_raw = cv2.Canny(gray, 50, 150)
        
        # Method 2: Spatial domain (Gaussian blur)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges_spatial = cv2.Canny(blurred, 50, 150)
        
        # Method 3: Frequency domain (Fourier)
        edges_fourier, filtered = preprocessor.enhanced_edge_detection(frame)
        
        # Method 4: High-pass filter only
        highpass = preprocessor.apply_highpass_filter(gray, radius=30)
        
        # Create comparison display
        h, w = frame.shape[:2]
        comparison = np.zeros((h * 2, w * 2), dtype=np.uint8)
        
        # Top-left: Raw edges
        comparison[0:h, 0:w] = edges_raw
        cv2.putText(comparison, "Raw Canny", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2)
        
        # Top-right: Spatial filtering
        comparison[0:h, w:w*2] = edges_spatial
        cv2.putText(comparison, "Gaussian + Canny", (w + 10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2)
        
        # Bottom-left: High-pass filter
        comparison[h:h*2, 0:w] = highpass
        cv2.putText(comparison, "High-Pass (Fourier)", (10, h + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2)
        
        # Bottom-right: Full Fourier pipeline
        comparison[h:h*2, w:w*2] = edges_fourier
        cv2.putText(comparison, "Bandpass + Canny", (w + 10, h + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2)
        
        cv2.imshow("Fourier Preprocessing Comparison", comparison)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    demo_fourier_preprocessing()


# ============================================================================
# INTEGRATION WITH MAIN SYSTEM
# ============================================================================
"""
INTEGRATION: Enhanced Body Contour Detection

Replace simple edge detection with Fourier-enhanced version:

In body_segmentation.py or main pipeline:

    from fourier_preprocessor import FourierPreprocessor
    
    # Initialize
    preprocessor = FourierPreprocessor()
    
    # In processing loop:
    edges, filtered = preprocessor.enhanced_edge_detection(frame)
    
    # Find contours on enhanced edges
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, 
                                    cv2.CHAIN_APPROX_SIMPLE)

BENEFITS:
- Cleaner edge detection
- Better noise reduction
- More accurate contours
- Handles lighting variations better

WHEN TO USE:
- Noisy environments
- Poor lighting conditions
- Complex backgrounds
- Need precise body outline

PERFORMANCE NOTE:
- FFT operations are fast (O(n log n))
- Slight overhead compared to spatial filtering
- Worth it for challenging conditions
"""
