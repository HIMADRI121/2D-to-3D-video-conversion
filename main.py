#this code is in ipynb format

import cv2
import numpy as np
from typing import Tuple, Optional
import torch
import torch.nn.functional as F

class VideoTo3DConverter:
    def __init__(self):
        """Initialize the 2D to 3D converter with default parameters"""
        self.depth_estimation_model = None  # Placeholder for a depth estimation model
        
    def generate_depth_map(self, frame: np.ndarray) -> np.ndarray:
        """
        Generate a depth map from a single frame.
        This is a simplified version - in practice, you'd want to use
        a pre-trained deep learning model for better results.
        
        Args:
            frame: Input frame in BGR format
            
        Returns:
            Depth map as grayscale image
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (15, 15), 0)
        
        # Use Sobel operators to detect edges
        sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        
        # Compute gradient magnitude
        gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        
        # Normalize to 0-255 range
        depth_map = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX)
        
        return depth_map.astype(np.uint8)
    
    def create_stereo_pair(self, 
                          frame: np.ndarray, 
                          depth_map: np.ndarray,
                          shift_scale: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create left and right views based on the depth map.
        
        Args:
            frame: Input frame
            depth_map: Corresponding depth map
            shift_scale: Scale factor for pixel shifting
            
        Returns:
            Tuple of left and right view images
        """
        height, width = frame.shape[:2]
        
        # Create displacement maps for left and right views
        displacement = (depth_map.astype(float) * shift_scale).astype(np.float32)
        
        # Create matrices for remapping
        x_map = np.tile(np.arange(width), (height, 1)).astype(np.float32)
        y_map = np.tile(np.arange(height), (width, 1)).T.astype(np.float32)
        
        # Create left view (shift right)
        left_x_map = x_map + displacement
        left_view = cv2.remap(frame, left_x_map, y_map, cv2.INTER_LINEAR)
        
        # Create right view (shift left)
        right_x_map = x_map - displacement
        right_view = cv2.remap(frame, right_x_map, y_map, cv2.INTER_LINEAR)
        
        return left_view, right_view
    
    def create_anaglyph(self, left_view: np.ndarray, right_view: np.ndarray) -> np.ndarray:
        """
        Create a red-cyan anaglyph from stereo pair.
        
        Args:
            left_view: Left eye view
            right_view: Right eye view
            
        Returns:
            Anaglyph image
        """
        # Split the views into color channels
        left_b, left_g, left_r = cv2.split(left_view)
        right_b, right_g, right_r = cv2.split(right_view)
        
        # Create anaglyph (red channel from left, blue/green from right)
        anaglyph = cv2.merge([right_b, right_g, left_r])
        
        return anaglyph
    
    def convert_video(self, 
                     input_path: str, 
                     output_path: str,
                     shift_scale: float = 0.05) -> None:
        """
        Convert an entire video from 2D to 3D.
        
        Args:
            input_path: Path to input video file
            output_path: Path to save output video
            shift_scale: Scale factor for stereo separation
        """
        # Open input video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError("Could not open input video")
            
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Generate depth map
            depth_map = self.generate_depth_map(frame)
            
            # Create stereo pair
            left_view, right_view = self.create_stereo_pair(frame, depth_map, shift_scale)
            
            # Create anaglyph
            anaglyph = self.create_anaglyph(left_view, right_view)
            
            # Write frame
            out.write(anaglyph)
            
        # Release resources
        cap.release()
        out.release()

# Example usage
def main():
    converter = VideoTo3DConverter()
    
    try:
        converter.convert_video(
            input_path='Hostel_2_video.mp4',
            output_path='Output_hostel_2_3d_video.mp4',
            shift_scale=0.05
        )
        print("Conversion completed successfully!")
        
    except Exception as e:
        print(f"Error during conversion: {str(e)}")

if __name__ == "__main__":
    main()
