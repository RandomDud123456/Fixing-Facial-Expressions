import cv2
import numpy as np
import os
from match_cropped import CroppedImageMatcher

class ImagePatcher:
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.matcher = CroppedImageMatcher(folder_path)
    
    def create_patched_image(self):
        directory_folder_path, filename_folder_path = os.path.split(self.folder_path)
        original_image_path = os.path.join(self.folder_path, filename_folder_path + ".JPG")
        original_image = cv2.imread(original_image_path)
        
        if original_image is None:
            print("Original image not found.")
            return
    
        # Iterate through each output image to find its matching crop coordinates
        # and then patch it onto the original image.
        for file_name in os.listdir(self.folder_path):
            if file_name.startswith("output_") and file_name.endswith(".jpg"):
                target_file_name = file_name.replace("output_", "")
                output_image_path = os.path.join(self.folder_path, file_name)
                target_image_path = os.path.join(self.folder_path, target_file_name)
                
                # Use CroppedImageMatcher class to get the cropping coordinates
                coords = self.matcher.match_cropped_to_group(original_image_path, target_image_path)
                if coords:
                    x1, y1, x2, y2 = coords
                    # Read the output image to be patched
                    output_image = cv2.imread(output_image_path)
                    # Resize output image to match the patch size
                    output_image_resized = cv2.resize(output_image, (x2 - x1, y2 - y1))
                    # Patch the original image
                    original_image[y1:y2, x1:x2] = output_image_resized
    
        # Save the patched original image
        new_image_path = os.path.join(self.folder_path, "4N7A1641_patched.JPG")
        cv2.imwrite(new_image_path, original_image)
        print(f"Patched image saved to {new_image_path}")

# Example usage
folder_path = "/home/saketh/IIITH Sem 4/DASS/Project/Modular Code/Mini_Dataset/292A2175"  # Make sure to replace this with the actual folder path
patcher = ImagePatcher(folder_path)
patcher.create_patched_image()