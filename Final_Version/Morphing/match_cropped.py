import cv2
import os
import numpy as np

class CroppedImageMatcher:
    def __init__(self, folder_path, threshold=0.8):
        self.folder_path = folder_path
        self.threshold = threshold
    
    def match_cropped_to_group(self, original_image_path, template_image_path):
        # Load the original and template images
        original_image = cv2.imread(original_image_path)
        template_image = cv2.imread(template_image_path)
    
        # Convert images to grayscale
        original_gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        template_gray = cv2.cvtColor(template_image, cv2.COLOR_BGR2GRAY)
    
        # Find the template
        w, h = template_gray.shape[::-1]
        res = cv2.matchTemplate(original_gray, template_gray, cv2.TM_CCOEFF_NORMED)
        
        # Check for matches above the threshold
        loc = np.where(res >= self.threshold)
        points = zip(*loc[::-1])  # Switch x and y coordinates to x, y format
        
        for pt in points:
            # Return the first match's top-left and bottom-right coordinates
            return (pt[0], pt[1], pt[0] + w, pt[1] + h)
        
        # If no match is found, return None
        return None
    
    def match_cropped_images_in_folder(self):
        results = []
        original_image_path = None
        for file_name in os.listdir(self.folder_path):
            if file_name.endswith('.JPG'):
                original_image_path = os.path.join(self.folder_path, file_name)
                break
    
        if original_image_path is None:
            print("Original image not found in the folder.")
            return results
        
        original_image_name = os.path.basename(original_image_path)
        original_image_id = os.path.splitext(original_image_name)[0]
    
        for file_name in os.listdir(self.folder_path):
            if file_name.endswith('.jpg') and not file_name.endswith('frame.jpg') and not file_name.startswith('output') and file_name != original_image_name:
                if file_name.startswith(original_image_id + '_'):
                    cropped_image_path = os.path.join(self.folder_path, file_name)
                    cropped_image_name = os.path.basename(cropped_image_path)
                    result = self.match_cropped_to_group(original_image_path, cropped_image_path)
                    results.append((cropped_image_name, result))
        return results

# Example usage
folder_path = '/home/saketh/IIITH Sem 4/DASS/Project/Modular Code/Mini_Dataset/292A2172'
matcher = CroppedImageMatcher(folder_path)
results = matcher.match_cropped_images_in_folder()
print(results)