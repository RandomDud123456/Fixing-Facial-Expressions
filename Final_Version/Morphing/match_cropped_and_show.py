import cv2
import numpy as np

# Load the original image and the template (patch) image
original_image = cv2.imread('/home/saketh/IIITH Sem 4/DASS/Project/Modular Code/Mini_Dataset/292A2172/292A2172.JPG')
template_image = cv2.imread('/home/saketh/IIITH Sem 4/DASS/Project/Modular Code/Mini_Dataset/292A2172/292A2172_1.jpg')

# Convert images to grayscale
original_gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
template_gray = cv2.cvtColor(template_image, cv2.COLOR_BGR2GRAY)

# Find the template in the original image
w, h = template_gray.shape[::-1]
res = cv2.matchTemplate(original_gray, template_gray, cv2.TM_CCOEFF_NORMED)
threshold = 0.8
loc = np.where(res >= threshold)

# Draw rectangles around the found template
for pt in zip(*loc[::-1]):
    cv2.rectangle(original_image, pt, (pt[0] + w, pt[1] + h), (0,255,255), 2)

# Show the result
cv2.imshow('Detected', original_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# import cv2
# import numpy as np

# def find_template_coordinates(original_image_path, template_image_path, threshold=0.8):
#     # Load the original and template images
#     original_image = cv2.imread(original_image_path)
#     template_image = cv2.imread(template_image_path)

#     # Convert images to grayscale
#     original_gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
#     template_gray = cv2.cvtColor(template_image, cv2.COLOR_BGR2GRAY)

#     # Find the template
#     w, h = template_gray.shape[::-1]
#     res = cv2.matchTemplate(original_gray, template_gray, cv2.TM_CCOEFF_NORMED)
    
#     # Check for matches above the threshold
#     loc = np.where(res >= threshold)
#     points = zip(*loc[::-1])  # Switch x and y coordinates to x, y format
    
#     for pt in points:
#         # Return the first match's top-left and bottom-right coordinates
#         return (pt[0], pt[1], pt[0] + w, pt[1] + h)
    
#     # If no match is found, return None
#     return None

# # Example usage
# original_image_path = '/home/saketh/IIITH Sem 4/DASS/Project/Modular Code/Mini_Dataset/292A2173/292A2173.JPG'
# template_image_path = '/home/saketh/IIITH Sem 4/DASS/Project/Modular Code/Mini_Dataset/292A2173/292A2173_3.jpg'
# coordinates = find_template_coordinates(original_image_path, template_image_path)

# if coordinates:
#     print(f"Template found at: {coordinates}")
# else:
#     print("Template not found.")

