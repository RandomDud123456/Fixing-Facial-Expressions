import cv2
import os
import dlib
import math

def calcDistance(facial_key_points, n1, n2):
    point1 = facial_key_points[n1 - 1]
    point2 = facial_key_points[n2 - 1]
    x1, y1 = point1
    x2, y2 = point2
    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return distance

input_folder = "./DifferentEmotionSmallDataset"

image_filenames = os.listdir(input_folder)

image_paths = [os.path.join(input_folder, filename) for filename in image_filenames if filename.lower().endswith('.jpg')]

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

lip_widths = []
left_eye_lower_corner_and_lips_left_corner_dists = []
left_eye_widths = []



for img in image_paths:
    image = cv2.imread(img)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    for i, face in enumerate(faces):
        face_landmarks = []
        landmarks = predictor(gray, face)
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            face_landmarks.append((x, y))
        left_eye_height = calcDistance(face_landmarks, 38, 42)
        left_eye_width = calcDistance(face_landmarks, 37, 40)
        right_eye_height = calcDistance(face_landmarks, 45, 47)
        right_eye_width = calcDistance(face_landmarks, 43, 46)
        left_eyebrow_width = calcDistance(face_landmarks, 18, 22)
        right_eyebro_width = calcDistance(face_landmarks, 23, 27)
        lip_width = calcDistance(face_landmarks, 61, 65)
        left_eye_upper_corner_and_left_eyebrow_center_dist = calcDistance(face_landmarks, 20, 38)
        right_eye_upper_corner_and_right_eyebrow_center_dist = calcDistance(face_landmarks, 25, 45)
        nose_center_and_lips_center_dist = calcDistance(face_landmarks, 34, 52)
        left_eye_lower_corner_and_lips_left_corner_dist = calcDistance(face_landmarks, 42, 49)
        right_eye_lower_corner_and_lips_right_corner_dist = calcDistance(face_landmarks, 47, 55)
        lip_widths.append((img, lip_width, face_landmarks))
        left_eye_widths.append((img, left_eye_width, face_landmarks))
        left_eye_lower_corner_and_lips_left_corner_dists.append((img, left_eye_lower_corner_and_lips_left_corner_dist, face_landmarks))

# Find the image with the highest lip width
max_lip_width_image = max(lip_widths, key=lambda x: x[1])
# Find the image with the highest left_eye_lower_corner_and_lips_left_corner_dist
max_left_eye_lower_corner_and_lips_left_corner_dist_image = min(left_eye_lower_corner_and_lips_left_corner_dists, key=lambda x: x[1])
# Find the image with the highest left eye width
max_left_eye_width_image = max(left_eye_widths, key=lambda x:x[1])


# Load the neutral image and create a duplicate for patching
neutral_image_path = "DifferentEmotionSmallDataset/split_image_1.jpg"  # Replace with the actual path
neutral_image = cv2.imread(neutral_image_path)
duplicate_neutral_image = neutral_image.copy()
duplicate_neutral_image_2 = neutral_image.copy()
duplicate_neutral_image_3 = neutral_image.copy()
duplicate_neutral_image_common = neutral_image.copy()

# Extract the mouth region from the image with the highest lip width
mouth_landmarks = max_lip_width_image[2][48:60]
x_min = min(point[0] for point in mouth_landmarks)
x_max = max(point[0] for point in mouth_landmarks)
y_min = min(point[1] for point in mouth_landmarks)
y_max = max(point[1] for point in mouth_landmarks)
max_lip_width_image_path = max_lip_width_image[0]
max_lip_width_image_array = cv2.imread(max_lip_width_image_path)
mouth_roi = max_lip_width_image_array[y_min:y_max, x_min:x_max]

# Extract the mouth region from the image with the highest left_eye_lower_corner_and_lips_left_corner_dist
# mouth_landmarks_2 = max_left_eye_lower_corner_and_lips_left_corner_dist_image[2][48:60] + max_left_eye_lower_corner_and_lips_left_corner_dist_image[2][29:36]
mouth_landmarks_2 = max_left_eye_lower_corner_and_lips_left_corner_dist_image[2][48:60]
x_min_2 = min(point[0] for point in mouth_landmarks_2)
x_max_2 = max(point[0] for point in mouth_landmarks_2)
y_min_2 = min(point[1] for point in mouth_landmarks_2)
y_max_2 = max(point[1] for point in mouth_landmarks_2)
max_left_eye_lower_corner_and_lips_left_corner_dist_image_path = max_left_eye_lower_corner_and_lips_left_corner_dist_image[0]
max_left_eye_lower_corner_and_lips_left_corner_dist_image_array = cv2.imread(max_left_eye_lower_corner_and_lips_left_corner_dist_image_path)
mouth_roi_2 = max_left_eye_lower_corner_and_lips_left_corner_dist_image_array[y_min_2:y_max_2, x_min_2:x_max_2]

# Extract the mouth region from the image with the highest left eye width
# eye_landmarks = max_left_eye_width_image[2][18:48]
eye_landmarks = max_left_eye_width_image[2][18:27] + max_left_eye_width_image[2][37:48]

x_min_3 = min(point[0] for point in eye_landmarks)
x_max_3 = max(point[0] for point in eye_landmarks)
y_min_3 = min(point[1] for point in eye_landmarks)
y_max_3 = max(point[1] for point in eye_landmarks)
max_left_eye_width_image_path = max_left_eye_width_image[0]
max_left_eye_width_image_array = cv2.imread(max_left_eye_width_image_path)
eye_roi = max_left_eye_lower_corner_and_lips_left_corner_dist_image_array[y_min_3:y_max_3, x_min_3:x_max_3]


# Paste the mouth ROI onto the duplicate neutral image
duplicate_neutral_image[y_min:y_max, x_min:x_max] = mouth_roi

# Paste the mouth ROI 2 onto the duplicate neutral image 2
duplicate_neutral_image_2[y_min_2:y_max_2, x_min_2:x_max_2] = mouth_roi_2

# Paste the left eye ROI onto the duplicate neutral image 3
duplicate_neutral_image_3[y_min_3:y_max_3, x_min_3:x_max_3] = eye_roi

# Paste everything on single photo
duplicate_neutral_image_common[y_min_2:y_max_2, x_min_2:x_max_2] = mouth_roi_2
duplicate_neutral_image_common[y_min_3:y_max_3, x_min_3:x_max_3] = eye_roi

# Save or display the resulting image
cv2.imwrite("patched_image.jpg", duplicate_neutral_image)
cv2.imwrite("patched_image2.jpg", duplicate_neutral_image_2)
cv2.imwrite("patched_image3.jpg", duplicate_neutral_image_3)
cv2.imwrite("patched_image_merged.jpg", duplicate_neutral_image_common)
# cv2.imshow("Resulting Image", duplicate_neutral_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
