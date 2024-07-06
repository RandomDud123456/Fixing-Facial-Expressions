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

for img in image_paths:
    image = cv2.imread(img)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    facial_key_points = {}
    for i, face in enumerate(faces):
        facial_key_points = {}
        face_landmarks = []
        landmarks = predictor(gray, face)
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            face_landmarks.append((x, y))
            cv2.circle(image, (x, y), 1, (0, 255, 0), -1) 
            cv2.putText(image, str(n + 1), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1, cv2.LINE_AA) 

        facial_key_points = face_landmarks
        print(face_landmarks)

        print(f"-----------------Image {i + 1}------------------")

        left_eye_height = calcDistance(facial_key_points, 38, 42)
        left_eye_width = calcDistance(facial_key_points, 37, 40)
        right_eye_height = calcDistance(facial_key_points, 45, 47)
        right_eye_width = calcDistance(facial_key_points, 43, 46)
        left_eyebrow_width = calcDistance(facial_key_points, 18, 22)
        right_eyebro_width = calcDistance(facial_key_points, 23, 27)
        lip_width = calcDistance(facial_key_points, 61, 65)
        left_eye_upper_corner_and_left_eyebrow_center_dist = calcDistance(facial_key_points, 20, 38)
        right_eye_upper_corner_and_right_eyebrow_center_dist = calcDistance(facial_key_points, 25, 45)
        nose_center_and_lips_center_dist = calcDistance(facial_key_points, 34, 52)
        left_eye_lower_corner_and_lips_left_corner_dist = calcDistance(facial_key_points, 42, 49)
        right_eye_lower_corner_and_lips_right_corner_dist = calcDistance(facial_key_points, 47, 55)
        

        print(f"left_eye_height: {left_eye_height}")
        print(f"left_eye_width: {left_eye_width}")
        print(f"right_eye_height: {right_eye_height}")
        print(f"right_eye_width: {right_eye_width}")
        print(f"left_eyebrow_width: {left_eyebrow_width}")
        print(f"right_eyebro_width: {right_eyebro_width}")
        print(f"lip_width: {lip_width}")
        print(f"left_eye_upper_corner_and_left_eyebrow_center_dist: {left_eye_upper_corner_and_left_eyebrow_center_dist}")
        print(f"right_eye_upper_corner_and_right_eyebrow_center_dist: {right_eye_upper_corner_and_right_eyebrow_center_dist}")
        print(f"nose_center_and_lips_center_dist: {nose_center_and_lips_center_dist}")
        print(f"left_eye_lower_corner_and_lips_left_corner_dist: {left_eye_lower_corner_and_lips_left_corner_dist}")
        print(f"right_eye_lower_corner_and_lips_right_corner_dist: {right_eye_lower_corner_and_lips_right_corner_dist}")
        print(f"----------------------END-----------------------")

