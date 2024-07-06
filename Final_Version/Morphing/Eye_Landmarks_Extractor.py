import os
import cv2
import numpy as np
from fastdeploy.vision import facedet
from fastdeploy.runtime import RuntimeOption, ModelFormat
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class EyeLandmarkExtractor:
    def __init__(self, landmarks_file, coordinates_file):
        self.landmarks_file = landmarks_file
        self.coordinates_file = coordinates_file
        self.right_eye = []
        self.left_eye = []
        self.get_eye_landmarks()

    def get_eye_landmarks(self):
        all_landmarks = []
        with open(self.coordinates_file, "r") as cf:
            for line in cf:
                if line.startswith("Landmark"):
                    continue
                if line.strip():
                    x, y, z = map(float, line.strip().split(", "))
                    all_landmarks.append((x, y))

        flag = None
        with open(self.landmarks_file, "r") as lf:
            for line in lf:
                if line.startswith("Right"):
                    flag = "right"
                    continue
                if line.startswith("Left"):
                    flag = "left"
                    continue
                if line.strip():
                    l = list(line.strip().split(","))
                    if flag == "right":
                        for val in l:
                            self.right_eye.append(all_landmarks[int(val)])
                    if flag == "left":
                        for val in l:
                            self.left_eye.append(all_landmarks[int(val)])

class FaceDetector:
    def __init__(self, model_file):
        self.model_file = model_file
        self.retinaface = facedet.RetinaFace(self.model_file, runtime_option=None, model_format=ModelFormat.ONNX)

    def detect_faces(self, image):
        results = self.retinaface.predict(image)
        return results

class FaceLandmarker:
    def __init__(self, model_asset_path):
        base_options = python.BaseOptions(model_asset_path=model_asset_path)
        options = vision.FaceLandmarkerOptions(base_options=base_options, output_face_blendshapes=True,
                                               output_facial_transformation_matrixes=True, num_faces=10,
                                               min_face_detection_confidence=0.5)
        self.detector = vision.FaceLandmarker.create_from_options(options)

class ImageProcessor:
    def __init__(self, image_path, eye_landmarker, face_detector, face_landmarker):
        self.image_path = image_path
        self.eye_landmarker = eye_landmarker
        self.face_detector = face_detector
        self.face_landmarker = face_landmarker

    def process_image(self):
        image = cv2.imread(self.image_path)

        # Combine right and left eye landmarks
        keypoints = self.eye_landmarker.right_eye + self.eye_landmarker.left_eye

        directory, filename = os.path.split(self.image_path)
        results = self.face_detector.detect_faces(image)

        for i in range(len(results.boxes)):
            box = results.boxes[i]
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            face_image = image[y1:y2, x1:x2]
            face_label = f'Face{i + 1}'
            output_face_path = f'Face{i + 1}' + filename
            cv2.imwrite(output_face_path, face_image)

            vis_image = face_image.copy()

            # Convert keypoints to integer coordinates
            keypoints_int = [(int(x * face_image.shape[1]), int(y * face_image.shape[0])) for x, y in keypoints]

            mask = np.zeros(face_image.shape[:2], dtype=np.uint8)
            points = np.array(keypoints_int, dtype=np.int32)
            cv2.fillPoly(mask, [points], (255, 255, 255))
            roi = cv2.bitwise_and(face_image, face_image, mask=mask)

            background = np.zeros(face_image.shape, dtype=np.uint8)
            result = background.copy()
            result[mask == 255] = roi[mask == 255]
            cv2.imwrite("out.jpeg", result)


# Usage
face_detector = FaceDetector("Pytorch_RetinaFace_resnet50-640-640.onnx")
face_landmarker = FaceLandmarker("face_landmarker_v2_with_blendshapes.task")

image_path = "base.jpeg"
filename = image_path
image = cv2.imread(image_path)

results = face_detector.detect_faces(image)

with open("base.txt", "w") as file:
    for i in range(len(results.boxes)):
        box = results.boxes[i]
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        face_image = image[y1:y2, x1:x2]
        face_label = f'Face{i + 1}'
        output_face_path = f'Face{i + 1}' + filename
        face_input = mp.Image.create_from_file(output_face_path)
        detection_result = face_landmarker.detector.detect(face_input)
        face_landmarks_list = detection_result.face_landmarks
        landmark_coordinates = []

        # Loop through the detected faces to get the landmark coordinates
        for idx in range(len(face_landmarks_list)):
            face_landmarks = face_landmarks_list[idx]

            landmarks = [(landmark.x, landmark.y, landmark.z) for landmark in face_landmarks]
            landmark_coordinates.append(landmarks)
        file.write(f"Landmark coordinates for {face_label}:\n")
        for landmarks in landmark_coordinates:
            for landmark in landmarks:
                file.write(f"{landmark[0]}, {landmark[1]}, {landmark[2]}\n")
        file.write("\n")

eye_landmarker = EyeLandmarkExtractor("eye_landmarks.txt", "base.txt")
image_processor = ImageProcessor("base.jpeg", eye_landmarker, face_detector, face_landmarker)
image_processor.process_image()
