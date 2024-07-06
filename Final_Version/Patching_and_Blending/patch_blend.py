"""
Author: Saketh Reddy Vemula

usage: patch_blend.py [-h] base_path eyes_target_path mouth_target_path eyes_landmarks_path mouth_landmarks_path

Patching and Blending

positional arguments:
  base_path             Path to the base image.
  eyes_target_path      Path to the eyes target image
  mouth_target_path     Path to the mouth target image
  eyes_landmarks_path   Path to the text file containing landmarks of eye
  mouth_landmarks_path  Path to the text file containing landmarks of mouth

options:
  -h, --help            show this help message and exit
"""

import os
import cv2
import numpy as np
from fastdeploy.vision import facedet
from fastdeploy.runtime import RuntimeOption, ModelFormat
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import argparse

class EyeLandmarkExtractor:
    """
    Class for extracting eye landmarks from text files.

    Args:
        landmarks_file (str): Path to the text file containing eye landmark indices.
        coordinates_file (str): Path to the text file containing face landmark coordinates.

    Attributes:
        landmarks_file (str): Path to the text file containing eye landmark indices.
        coordinates_file (str): Path to the text file containing face landmark coordinates.
        right_eye (list): List of (x, y) coordinates for the right eye landmarks.
        left_eye (list): List of (x, y) coordinates for the left eye landmarks.

    Methods:
        get_eye_landmarks(): Extracts the right and left eye landmark coordinates from the files.
    """
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

class MouthLandmarkExtractor:
    """
    Class for extracting mouth landmarks from text files.

    Args:
        landmarks_file (str): Path to the text file containing mouth landmark indices.
        coordinates_file (str): Path to the text file containing face landmark coordinates.

    Attributes:
        landmarks_file (str): Path to the text file containing mouth landmark indices.
        coordinates_file (str): Path to the text file containing face landmark coordinates.
        mouth (list): List of (x, y) coordinates for the mouth landmarks.

    Methods:
        get_mouth_landmarks(): Extracts the mouth landmark coordinates from the files.
    """
    def __init__(self, landmarks_file, coordinates_file):
        self.landmarks_file = landmarks_file
        self.coordinates_file = coordinates_file
        self.mouth = []
        self.get_mouth_landmarks()

    def get_mouth_landmarks(self):
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
                if line.startswith("Mouth"):
                    flag = "mouth"
                    continue
                if line.strip():
                    l = list(line.strip().split(","))
                    if flag == "mouth":
                        for val in l:
                            self.mouth.append(all_landmarks[int(val)])

class FaceDetector:
    """
    Class for detecting faces in an image using the RetinaFace model.

    Args:
        model_file (str): Path to the RetinaFace model file.

    Attributes:
        model_file (str): Path to the RetinaFace model file.
        retinaface (fastdeploy.vision.facedet.RetinaFace): Instance of the RetinaFace model.

    Methods:
        detect_faces(image): Detects faces in the given image.
    """
    def __init__(self, model_file):
        self.model_file = model_file
        self.retinaface = facedet.RetinaFace(self.model_file, runtime_option=None, model_format=ModelFormat.ONNX)

    def detect_faces(self, image):
        results = self.retinaface.predict(image)
        return results

class FaceLandmarker:
    """
    Class for detecting facial landmarks using the MediaPipe FaceLandmarker task.

    Args:
        model_asset_path (str): Path to the FaceLandmarker model asset file.

    Attributes:
        detector (mediapipe.tasks.vision.FaceLandmarker): Instance of the FaceLandmarker task.
    """
    def __init__(self, model_asset_path):
        base_options = python.BaseOptions(model_asset_path=model_asset_path)
        options = vision.FaceLandmarkerOptions(base_options=base_options, output_face_blendshapes=True,
                                               output_facial_transformation_matrixes=True, num_faces=10,
                                               min_face_detection_confidence=0.5)
        self.detector = vision.FaceLandmarker.create_from_options(options)

class EyesImageProcessor:
    """
    Class for processing eye images and blending them with a base image.

    Args:
        eye_base_path (str): Path to the base image.
        eye_target_path (str): Path to the target eye image.
        base_eye_landmarker (EyeLandmarkExtractor): Instance of the EyeLandmarkExtractor for the base image.
        target_eye_landmarker (EyeLandmarkExtractor): Instance of the EyeLandmarkExtractor for the target image.
        face_detector (FaceDetector): Instance of the FaceDetector for detecting faces.
        face_landmarker (FaceLandmarker): Instance of the FaceLandmarker for detecting facial landmarks.
        eye_final_output_path (str): Path to save the final output image with blended eyes.

    Methods:
        process_images(): Processes the base and target eye images, aligns them, and blends the target eye onto the base image.
    """
    def __init__(self, eye_base_path, eye_target_path, base_eye_landmarker, target_eye_landmarker, face_detector, face_landmarker, eye_final_output_path):
        self.eye_base_path = eye_base_path
        self.eye_target_path = eye_target_path
        self.base_eye_landmarker = base_eye_landmarker
        self.target_eye_landmarker = target_eye_landmarker
        self.face_detector = face_detector
        self.face_landmarker = face_landmarker
        self.eye_final_output_path = eye_final_output_path

    def process_images(self):
        base_image = cv2.imread(self.eye_base_path)
        target_image = cv2.imread(self.eye_target_path)

        base_keypoints = self.base_eye_landmarker.right_eye + self.base_eye_landmarker.left_eye
        target_keypoints = self.target_eye_landmarker.right_eye + self.target_eye_landmarker.left_eye

        base_results = self.face_detector.detect_faces(base_image)
        target_results = self.face_detector.detect_faces(target_image)

        if len(base_results.boxes) > 0 and len(target_results.boxes) > 0:
            base_box = base_results.boxes[0]
            target_box = target_results.boxes[0]

            base_x1, base_y1, base_x2, base_y2 = int(base_box[0]), int(base_box[1]), int(base_box[2]), int(base_box[3])
            target_x1, target_y1, target_x2, target_y2 = int(target_box[0]), int(target_box[1]), int(target_box[2]), int(target_box[3])

            base_face_image = base_image[base_y1:base_y2, base_x1:base_x2]
            target_face_image = target_image[target_y1:target_y2, target_x1:target_x2]

            base_keypoints_int = [(int(x * base_face_image.shape[1]), int(y * base_face_image.shape[0])) for x, y in base_keypoints]
            target_keypoints_int = [(int(x * target_face_image.shape[1]), int(y * target_face_image.shape[0])) for x, y in target_keypoints]

            # Select corresponding landmark points for affine transformation
            base_points = np.float32([base_keypoints_int[0], base_keypoints_int[4], base_keypoints_int[8]])
            target_points = np.float32([target_keypoints_int[0], target_keypoints_int[4], target_keypoints_int[8]])

            # Calculate the affine transformation matrix
            affine_matrix = cv2.getAffineTransform(target_points, base_points)

            # Apply the affine transformation to the target face image
            aligned_target_face_image = cv2.warpAffine(target_face_image, affine_matrix, (base_face_image.shape[1], base_face_image.shape[0]))

            base_mask = np.zeros(base_face_image.shape[:2], dtype=np.uint8)
            base_points = np.array(base_keypoints_int, dtype=np.int32)
            cv2.fillPoly(base_mask, [base_points], (255, 255, 255))
            base_eye_patch = cv2.bitwise_and(base_face_image, base_face_image, mask=base_mask)

            target_mask = np.zeros(aligned_target_face_image.shape[:2], dtype=np.uint8)
            target_points = cv2.transform(np.array([target_keypoints_int]), affine_matrix)[0].astype(np.int32)
            cv2.fillPoly(target_mask, [target_points], (255, 255, 255))
            target_eye_patch = cv2.bitwise_and(aligned_target_face_image, aligned_target_face_image, mask=target_mask)

            # Calculate the top-left and bottom-right corner coordinates of the base eye patch
            base_eye_patch_y, base_eye_patch_x = np.where(base_mask > 0)
            base_eye_patch_top_left = (base_eye_patch_x.min(), base_eye_patch_y.min())
            base_eye_patch_bottom_right = (base_eye_patch_x.max(), base_eye_patch_y.max())

            # Extract the region of interest (ROI) from the base face image
            roi_x1, roi_y1 = base_eye_patch_top_left
            roi_x2, roi_y2 = base_eye_patch_bottom_right
            roi_width = roi_x2 - roi_x1 + 1
            roi_height = roi_y2 - roi_y1 + 1
            roi = base_face_image[roi_y1:roi_y1+roi_height, roi_x1:roi_x1+roi_width]

            # Find the non-zero pixel coordinates in the target eye patch
            non_zero_coords = np.nonzero(target_eye_patch)
            min_y, max_y = np.min(non_zero_coords[0]), np.max(non_zero_coords[0])
            min_x, max_x = np.min(non_zero_coords[1]), np.max(non_zero_coords[1])

            # Crop the target eye patch to include only non-zero pixels
            cropped_target_eye_patch = target_eye_patch[min_y:max_y+1, min_x:max_x+1]

            # Update the target mask to match the cropped target eye patch
            cropped_target_mask = target_mask[min_y:max_y+1, min_x:max_x+1]

            # Resize the cropped target eye patch and mask to match the ROI dimensions
            resized_target_eye_patch = cv2.resize(cropped_target_eye_patch, (roi_width, roi_height))
            resized_target_mask = cv2.resize(cropped_target_mask, (roi_width, roi_height))

            # Apply Poisson blending within the ROI
            blended_roi = cv2.seamlessClone(resized_target_eye_patch, roi, resized_target_mask, (roi_width//2, roi_height//2), cv2.NORMAL_CLONE)

            # Update the base face image with the blended ROI
            base_face_image[roi_y1:roi_y1+roi_height, roi_x1:roi_x1+roi_width] = blended_roi

            base_image[base_y1:base_y2, base_x1:base_x2] = base_face_image

            cv2.imwrite(self.eye_final_output_path, base_image)

class MouthImageProcessor:
    """
    Class for processing mouth images and blending them with a base image.

    Args:
        mouth_base_path (str): Path to the base image.
        mouth_target_path (str): Path to the target mouth image.
        base_mouth_landmarker (MouthLandmarkExtractor): Instance of the MouthLandmarkExtractor for the base image.
        target_mouth_landmarker (MouthLandmarkExtractor): Instance of the MouthLandmarkExtractor for the target image.
        face_detector (FaceDetector): Instance of the FaceDetector for detecting faces.
        face_landmarker (FaceLandmarker): Instance of the FaceLandmarker for detecting facial landmarks.
        scale_factor (float): Scale factor for adjusting the size of the target mouth.
        final_output_path (str): Path to save the final output image with blended mouth.

    Methods:
        process_images(): Processes the base and target mouth images, aligns them, and blends the target mouth onto the base image.
    """
    def __init__(self, mouth_base_path, mouth_target_path, base_mouth_landmarker, target_mouth_landmarker, face_detector, face_landmarker, scale_factor, final_output_path):
        self.mouth_base_path = mouth_base_path
        self.mouth_target_path = mouth_target_path
        self.base_mouth_landmarker = base_mouth_landmarker
        self.target_mouth_landmarker = target_mouth_landmarker
        self.face_detector = face_detector
        self.face_landmarker = face_landmarker
        self.scale_factor = scale_factor
        self.final_output_path = final_output_path

    def process_images(self):
        base_image = cv2.imread(self.mouth_base_path)
        target_image = cv2.imread(self.mouth_target_path)

        base_keypoints = self.base_mouth_landmarker.mouth
        target_keypoints = self.target_mouth_landmarker.mouth

        base_results = self.face_detector.detect_faces(base_image)
        target_results = self.face_detector.detect_faces(target_image)
        # cv2.imwrite("target_results.jpeg", target_image)

        if len(base_results.boxes) > 0 and len(target_results.boxes) > 0:
            base_box = base_results.boxes[0]
            target_box = target_results.boxes[0]

            base_x1, base_y1, base_x2, base_y2 = int(base_box[0]), int(base_box[1]), int(base_box[2]), int(base_box[3])
            target_x1, target_y1, target_x2, target_y2 = int(target_box[0]), int(target_box[1]), int(target_box[2]), int(target_box[3])

            base_face_image = base_image[base_y1:base_y2, base_x1:base_x2]
            target_face_image = target_image[target_y1:target_y2, target_x1:target_x2]

            base_keypoints_int = [(int(x * base_face_image.shape[1]), int(y * base_face_image.shape[0])) for x, y in base_keypoints]
            target_keypoints_int = [(int(x * target_face_image.shape[1]), int(y * target_face_image.shape[0])) for x, y in target_keypoints]

            # Select corresponding landmark points for affine transformation
            base_points = np.float32([base_keypoints_int[0], base_keypoints_int[4], base_keypoints_int[8]])
            target_points = np.float32([target_keypoints_int[0], target_keypoints_int[4], target_keypoints_int[8]])

            # Calculate the scaled target points
            scaled_target_points = target_points + self.scale_factor * (base_points - target_points)

            # Calculate the affine transformation matrix
            affine_matrix = cv2.getAffineTransform(target_points, scaled_target_points)

            # Apply the affine transformation to the target face image
            aligned_target_face_image = cv2.warpAffine(target_face_image, affine_matrix, (base_face_image.shape[1], base_face_image.shape[0]))

            base_mask = np.zeros(base_face_image.shape[:2], dtype=np.uint8)
            base_points = np.array(base_keypoints_int, dtype=np.int32)
            cv2.fillPoly(base_mask, [base_points], (255, 255, 255))
            base_mouth_patch = cv2.bitwise_and(base_face_image, base_face_image, mask=base_mask)

            target_mask = np.zeros(aligned_target_face_image.shape[:2], dtype=np.uint8)
            target_points = cv2.transform(np.array([target_keypoints_int]), affine_matrix)[0].astype(np.int32)
            cv2.fillPoly(target_mask, [target_points], (255, 255, 255))
            target_mouth_patch = cv2.bitwise_and(aligned_target_face_image, aligned_target_face_image, mask=target_mask)
            # cv2.imwrite("target_mouth.jpeg", target_mouth_patch)

            # Calculate the top-left and bottom-right corner coordinates of the base mouth patch
            base_mouth_patch_y, base_mouth_patch_x = np.where(base_mask > 0)
            base_mouth_patch_top_left = (base_mouth_patch_x.min(), base_mouth_patch_y.min())
            base_mouth_patch_bottom_right = (base_mouth_patch_x.max(), base_mouth_patch_y.max())

            # Extract the region of interest (ROI) from the base face image
            roi_x1, roi_y1 = base_mouth_patch_top_left
            roi_x2, roi_y2 = base_mouth_patch_bottom_right
            roi_width = roi_x2 - roi_x1 + 1
            roi_height = roi_y2 - roi_y1 + 1
            roi = base_face_image[roi_y1:roi_y1+roi_height, roi_x1:roi_x1+roi_width]

            # Find the non-zero pixel coordinates in the target mouth patch
            non_zero_coords = np.nonzero(target_mouth_patch)
            min_y, max_y = np.min(non_zero_coords[0]), np.max(non_zero_coords[0])
            min_x, max_x = np.min(non_zero_coords[1]), np.max(non_zero_coords[1])

            # Crop the target mouth patch to include only non-zero pixels
            cropped_target_mouth_patch = target_mouth_patch[min_y:max_y+1, min_x:max_x+1]

            # Update the target mask to match the cropped target mouth patch
            cropped_target_mask = target_mask[min_y:max_y+1, min_x:max_x+1]

            # Resize the cropped target mouth patch and mask to match the ROI dimensions
            resized_target_mouth_patch = cv2.resize(cropped_target_mouth_patch, (roi_width, roi_height))
            resized_target_mask = cv2.resize(cropped_target_mask, (roi_width, roi_height))
            # cv2.imwrite("debug.jpeg", target_mouth_patch)

            # Apply Poisson blending within the ROI
            blended_roi = cv2.seamlessClone(resized_target_mouth_patch, roi, resized_target_mask, (roi_width//2, roi_height//2), cv2.NORMAL_CLONE)

            # Update the base face image with the blended ROI
            base_face_image[roi_y1:roi_y1+roi_height, roi_x1:roi_x1+roi_width] = blended_roi

            base_image[base_y1:base_y2, base_x1:base_x2] = base_face_image

            cv2.imwrite(self.final_output_path, base_image)

def main():
    """
    Main function for parsing command-line arguments and running the image processing pipeline.
    """
    parser = argparse.ArgumentParser(description='Patching and Blending')
    parser.add_argument('base_path', type=str, help='Path to the base image.')
    parser.add_argument('eyes_target_path', type=str, help='Path to the eyes target image')
    parser.add_argument('mouth_target_path', type=str, help='Path to the mouth target image')
    parser.add_argument('--eyes_landmarks_path', type=str, default='eyes.txt', help='Path to the text file containing landmarks of eye')
    parser.add_argument('--mouth_landmarks_path', type=str, default='mouth2.txt',help='Path to the text file containing landmarks of mouth')
    args = parser.parse_args()


    base_path = args.base_path
    output_directory = os.path.dirname(base_path)
    output_filename = os.path.splitext(os.path.basename(base_path))[0] + '_result.jpg'
    output_path = os.path.join(output_directory, output_filename)

    face_detector = FaceDetector("./models/Pytorch_RetinaFace_resnet50-640-640.onnx")
    face_landmarker = FaceLandmarker("./models/face_landmarker_v2_with_blendshapes.task")

    eye_base_path = base_path
    eye_landmarks_path = args.eyes_landmarks_path
    eye_target_path = args.eyes_target_path
    eye_final_output_path = output_path


    base_filename = eye_base_path
    target_filename = eye_target_path

    base_image = cv2.imread(eye_base_path)
    target_image = cv2.imread(eye_target_path)

    base_results = face_detector.detect_faces(base_image)
    target_results = face_detector.detect_faces(target_image)

    with open("base.txt", "w") as file:
        for i in range(len(base_results.boxes)):
            box = base_results.boxes[i]
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            face_image = base_image[y1:y2, x1:x2]
            face_label = f'Face{i + 1}'
            # output_face_path = f'Face{i + 1}' + base_filename
            output_face_path = os.path.join(os.path.dirname(eye_base_path), os.path.splitext(os.path.basename(eye_base_path))[0]) + face_label + '.jpg'
            cv2.imwrite(output_face_path, face_image)
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

            # delete the written file
            os.remove(output_face_path)
            

    with open("target.txt", "w") as file:
        for i in range(len(target_results.boxes)):
            box = target_results.boxes[i]
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            face_image = target_image[y1:y2, x1:x2]
            face_label = f'Face{i + 1}'
            # output_face_path = f'Face{i + 1}' + target_filename
            output_face_path = os.path.join(os.path.dirname(eye_target_path), os.path.splitext(os.path.basename(eye_target_path))[0]) + face_label + '.jpg'
            cv2.imwrite(output_face_path, face_image)
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

            # delete the written file
            os.remove(output_face_path)

    base_eye_landmarker = EyeLandmarkExtractor(eye_landmarks_path, "base.txt")
    target_eye_landmarker = EyeLandmarkExtractor(eye_landmarks_path, "target.txt")

    image_processor = EyesImageProcessor(eye_base_path, eye_target_path, base_eye_landmarker, target_eye_landmarker, face_detector, face_landmarker, eye_final_output_path)
    image_processor.process_images()

    mouth_base_path = eye_final_output_path
    mouth_target_path = args.mouth_target_path
    mouth_landmarks_path = args.mouth_landmarks_path
    base_filename = mouth_base_path
    target_filename = mouth_target_path
    mouth_final_output_path = output_path

    base_image = cv2.imread(mouth_base_path)
    target_image = cv2.imread(mouth_target_path)

    base_results = face_detector.detect_faces(base_image)
    target_results = face_detector.detect_faces(target_image)

    with open("target.txt", "w") as file:
        for i in range(len(target_results.boxes)):
            box = target_results.boxes[i]
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            face_image = target_image[y1:y2, x1:x2]
            face_label = f'Face{i + 1}'
            # output_face_path = f'Face{i + 1}' + target_filename
            output_face_path = os.path.join(os.path.dirname(mouth_target_path), os.path.splitext(os.path.basename(mouth_target_path))[0]) + face_label + '.jpg'
            cv2.imwrite(output_face_path, face_image)
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

            # delete the written file
            os.remove(output_face_path)

    base_mouth_landmarker = MouthLandmarkExtractor(mouth_landmarks_path, "base.txt")
    target_mouth_landmarker = MouthLandmarkExtractor(mouth_landmarks_path, "target.txt")

    scale_factor = 0.5
    image_processor = MouthImageProcessor(mouth_base_path, mouth_target_path, base_mouth_landmarker, target_mouth_landmarker, face_detector, face_landmarker, scale_factor, mouth_final_output_path)
    image_processor.process_images()

if __name__ == "__main__":
    main()






