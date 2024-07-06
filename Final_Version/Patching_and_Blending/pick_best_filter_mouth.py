"""
Author: Saketh Reddy Vemula

usage: pick_best_filter_mouth.py [-h] [--topK TOPK] base_image_path cluster_folder_path

Sort the best and filter out the eyes

positional arguments:
  base_image_path      Path to the base image
  cluster_folder_path  Path to the folder containing cluster images

options:
  -h, --help           show this help message and exit
  --topK TOPK          Number of entries in sorted order to sort according to smile scores.
"""

import os
import argparse
import csv
import cv2
import mediapipe as mp
import numpy as np
import math
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class LandmarkDetector:
    """
    Class for detecting facial landmarks using the MediaPipe FaceLandmarker task.

    Args:
        model_path (str): Path to the model asset file for the FaceLandmarker task.

    Attributes:
        model_path (str): Path to the model asset file for the FaceLandmarker task.
        detector (mediapipe.tasks.vision.FaceLandmarker): Instance of the FaceLandmarker task.

    Methods:
        initialize_detector(): Initializes the FaceLandmarker instance with the specified options.
        detect_landmarks(image_path): Detects facial landmarks from an input image.
    """
    def __init__(self, model_path):
        self.model_path = model_path
        self.initialize_detector()

    def initialize_detector(self):
        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1,
            min_face_detection_confidence=0.5)
        self.detector = vision.FaceLandmarker.create_from_options(options)

    def detect_landmarks(self, image_path):
        face_input = cv2.imread(image_path)
        face_input = cv2.cvtColor(face_input, cv2.COLOR_BGR2RGB)
        face_input = cv2.resize(face_input, (256, 256))
        face_input = np.expand_dims(face_input, axis=0)

        face_input_image = mp.Image.create_from_file(image_path)
        return self.detector.detect(face_input_image)

class PoseCalculator:
    """
    Class for calculating the pose (rotation angles) of a face from an image.

    Attributes:
        mp_face_mesh (mediapipe.solutions.face_mesh): MediaPipe FaceMesh solution for face mesh estimation.
        face_mesh (mediapipe.solutions.face_mesh.FaceMesh): Instance of the FaceMesh solution.

    Methods:
        calculate_pose(image_path): Calculates the pose (rotation angles) of the face in the given image.
    """
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def calculate_pose(self, image_path):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image)

        img_h, img_w, img_c = image.shape
        face_3d = []
        face_2d = []

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for idx, lm in enumerate(face_landmarks.landmark):
                    if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                        if idx == 1:
                            nose_2d = (lm.x * img_w, lm.y * img_h)
                            nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)
                        x, y = int(lm.x * img_w), int(lm.y * img_h)

                        face_2d.append([x, y])
                        face_3d.append([x, y, lm.z])

                face_2d = np.array(face_2d, dtype=np.float64)
                face_3d = np.array(face_3d, dtype=np.float64)
                focal_length = 1 * img_w

                cam_matrix = np.array([[focal_length, 0, img_h / 2],
                                       [0, focal_length, img_w / 2],
                                       [0, 0, 1]])

                dist_matrix = np.zeros((4, 1), dtype=np.float64)

                success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
                rmat, jac = cv2.Rodrigues(rot_vec)
                angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
                x = angles[0] * 360
                y = angles[1] * 360
                z = angles[2] * 360
                return x, y, z
        else:
            return 0, 0, 0

class BlendshapeExtractor:
    """
    Class for extracting blendshape values (e.g., mouth smile) from facial landmarks.

    Attributes:
        landmark_detector (LandmarkDetector): Instance of the LandmarkDetector class for detecting facial landmarks.

    Methods:
        extract_blendshapes(detection_result): Extracts the blendshape values from a detection result.
        calculate_blendshapes(image_path): Calculates the blendshape values for the given image.
    """
    def __init__(self):
        self.landmark_detector = LandmarkDetector('./models/face_landmarker_v2_with_blendshapes.task')

    def extract_blendshapes(self, detection_result):
        if detection_result.face_blendshapes:
            blendshapes = {
                blendshape.category_name: blendshape.score
                for blendshape in detection_result.face_blendshapes[0]
            }
            return blendshapes
        else:
            return None

    def calculate_blendshapes(self, image_path):
        detection_result = self.landmark_detector.detect_landmarks(image_path)
        blendshapes = self.extract_blendshapes(detection_result)
        mouthSmileBlendshapes = {
            'mouthSmileLeft': 0.0,
            'mouthSmileRight': 0.0,
            'mouthStretchLeft': 0.0,
            'mouthStretchRight': 0.0,
            'mouthUpperUpLeft': 0.0,
            'mouthUpperUpRight': 0.0
        }

        if blendshapes:
            mouthSmileBlendshapes['mouthSmileLeft'] = blendshapes['mouthSmileLeft']
            mouthSmileBlendshapes['mouthSmileRight'] = blendshapes['mouthSmileRight']
            mouthSmileBlendshapes['mouthStretchLeft'] = blendshapes['mouthStretchLeft']
            mouthSmileBlendshapes['mouthStretchRight'] = blendshapes['mouthStretchRight']
            mouthSmileBlendshapes['mouthUpperUpLeft'] = blendshapes['mouthUpperUpLeft']
            mouthSmileBlendshapes['mouthUpperUpRight'] = blendshapes['mouthUpperUpRight']

        return mouthSmileBlendshapes

class Utilities:
    """
    Class containing utility functions.

    Methods:
        cosine_similarity(a, b): Calculates the cosine similarity between two vectors.
    """
    @staticmethod
    def cosine_similarity(a, b):
        dot_product = a[0] * b[0] + a[1] * b[1]
        magnitude_a = math.sqrt(a[0] ** 2 + a[1] ** 2)
        magnitude_b = math.sqrt(b[0] ** 2 + b[1] ** 2)
        if magnitude_a == 0 or magnitude_b == 0:
            return 0
        else:
            return dot_product / (magnitude_a * magnitude_b)

class ImageProcessor:
    """
    Class for processing images, calculating poses, and sorting the images based on pose similarity and mouth smile scores.

    Args:
        base_image_path (str): Path to the base image.
        cluster_folder_path (str): Path to the folder containing cluster images.
        topK (int): Number of entries to be sorted based on smile scores.

    Attributes:
        base_image_path (str): Path to the base image.
        cluster_folder_path (str): Path to the folder containing cluster images.
        topK (int): Number of entries to be sorted based on smile scores.
        csv_file_path (str): Path to the CSV file for storing pose and blendshape data.
        pose_calculator (PoseCalculator): Instance of the PoseCalculator class for calculating poses.
        blendshape_extractor (BlendshapeExtractor): Instance of the BlendshapeExtractor class for extracting blendshapes.
        base_pose (tuple): Pose (rotation angles) of the base image.
        base_blendshapes (dict): Blendshape values of the base image.

    Methods:
        create_csv_file(): Creates a CSV file for storing pose and blendshape data.
        process_images(): Processes all images in the cluster folder, calculating poses and blendshapes.
        read_pose_data(): Reads pose and blendshape data from the CSV file.
        sort_and_filter(pose_data): Sorts pose data by similarity to the base pose and sorts the top K entries based on smile scores.
        write_top_k(top_k): Writes the top K pose and blendshape data to the CSV file.
        run(): Runs the entire process of creating the CSV file, processing images, sorting, and filtering.
    """
    def __init__(self, base_image_path, cluster_folder_path, topK):
        self.base_image_path = base_image_path
        self.cluster_folder_path = cluster_folder_path
        self.topK = topK
        self.csv_file_path = os.path.join(cluster_folder_path, f"{os.path.basename(cluster_folder_path)}_mouth_csv.csv")
        self.pose_calculator = PoseCalculator()
        self.blendshape_extractor = BlendshapeExtractor()
        self.base_pose = self.pose_calculator.calculate_pose(base_image_path)
        self.base_blendshapes = self.blendshape_extractor.calculate_blendshapes(base_image_path)

    def create_csv_file(self):
        with open(self.csv_file_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['path/to/file', 'x', 'y', 'z', 'mouthSmileLeft', 'mouthSmileRight', 'mouthStretchLeft', 'mouthStretchRight', 'mouthUpperUpLeft', 'mouthUpperUpRight'])

    def process_images(self):
        for filename in os.listdir(self.cluster_folder_path):
            if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.JPG') or filename.endswith('.jpeg'):
                image_path = os.path.join(self.cluster_folder_path, filename)
                print(f"Processing image: {image_path}")
                x, y, z = self.pose_calculator.calculate_pose(image_path)
                if (x != 0 and y != 0 and z != 0):
                    blendshapes = self.blendshape_extractor.calculate_blendshapes(image_path)
                    with open(self.csv_file_path, 'a', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow([image_path, x, y, z, blendshapes['mouthSmileLeft'], blendshapes['mouthSmileRight'], blendshapes['mouthStretchLeft'], blendshapes['mouthStretchRight'], blendshapes['mouthUpperUpLeft'], blendshapes['mouthUpperUpRight']])

    def read_pose_data(self):
        pose_data = []
        with open(self.csv_file_path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Skip the header row
            for row in reader:
                image_path, x, y, z, mouth_smile_left, mouth_smile_right, mouth_stretch_left, mouth_stretch_right, mouth_upper_up_left, mouth_upper_up_right = row
                x, y, z = float(x), float(y), float(z)
                mouth_smile_left, mouth_smile_right, mouth_stretch_left, mouth_stretch_right, mouth_upper_up_left, mouth_upper_up_right = float(mouth_smile_left), float(mouth_smile_right), float(mouth_stretch_left), float(mouth_stretch_right), float(mouth_upper_up_left), float(mouth_upper_up_right)
                pose_data.append((image_path, x, y, z, mouth_smile_left, mouth_smile_right, mouth_stretch_left, mouth_stretch_right, mouth_upper_up_left, mouth_upper_up_right))
        return pose_data

    def sort_and_filter(self, pose_data):
        base_x, base_y, base_z = self.base_pose
        pose_data.sort(key=lambda x: -Utilities.cosine_similarity((base_x, base_y), (x[1], x[2])))

        top_k = pose_data[:self.topK]
        top_k.sort(key=lambda x: (x[4] + x[5]) / 2, reverse=True)

        return top_k

    def write_top_k(self, top_k):
        with open(self.csv_file_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['path/to/file', 'x', 'y', 'z', 'mouthSmileLeft', 'mouthSmileRight', 'mouthStretchLeft', 'mouthStretchRight', 'mouthUpperUpLeft', 'mouthUpperUpRight'])
            for data in top_k:
                writer.writerow(data)

    def run(self):
        self.create_csv_file()
        self.process_images()
        pose_data = self.read_pose_data()
        top_k = self.sort_and_filter(pose_data)
        self.write_top_k(top_k)

def main():
    """
    Main function for parsing command-line arguments and running the ImageProcessor.
    """
    parser = argparse.ArgumentParser(description='Sort the best and filter out the eyes')
    parser.add_argument('base_image_path', type=str, help='Path to the base image')
    parser.add_argument('cluster_folder_path', type=str, help='Path to the folder containing cluster images')
    parser.add_argument('--topK', type=int, default=5, help='Number of entries in sorted order to sort according to smile scores.')
    args = parser.parse_args()

    image_processor = ImageProcessor(args.base_image_path, args.cluster_folder_path, args.topK)
    image_processor.run()

if __name__ == '__main__':
    main()