"""
    run: python3 combined_oop.py
    Input: Group image path
    Output: corresponding landmarked group photo
    Options: to print faces and landmarked faces seperately
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from fastdeploy.vision import facedet
from fastdeploy.runtime import RuntimeOption, ModelFormat
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class FaceDetector:
    """
        Detect faces using RetinaFace FastDeploy
    """
    def __init__(self, model_path):
        """
            Initialize
        """
        self.detector = facedet.RetinaFace(model_path, runtime_option=None, model_format=ModelFormat.ONNX)

    def detect_faces(self, image):
        """
            detect those faces using predict call
        """
        return self.detector.predict(image)

class LandmarkDetector:
    """
        Landmarking using MediaPipe
    """
    def __init__(self, model_path):
        """
            Initialize the Landmarking tool with correct options.
        """
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=14,
            min_face_detection_confidence=0.5)
        self.detector = vision.FaceLandmarker.create_from_options(options)

    def detect_landmarks(self, image):
        """
            detect landmarks
        """
        face_input = mp.Image.create_from_file(image)
        return self.detector.detect(face_input)

    @staticmethod
    def draw_landmarks_on_image(rgb_image, detection_result):
        """
            Draw the landmarks on the face
        """
        face_landmarks_list = detection_result.face_landmarks
        annotated_image = np.copy(rgb_image)

        for idx in range(len(face_landmarks_list)):
            face_landmarks = face_landmarks_list[idx]
            face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            face_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
            ])

            mp.solutions.drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks_proto,
                connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles
                .get_default_face_mesh_tesselation_style())

            # Additional draw_landmarks calls for FACEMESH_CONTOURS, FACEMESH_IRISES omitted for brevity

        return annotated_image

class Plotter:
    """
        Helper function to plot the blendshape's bar graphs.
    """
    @staticmethod
    def plot_face_blendshapes_bar_graph(face_blendshapes, output_plot_path, face_image_path):
        """
            plotting function
        """
        # Extract the face blendshapes category names and scores.
        face_blendshapes_names = [face_blendshapes_category.category_name for face_blendshapes_category in face_blendshapes]
        face_blendshapes_scores = [face_blendshapes_category.score for face_blendshapes_category in face_blendshapes]
        # The blendshapes are ordered in decreasing score value.
        face_blendshapes_ranks = range(len(face_blendshapes_names))

        fig, ax = plt.subplots(figsize=(12, 12))
        bar = ax.barh(face_blendshapes_ranks, face_blendshapes_scores, label=[str(x) for x in face_blendshapes_ranks])
        ax.set_yticks(face_blendshapes_ranks, face_blendshapes_names)
        ax.invert_yaxis()

        # Label each bar with values
        for score, patch in zip(face_blendshapes_scores, bar.patches):
            plt.text(patch.get_x() + patch.get_width(), patch.get_y(), f"{score:.4f}", va="top")

        ax.set_xlabel('Score')
        ax.set_title("Face Blendshapes")

        # Load the face image
        face_image = plt.imread(face_image_path)

        # Create an inset of width 30% and height 30% of the figure, positioned at the bottom right
        ax_inset = inset_axes(ax, width="30%", height="30%", loc="lower right")
        ax_inset.imshow(face_image)
        ax_inset.axis('off')

        plt.tight_layout()
        # plt.savefig(output_plot_path)

class Editter:
    @staticmethod
    def find_image_in_miniclusters(image_filename, dataset_path):
        # Look for 'mini' folders within the dataset
        for root, dirs, files in os.walk(dataset_path):
            # Check if the folder ends with 'mini'
            if root.endswith('mini'):
                # Walk through each subdirectory in the 'mini' folder
                for sub_root, sub_dirs, sub_files in os.walk(root):
                    # Check if the image file is in the current directory's files
                    if image_filename in sub_files:
                        # Return the path to the directory containing the image
                        return sub_root
        # If the image file is not found, return a message indicating so
        return "Image not found in 'mini' folders."

    @staticmethod
    def calculate_similarity(target_blendshapes, face_blendshapes, specific_blendshapes_names):
        # This function calculates similarity between two sets of blendshapes
        # Placeholder for actual implementation
        # similarity_score = np.dot(target_blendshapes, face_blendshapes) / (np.linalg.norm(target_blendshapes) * np.linalg.norm(face_blendshapes))
        # return similarity_score
        target_scores = [face_blendshapes_category.score for face_blendshapes_category in target_blendshapes if face_blendshapes_category.category_name not in specific_blendshapes_names]
        face_scores = [face_blendshapes_category.score for face_blendshapes_category in face_blendshapes if face_blendshapes_category.category_name not in specific_blendshapes_names]

        if len(target_scores) == 0 or len(face_scores) == 0:
            return 0
        
        # Calculate scores
        similarity_score = np.dot(target_scores, face_scores) / (np.linalg.norm(target_scores) * np.linalg.norm(face_scores))
        return similarity_score

    @staticmethod
    def process_images(target_image_path, cluster_folder_path):
        # Initialize detectors
        face_detector = FaceDetector("./models/Pytorch_RetinaFace_resnet50-640-640.onnx")
        landmark_detector = LandmarkDetector('./models/face_landmarker_v2_with_blendshapes.task')

        target_result = landmark_detector.detect_landmarks(target_image_path)
        target_face_blendshapes = target_result.face_blendshapes[0]
        specific_blendshapes_names = ['mouthSmileLeft', 'mouthSmileRight']
        similarities = []

        for filename in os.listdir(cluster_folder_path):
            face_path = os.path.join(cluster_folder_path, filename)
            
            face_result = landmark_detector.detect_landmarks(face_path)
            face_result_blendshapes = face_result.face_blendshapes[0]
            similarity = Editter.calculate_similarity(target_face_blendshapes, face_result_blendshapes, specific_blendshapes_names)
            similarities.append((filename, similarity))

        similarities.sort(key=lambda x: x[1], reverse=True)
        sorted_filenames = [filename for filename, _ in similarities]
        return sorted_filenames

if __name__ == "__main__": 
    target_image_path = "/home/saketh/IIITH Sem 4/DASS/Project/Modular Code/Mini_Dataset/292A2174/292A2174_7.jpg"
    _, image_filename = os.path.split(target_image_path)
    dataset_path = '/home/saketh/IIITH Sem 4/DASS/Project/Modular Code/Mini_Dataset'
    image_path = Editter.find_image_in_miniclusters(image_filename, dataset_path)
    print(image_path)

    mini_cluster_folder_path = image_path

    sorted_faces = Editter.process_images(target_image_path, mini_cluster_folder_path)
    print("Sorted faces based on the similarity to target:", sorted_faces)


