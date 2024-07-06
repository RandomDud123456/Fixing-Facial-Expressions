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
import umap
from sklearn.cluster import KMeans
import shutil
import optuna

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

        return annotated_image
    
    def calculate_yaw(self, face_landmarks):
        """
        Calculate the yaw of the face based on landmarks.
        Assumes the face_landmarks are a list of MediaPipe NormalizedLandmark objects.
        """
        nose_tip_index = 4
        left_eye_index = 468
        right_eye_index = 473
        
        nose_tip = np.array([face_landmarks[nose_tip_index].x, face_landmarks[nose_tip_index].y])
        left_eye = np.array([face_landmarks[left_eye_index].x, face_landmarks[left_eye_index].y])
        right_eye = np.array([face_landmarks[right_eye_index].x, face_landmarks[right_eye_index].y])

        eye_midpoint = (left_eye + right_eye) / 2.0
        direction_vector = nose_tip - eye_midpoint
        yaw = np.arctan2(direction_vector[1], direction_vector[0]) * 180.0 / np.pi

        return yaw

# Initialize detectors
face_detector = FaceDetector("/home/saketh/IIITH Sem 4/DASS/Project/Modular Code/models/Pytorch_RetinaFace_resnet50-640-640.onnx")
landmark_detector = LandmarkDetector('/home/saketh/IIITH Sem 4/DASS/Project/Modular Code/models/face_landmarker_v2_with_blendshapes.task')

# Define the folder path containing the images
folder_path = "/home/saketh/IIITH Sem 4/DASS/Project/Modular Code/Mini_Dataset/cluster_20"

# Get a list of image file paths in the folder
image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.jpg') or f.endswith('.png')]

# Initialize lists to store yaw values and image paths
yaw_values = []
image_paths = []

for image_path in image_files:
    image = cv2.imread(image_path)
    results = face_detector.detect_faces(image)

    for i in range(len(results.boxes)):
        box = results.boxes[i]
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        face_image = image[y1:y2, x1:x2]

        detection_result = landmark_detector.detect_landmarks(image_path)
        if detection_result.face_landmarks:
            yaw = landmark_detector.calculate_yaw(detection_result.face_landmarks[0])
            yaw_values.append(yaw)
            image_paths.append(image_path)

def objective(trial):
    n_neighbors = trial.suggest_int('n_neighbors', 2, 10)
    min_dist = trial.suggest_float('min_dist', 0.1, 1.0)
    n_clusters = trial.suggest_int('n_clusters', 3, 3)
    
    umap_embeddings = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist).fit_transform(np.array(yaw_values).reshape(-1, 1))
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(umap_embeddings)
    
    return kmeans.inertia_

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

best_params = study.best_params
print("Best parameters:", best_params)

# Perform clustering with the best hyperparameters
umap_embeddings = umap.UMAP(n_neighbors=best_params['n_neighbors'], min_dist=best_params['min_dist']).fit_transform(np.array(yaw_values).reshape(-1, 1))
kmeans = KMeans(n_clusters=best_params['n_clusters'], random_state=0).fit(umap_embeddings)
cluster_labels = kmeans.labels_

# Create folders for each minicluster
output_folder = "/home/saketh/IIITH Sem 4/DASS/Project/Modular Code/Mini_Dataset/cluster_20_mini"
for i in range(best_params['n_clusters']):
    cluster_folder = os.path.join(output_folder, f"minicluster_{i}")
    os.makedirs(cluster_folder, exist_ok=True)

# Move images to their respective minicluster folders
for image_path, cluster_label in zip(image_paths, cluster_labels):
    _, filename = os.path.split(image_path)
    output_path = os.path.join(output_folder, f"minicluster_{cluster_label}", filename)
    shutil.copy(image_path, output_path)

print("Clustering completed and images moved to minicluster folders.")