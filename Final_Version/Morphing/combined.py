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
    
    def calculate_yaw(self, face_landmarks, face_image):
        """
        Calculate the yaw of the face based on landmarks.
        Assumes the face_landmarks are a list of MediaPipe NormalizedLandmark objects.
        """
        # Assuming landmark indices for simplicity; check MediaPipe documentation for exact indices
        nose_tip_index = 4  # Index for nose tip in landmarks
        chin_index = 152 # Index of chin tip in landmarks
        left_eye_left_index = 33 # Index of left eye left point
        right_eye_right_index = 263 # Index of right eye right point
        left_mouth_corner = 185 # Index of left mouth corner (approx)
        right_mouth_corner = 409 # Index of rigth mouth corner (approx)
        left_eye_index = 468  # Approximate index for left eye in landmarks
        right_eye_index = 473  # Approximate index for right eye in landmarks
        
        # Extract the required landmarks
        nose_tip = np.array([face_landmarks[nose_tip_index].x, face_landmarks[nose_tip_index].y])
        left_eye = np.array([face_landmarks[left_eye_index].x, face_landmarks[left_eye_index].y])
        right_eye = np.array([face_landmarks[right_eye_index].x, face_landmarks[right_eye_index].y])

        # Calculate mid-point between the eyes
        eye_midpoint = (left_eye + right_eye) / 2.0

        # Calculate the vector from eye midpoint to nose tip
        direction_vector = nose_tip - eye_midpoint

        # Calculate yaw based on the horizontal component of the direction vector
        # This is a simplification and for more accurate calculation, a 3D model should be used
        yaw = np.arctan2(direction_vector[1], direction_vector[0]) * 180.0 / np.pi
        # print(yaw)
        # return yaw
    
        # landmarks = self.predict(gray_face_img, subject)
        image_points = np.array([
            (face_landmarks[nose_tip_index].x, face_landmarks[nose_tip_index].y),  # Nose tip
            (face_landmarks[chin_index].x, face_landmarks[chin_index].y),   # Chin
            (face_landmarks[left_eye_left_index].x, face_landmarks[left_eye_left_index].y), # Left eye left corner
            (face_landmarks[right_eye_right_index].x, face_landmarks[right_eye_right_index].y), # Right eye right corner
            (face_landmarks[left_mouth_corner].x, face_landmarks[left_mouth_corner].y), # Left mouth corner
            (face_landmarks[right_mouth_corner].x, face_landmarks[right_mouth_corner].y)  # Right mouth corner
        ], dtype="double")

        image_shape=face_image.shape

        model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye left corner
            (225.0, 170.0, -135.0),      # Right eye right corner
            (-150.0, -150.0, -125.0),    # Left Mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ])
        focal_length = image_shape[1]
        center = (image_shape[1] / 2, image_shape[0] / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype="double")
        dist_coeffs = np.zeros((4, 1))  

        success, rotation_vector, translation_vector = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        euler_angles = cv2.decomposeProjectionMatrix(camera_matrix @ np.hstack((rotation_matrix, translation_vector)))[6]
            
        pitch, yaw, roll = euler_angles.flatten()
        # print(pitch, yaw, roll)


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
        plt.savefig(output_plot_path)

if __name__ == "__main__":
    # Initialize detectors
    face_detector = FaceDetector("../models/Pytorch_RetinaFace_resnet50-640-640.onnx")
    landmark_detector = LandmarkDetector('../models/face_landmarker_v2_with_blendshapes.task')

    # Assume image loading and other operations are here
    # Load the image
    IMAGE_PATH = "./Mini_Dataset/292A2172/292A2172.JPG"
    parts = IMAGE_PATH.split('/')
    group_id = parts[-2]
    directory, filename = os.path.split(IMAGE_PATH)
    # print(directory)
    # print(filename)
    image = cv2.imread(IMAGE_PATH)

    # Example usage:
    results = face_detector.detect_faces(image)

    created_files = []

    for i in range(len(results.boxes)):
        box = results.boxes[i]
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])

        # Crop the face region
        face_image = image[y1:y2, x1:x2]
        face_label = f'Face{i + 1}'
        filename_without_jpg = filename.split('.')[0]
        output_face_path = directory + "/" + "extracted_" + filename_without_jpg + f'_{i + 1}' + ".jpg"
        cv2.imwrite(output_face_path, face_image)

        face_input = mp.Image.create_from_file(output_face_path)
        directory_face, filename_face = os.path.split(output_face_path)
        new_filename = "landmark_" + filename_face
        face_output_path = os.path.join(directory, new_filename)

        detection_result = landmark_detector.detect_landmarks(output_face_path)
        if detection_result.face_landmarks:  # Ensure there are detected landmarks
            yaw = landmark_detector.calculate_yaw(detection_result.face_landmarks[0], face_image)
            # print(f"Yaw of the face: {yaw} degrees")
        annotated_image = landmark_detector.draw_landmarks_on_image(face_input.numpy_view(), detection_result)
        cv2.imwrite(face_output_path, cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

        # created_files.append(output_face_path) # For removing the output faces.
        created_files.append(face_output_path) # For removing the output landmarked faces.

        # plot_face_blendshapes_bar_graph(detection_result.face_blendshapes[0])
        # print(detection_result.facial_transformation_matrixes)
        image[y1:y2, x1:x2] = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)

        # Before replacing the face in the original image, add the label
        label_position = (x1, y1 - 10)  # Positioning the label above the face bounding box
        FONT_SCALE = 0.3
        font_color = (0, 255, 0)  # White color
        LINE_TYPE = 1

        cv2.putText(image, face_label, label_position, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, font_color, LINE_TYPE)
        # Plotter.plot_face_blendshapes_bar_graph(detection_result.face_blendshapes[0], output_face_path, output_face_path)
        # Before calling Plotter.plot_face_blendshapes_bar_graph()
        if detection_result.face_blendshapes:  # Check if the list is not empty
            Plotter.plot_face_blendshapes_bar_graph(detection_result.face_blendshapes[0], output_face_path, output_face_path)
        else:
            print("No face blendshapes detected for this face.")

    # Rest of the processing and cleanup
    final_output_path = directory + "/" + str(group_id) + "_landmarked.jpg"
    # cv2.imwrite("Landmarked_Output.jpg", image)
    cv2.imwrite(final_output_path, image)

    # Now delete the intermediate files
    for file_path in created_files:
        try:
            os.remove(file_path)
            # print(f"Deleted {file_path}")
        except Exception as e:
            # print(f"Error deleting {file_path}: {e}")
            pass
