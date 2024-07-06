"""
Author: Saketh Reddy Vemula

usage: driver.py [-h] group_photo_path

Edit group photo with multiple faces using clusters.

positional arguments:
  group_photo_path  Path to the group photo

options:
  -h, --help        show this help message and exit
"""

import os
import shutil
import argparse
import subprocess
import cv2
import time
from fastdeploy.vision import facedet
from fastdeploy.runtime import RuntimeOption, ModelFormat

class FaceDetector:
    """
    A class for detecting faces in an image using the RetinaFace model.

    Attributes:
        model_file (str): Path to the RetinaFace model file.
        retinaface (fastdeploy.vision.facedet.RetinaFace): Instance of the RetinaFace model.

    Methods:
        detect_faces(image): Detects faces in the given image and returns the results.
    """
    def __init__(self, model_file):
        self.model_file = model_file
        self.retinaface = facedet.RetinaFace(self.model_file, runtime_option=None, model_format=ModelFormat.ONNX)

    def detect_faces(self, image):
        """
        Detects faces in the given image using the RetinaFace model.

        Args:
            image (numpy.ndarray): Input image.

        Returns:
            fastdeploy.vision.facedet.DetectionResult: Results containing bounding boxes and scores for detected faces.
        """
        results = self.retinaface.predict(image)
        return results

class FaceEditor:
    """
    A class for editing a face image using the best-filtered eyes and mouth from a cluster of images.

    Attributes:
        base_image_path (str): Path to the base image.
        cluster_folder_path (str): Path to the folder containing clusters of images.
        upper_limit_threshold_eyes (str): Upper limit threshold for eye blink detection.
        eyes_target_image (str): Path to the best-filtered eyes image.
        mouth_target_image (str): Path to the best-filtered mouth image.

    Methods:
        pick_best_filter_eye(): Selects the best-filtered eyes image from the cluster.
        pick_best_filter_mouth(): Selects the best-filtered mouth image from the cluster.
        patch_blend(): Blends the base image with the best-filtered eyes and mouth images.
        clean_up(): Removes temporary files.
        edit_face(): Performs the entire face editing process.
    """
    def __init__(self, base_image_path, cluster_folder_path, upper_limit_threshold_eyes):
        self.base_image_path = base_image_path
        self.cluster_folder_path = cluster_folder_path
        self.upper_limit_threshold_eyes = upper_limit_threshold_eyes
        self.eyes_target_image = None
        self.mouth_target_image = None

    def pick_best_filter_eye(self):
        """
        Selects the best-filtered eyes image from the cluster by running the 'pick_best_filter_eye.py' script.
        """
        subprocess.run(['python3', 'pick_best_filter_eye.py', self.base_image_path, self.cluster_folder_path, '--threshold', self.upper_limit_threshold_eyes], check=True)
        eyes_target_path = os.path.join(self.cluster_folder_path, f"{os.path.basename(self.cluster_folder_path)}_eyes_csv.csv")
        with open(eyes_target_path, "r") as csvfile:
            next(csvfile)  # Skip the header row
            self.eyes_target_image = os.path.abspath(next(csvfile).split(",")[0])

    def pick_best_filter_mouth(self):
        """
        Selects the best-filtered mouth image from the cluster by running the 'pick_best_filter_mouth.py' script.
        """
        subprocess.run(['python3', 'pick_best_filter_mouth.py', self.base_image_path, self.cluster_folder_path], check=True)
        mouth_target_path = os.path.join(self.cluster_folder_path, f"{os.path.basename(self.cluster_folder_path)}_mouth_csv.csv")
        with open(mouth_target_path, "r") as csvfile:
            next(csvfile)  # Skip the header row
            self.mouth_target_image = os.path.abspath(next(csvfile).split(",")[0])

    def patch_blend(self):
        """
        Blends the base image with the best-filtered eyes and mouth images by running the 'patch_blend.py' script.
        """
        subprocess.run(['python3', 'patch_blend.py', self.base_image_path, self.eyes_target_image, self.mouth_target_image], check=True)

    def clean_up(self):
        """
        Removes temporary files created during the face editing process.
        """
        os.remove("base.txt")
        os.remove("target.txt")

    def edit_face(self):
        """
        Performs the entire face editing process by calling the necessary methods in the correct order.
        """
        self.pick_best_filter_eye()
        self.pick_best_filter_mouth()
        self.patch_blend()
        self.clean_up()

class GroupPhotoEditor:
    """
    A class for editing a group photo by detecting and editing individual faces.

    Attributes:
        group_photo_path (str): Path to the group photo.
        base_image (numpy.ndarray): The group photo image.
        face_detector (FaceDetector): Instance of the FaceDetector class.

    Methods:
        process_group_photo(): Detects faces in the group photo, prompts for face editing, and saves the edited photo.
    """
    def __init__(self, group_photo_path):
        self.group_photo_path = group_photo_path
        self.base_image = cv2.imread(group_photo_path)
        self.face_detector = FaceDetector("./models/Pytorch_RetinaFace_resnet50-640-640.onnx")

    def process_group_photo(self):
        """
        Detects faces in the group photo, prompts the user to edit each face, and saves the edited group photo.
        """
        results = self.face_detector.detect_faces(self.base_image)

        for i, box in enumerate(results.boxes):
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            face_image = self.base_image[y1:y2, x1:x2]
            face_image_path = 'face.jpg'
            cv2.imwrite(face_image_path, face_image)
            cv2.imshow(f"Face {i+1}", face_image)
            cv2.waitKey(2000)  # Show the face for 2 seconds
            cv2.destroyWindow(f"Face {i+1}")

            edit_choice = input(f"Edit Face {i+1}? (y/n) ")
            if edit_choice.lower() == 'y':
                cluster_folder_path = input(f"Enter path to cluster folder for Face {i+1}: ")
                upper_limit_threshold_eyes = str(input(f"Enter upper limit for eye blink: "))

                face_editor = FaceEditor(face_image_path, cluster_folder_path, upper_limit_threshold_eyes)
                face_editor.edit_face()
                blended_face = cv2.imread('face_result.jpg')
                self.base_image[y1:y2, x1:x2] = blended_face
            else:
                continue

        cv2.imwrite("edited_group_photo.jpg", self.base_image)
        print("Edited group photo saved as 'edited_group_photo.jpg'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Edit group photo with multiple faces using clusters.')
    parser.add_argument('group_photo_path', type=str, help='Path to the group photo')
    args = parser.parse_args()
    group_photo_path = args.group_photo_path

    group_photo_editor = GroupPhotoEditor(group_photo_path)
    group_photo_editor.process_group_photo()