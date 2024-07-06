"""
Author: Saketh Reddy Vemula

usage: edit_face.py [-h] base_image_path cluster_folder_path

Edit base image with single face using cluster corresponding to it.

positional arguments:
  base_image_path      Path to the base image
  cluster_folder_path  Path to the folder containing cluster images

options:
  -h, --help           show this help message and exit
"""

import os
import shutil
import argparse
import subprocess

class FaceEditor:
    """
    A class for editing a base image with a single face using a cluster of images.

    Args:
        base_image_path (str): The path to the base image.
        cluster_folder_path (str): The path to the folder containing the cluster images.

    Attributes:
        base_image_path (str): The path to the base image.
        cluster_folder_path (str): The path to the folder containing the cluster images.
        eyes_target_image (str or None): The path to the target image for the eyes.
        mouth_target_image (str or None): The path to the target image for the mouth.

    Methods:
        pick_best_filter_eye(): Runs the `pick_best_filter_eye.py` script to select the best eye target image.
        pick_best_filter_mouth(): Runs the `pick_best_filter_mouth.py` script to select the best mouth target image.
        patch_blend(): Runs the `patch_blend.py` script to blend the target eye and mouth images with the base image.
        clean_up(): Removes temporary files generated during the process.
        edit_face(): Runs the entire face editing process by calling the other methods in the correct order.
    """
    def __init__(self, base_image_path, cluster_folder_path):
        self.base_image_path = base_image_path
        self.cluster_folder_path = cluster_folder_path
        self.eyes_target_image = None
        self.mouth_target_image = None

    def pick_best_filter_eye(self):
        """
        Runs the `pick_best_filter_eye.py` script to select the best eye target image.

        This method calls the `pick_best_filter_eye.py` script with the base image path and
        cluster folder path as arguments. It then retrieves the path of the selected eye
        target image from the generated CSV file and stores it in the `eyes_target_image`
        attribute.
        """
        subprocess.run(['python3', 'pick_best_filter_eye.py', self.base_image_path, self.cluster_folder_path], check=True)
        eyes_target_path = os.path.join(self.cluster_folder_path, f"{os.path.basename(self.cluster_folder_path)}_eyes_csv.csv")
        with open(eyes_target_path, "r") as csvfile:
            next(csvfile)  # Skip the header row
            self.eyes_target_image = os.path.abspath(next(csvfile).split(",")[0])

    def pick_best_filter_mouth(self):
        """
        Runs the `pick_best_filter_mouth.py` script to select the best mouth target image.

        This method calls the `pick_best_filter_mouth.py` script with the base image path and
        cluster folder path as arguments. It then retrieves the path of the selected mouth
        target image from the generated CSV file and stores it in the `mouth_target_image`
        attribute.
        """
        subprocess.run(['python3', 'pick_best_filter_mouth.py', self.base_image_path, self.cluster_folder_path], check=True)
        mouth_target_path = os.path.join(self.cluster_folder_path, f"{os.path.basename(self.cluster_folder_path)}_mouth_csv.csv")
        with open(mouth_target_path, "r") as csvfile:
            next(csvfile)  # Skip the header row
            self.mouth_target_image = os.path.abspath(next(csvfile).split(",")[0])

    def patch_blend(self):
        """
        Runs the `patch_blend.py` script to blend the target eye and mouth images with the base image.

        This method calls the `patch_blend.py` script with the base image path, the selected eye
        target image path, and the selected mouth target image path as arguments. The script
        performs the blending operation and generates the final output image.
        """
        subprocess.run(['python3', 'patch_blend.py', self.base_image_path, self.eyes_target_image, self.mouth_target_image], check=True)

    def clean_up(self):
        """
        Removes temporary files generated during the process.

        This method deletes the `base.txt` and `target.txt` files that were generated during
        the landmark extraction process.
        """
        os.remove("base.txt")
        os.remove("target.txt")

    def edit_face(self):
        """
        Runs the entire face editing process by calling the other methods in the correct order.

        This method coordinates the execution of the face editing process by calling the
        `pick_best_filter_eye()`, `pick_best_filter_mouth()`, `patch_blend()`, and `clean_up()`
        methods in the correct order.
        """
        self.pick_best_filter_eye()
        self.pick_best_filter_mouth()
        self.patch_blend()
        self.clean_up()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Edit base image with single face using cluster corresponding to it.')
    parser.add_argument('base_image_path', type=str, help='Path to the base image')
    parser.add_argument('cluster_folder_path', type=str, help='Path to the folder containing cluster images')
    args = parser.parse_args()

    face_editor = FaceEditor(args.base_image_path, args.cluster_folder_path)
    face_editor.edit_face()