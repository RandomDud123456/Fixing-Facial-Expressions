"""
    run: python3 webcam_main.py --target "path/to/folder"
"""

import cv2
import os
import argparse
import numpy as np
from Detector import FaceDetector
from MaskGenerator import MaskGenerator

cap = cv2.VideoCapture(0)
detector = FaceDetector()
maskGenerator = MaskGenerator()

def parse_arguments():
    parser = argparse.ArgumentParser(description="Face Masking")
    parser.add_argument("--target", required=True, help="Path to the target")
    return parser.parse_args()

def showImages(actual, target, output1, output2):
    img_actual = np.copy(actual)
    img_target = np.copy(target)
    img_out1 = np.copy(output1)
    img_out2 = np.copy(output2)
    # 640x480 -> 360x480
    img_actual = img_actual[:, 140:500]
    img_out1 = img_out1[:, 140:500]
    # 480x640 -> 360x480
    img_target = cv2.resize(img_target, (360, 480), interpolation=cv2.INTER_AREA)
    img_out2 = cv2.resize(img_out2, (360, 480), interpolation=cv2.INTER_AREA)

    h1 = np.concatenate((img_actual, img_target, img_out1, img_out2), axis=1)

    cv2.imshow('Face Mask', h1)

args = parse_arguments()

input_image_path = args.target
target_image_for_size = cv2.imread(args.target)
output_height, output_width = target_image_for_size.shape[:2]
directory_input, filename_input = os.path.split(input_image_path)

target_image, target_alpha = detector.load_target_img(input_image_path)
# target_image, target_alpha = detector.load_target_img("images/client.png")
target_landmarks, _, target_face_landmarks= detector.find_face_landmarks(target_image)
target_image_out = detector.drawLandmarks(target_image, target_face_landmarks)

maskGenerator.calculateTargetInfo(target_image, target_alpha, target_landmarks)

while True:
    success, frame = cap.read()
    frame = cv2.flip(frame, 1)

    landmarks, image, face_landmarks = detector.find_face_landmarks(frame)
    if len(landmarks) == 0:
        continue

    detector.stabilizeVideoStream(frame, landmarks)

    output = maskGenerator.applyTargetMask(frame, landmarks)
    output2 = maskGenerator.applyTargetMaskToTarget(landmarks)
    output2 = cv2.resize(output2, (output_width, output_height), interpolation=cv2.INTER_AREA)

    image_out = detector.drawLandmarks(image, face_landmarks)
    showImages(image_out, target_image_out, output, output2)

    key = cv2.waitKey(1) & 0xFF
    if key == 13:  # 13 is the Enter key
        cv2.imwrite(directory_input + "/output_" + filename_input, output2)
        break
    elif key == 27:  # Optionally, 27 is the ESC key to break the loop
        cv2.imwrite(directory_input, "/output_" + filename_input, target_image_for_size)
        break

cap.release()
cv2.destroyAllWindows()
