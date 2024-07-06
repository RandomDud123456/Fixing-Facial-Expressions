"""
    run: python3 main.py --frame path/to/frame_image.jpg --target path/to/target_image.jpg
"""

import os
import cv2
import numpy as np
import argparse
from Detector import FaceDetector
from MaskGenerator import MaskGenerator

# cap = cv2.VideoCapture(0)
# frame = cv2.imread('images/correct.jpg')
# frame = cv2.imread('images/obama.png')
# frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
detector = FaceDetector()
maskGenerator = MaskGenerator()

def parse_arguments():
    parser = argparse.ArgumentParser(description="Face Masking")
    parser.add_argument("--frame", required=True, help="Path to the frame image")
    parser.add_argument("--target", required=True, help="Path to the target image")
    return parser.parse_args()

def showImages(actual, target, output1, output2, directory_target_image, filename_target_image):
    target_height, target_width = target.shape[:2]  # Get the height and width of the target image

    # Resize all images to match the target image's dimensions
    img_actual = cv2.resize(actual, (target_width, target_height), interpolation=cv2.INTER_AREA)
    img_target = cv2.resize(target, (target_width, target_height), interpolation=cv2.INTER_AREA)
    img_out1 = cv2.resize(output1, (target_width, target_height), interpolation=cv2.INTER_AREA)
    img_out2 = cv2.resize(output2, (target_width, target_height), interpolation=cv2.INTER_AREA)

    # Concatenate the images side by side (along width)
    h1 = np.concatenate((img_actual, img_target, img_out1, img_out2), axis=1)

    # cv2.imshow('Face Mask', h1)
    cv2.imwrite(directory_target_image + "/output_" + filename_target_image, output2)


# Target
# target_image, target_alpha = detector.load_target_img("images/cage.png")
# target_image, target_alpha = detector.load_target_img("images/incorrect.jpg")
# target_image, target_alpha = detector.load_target_img("images/obama.png")
# target_image, target_alpha = detector.load_target_img("images/trump.png")
# target_image, target_alpha = detector.load_target_img("images/kim.png")
# target_image, target_alpha = detector.load_target_img("images/putin.png")
# target_image, target_alpha = detector.load_target_img("images/client.png")
    
args = parse_arguments()

frame = cv2.imread(args.frame)
target_image, target_alpha = detector.load_target_img(args.target)

target_image_for_size = cv2.imread(args.target)
output_height, output_width = target_image_for_size.shape[:2]

directory_target_image, filename_target_image = os.path.split(args.target)

target_landmarks, _, target_face_landmarks= detector.find_face_landmarks(target_image)
target_image_out = detector.drawLandmarks(target_image, target_face_landmarks)

maskGenerator.calculateTargetInfo(target_image, target_alpha, target_landmarks)

# Assuming target_image is already loaded and has the dimensions you want to match
height, width = target_image.shape[:2]  # Get the height and width of the target image

# Resize frame to match the target_image's dimensions
resized_frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
frame = resized_frame

landmarks, image, face_landmarks = detector.find_face_landmarks(frame)

detector.stabilizeVideoStream(frame, landmarks)

output = maskGenerator.applyTargetMask(frame, landmarks)
output2 = maskGenerator.applyTargetMaskToTarget(landmarks)

output2 = cv2.resize(output2, (output_width, output_height), interpolation=cv2.INTER_AREA)
# cv2.imshow("Output", output2)
# cv2.waitKey(0)

image_out = detector.drawLandmarks(image, face_landmarks)
showImages(image_out, target_image_out, output, output2, directory_target_image, filename_target_image)

# cv2.waitKey(0)