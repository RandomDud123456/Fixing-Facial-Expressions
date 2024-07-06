import os
import cv2
import dlib
import json

# Load the pre-trained face detector and facial landmark detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Download this file from Dlib's website

# Define paths
input_folder = "./CustomPaDataset"
output_folder = "./Output3"

# Create output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Loop over the images in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith(".jpg") or filename.endswith(".jpeg"):
        # Load the input image
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path)

        # Convert the image to grayscale (optional but recommended for face detection)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale image
        faces = detector(gray)

        # Loop over the faces detected
        for i, face in enumerate(faces):
            # Determine the facial landmarks for the face region
            landmarks = predictor(gray, face)

            # Loop over the facial landmarks and draw them on the image
            for n in range(0, 68):  # 68 landmarks for a human face in the common pre-trained model
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                cv2.circle(image, (x, y), 1, (0, 255, 0), -1)  # Draw a circle at each landmark point

        # Save the image with facial landmarks drawn
        output_image_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_image_path, image)
        print(f"Output image with facial landmarks saved to {output_image_path}")

print("Facial landmark detection completed for all images.")
