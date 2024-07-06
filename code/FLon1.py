import cv2
import dlib
import json

# Load the pre-trained face detector and facial landmark detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Download this file from Dlib's website

# Load the input image
image = cv2.imread("./portrait_image.jpg")

# Convert the image to grayscale (optional but recommended for face detection)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the grayscale image
faces = detector(gray)

# Initialize a dictionary to store facial key points
facial_key_points = {}

# Loop over the faces detected
for i, face in enumerate(faces):
    # Initialize a list to store facial landmarks for each face
    face_landmarks = []


    # Determine the facial landmarks for the face region
    landmarks = predictor(gray, face)
    
    # Loop over the facial landmarks and draw them on the image
    for n in range(0, 68):  # 68 landmarks for a human face in the common pre-trained model
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        face_landmarks.append((x, y))
        cv2.circle(image, (x, y), 1, (0, 255, 0), -1)  # Draw a circle at each landmark point
        cv2.putText(image, str(n + 1), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1, cv2.LINE_AA)  # Add text annotation for landmark number

    # Store the facial landmarks list in the dictionary with a unique key
    facial_key_points[f"face_{i+1}"] = face_landmarks

# Save the facial key points dictionary to a JSON file
with open("facial_key_points.json", "w") as json_file:
    json.dump(facial_key_points, json_file)

print("Facial key points saved to facial_key_points.json")

# Save the image with facial landmarks drawn
output_image_path = "mix.jpg"
cv2.imwrite(output_image_path, image)
print(f"Output image with facial landmarks saved to {output_image_path}")

# # Display the image with facial landmarks
cv2.imshow("Facial Landmarks", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
