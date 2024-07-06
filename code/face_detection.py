import cv2
import os
from fastdeploy.vision import facedet
from fastdeploy.runtime import RuntimeOption, ModelFormat

# Specify the path to your ONNX model
model_file = "./Pytorch_RetinaFace_resnet50-640-640.onnx"

# Load the model
retinaface = facedet.RetinaFace(model_file, runtime_option=None, model_format=ModelFormat.ONNX)

# Load the Image
image_path = "./groupPhoto.jpg"
image = cv2.imread(image_path)

# Perform face detection
results = retinaface.predict(image)

# Iterate over detected faces
for i in range(len(results.boxes)):
    box = results.boxes[i]
    score = results.scores[i]
    landmarks = results.landmarks[i]  # Assuming you also want to processandmarks

    # Draw bounding box
    cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)

    # Optionally, draw landmarks for each face
    for j in range(0, len(landmarks), 2):  # Assuming landmarks are [x1, y1, x2, y2, ..., xn, yn]
        cv2.circle(image, (int(landmarks[j]), int(landmarks[j+1])), radius=1, color=(0, 0, 255), thickness=2)

# Extract the directory and filename from the original image path
directory, filename = os.path.split(image_path)

# Construct the new filename by appending "face_detected" before the original filename
new_filename = "face_detected_" + filename

# Construct the full output path by joining the directory and the new filename
output_image_path = os.path.join(directory, new_filename)

# Save the image with detected faces
cv2.imwrite(output_image_path, image)
