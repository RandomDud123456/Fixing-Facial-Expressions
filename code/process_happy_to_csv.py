import cv2
import os
import csv
from fastdeploy.vision import facedet
from fastdeploy.runtime import RuntimeOption, ModelFormat
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np

def detect_and_extract_blendshapes(image_path, detector, retinaface):
    image = cv2.imread(image_path)
    results = retinaface.predict(image)
    blendshapes_data = []

    for i, box in enumerate(results.boxes):
        x1, y1, x2, y2 = map(int, box[:4])
        face_image = image[y1:y2, x1:x2]
        output_face_path = "temp_output_face_path.jpg"
        cv2.imwrite(output_face_path, face_image)

        face_input = mp.Image.create_from_file(output_face_path)
        detection_result = detector.detect(face_input)
        
        if detection_result.face_blendshapes:  # Check if blendshapes were detected
            blendshapes_dict = {category.category_name: category.score for blendshapes in detection_result.face_blendshapes for category in blendshapes}
            blendshapes_data.append(blendshapes_dict)

        os.remove(output_face_path)
    return blendshapes_data

def save_blendshapes_to_csv(blendshapes_data, output_csv_path):
    # Prepare header based on all possible blendshapes
    all_keys = set()
    for data in blendshapes_data.values():
        for blendshape_dict in data:
            all_keys.update(blendshape_dict.keys())
    
    header = ['Image'] + sorted(list(all_keys))
    
    with open(output_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        
        for image_name, data in blendshapes_data.items():
            for blendshapes_dict in data:
                row = [image_name] + [blendshapes_dict.get(key, 0) for key in sorted(list(all_keys))]
                writer.writerow(row)

if __name__ == "__main__":
    folder_path = './suprize'
    output_csv_path = 'suprize_faces_blendshapes.csv'

    # Initialize the face landmarker and face detector
    base_options = python.BaseOptions(model_asset_path='face_landmarker_v2_with_blendshapes.task')
    options = vision.FaceLandmarkerOptions(base_options=base_options, output_face_blendshapes=True, num_faces=1, min_face_detection_confidence=0.5)
    detector = vision.FaceLandmarker.create_from_options(options)

    model_file = "./Pytorch_RetinaFace_resnet50-640-640.onnx"
    retinaface = facedet.RetinaFace(model_file, model_format=ModelFormat.ONNX)

    blendshapes_data = {}

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # Check for image files
            image_path = os.path.join(folder_path, filename)
            data = detect_and_extract_blendshapes(image_path, detector, retinaface)
            blendshapes_data[filename] = data

    save_blendshapes_to_csv(blendshapes_data, output_csv_path)
    print(f"Blendshapes data saved to {output_csv_path}")
