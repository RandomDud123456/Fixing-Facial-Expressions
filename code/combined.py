import cv2
import os
from fastdeploy.vision import facedet
from fastdeploy.runtime import RuntimeOption, ModelFormat
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

def draw_landmarks_on_image(rgb_image, detection_result):
  face_landmarks_list = detection_result.face_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected faces to visualize.
  for idx in range(len(face_landmarks_list)):
    face_landmarks = face_landmarks_list[idx]

    # Draw the face landmarks.
    face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    face_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
    ])

    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
        .get_default_face_mesh_tesselation_style())
    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
        .get_default_face_mesh_contours_style())
    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_IRISES,
          landmark_drawing_spec=None,
          connection_drawing_spec=mp.solutions.drawing_styles
          .get_default_face_mesh_iris_connections_style())

  return annotated_image

def plot_face_blendshapes_bar_graph(face_blendshapes, output_plot_path):
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
  plt.tight_layout()
#   plt.show()
  # plt.savefig(output_plot_path)
  

if __name__ == "__main__":
    # STEP 2: Create an FaceLandmarker object.
    base_options = python.BaseOptions(model_asset_path='face_landmarker_v2_with_blendshapes.task')
    options = vision.FaceLandmarkerOptions(base_options=base_options,
                                        output_face_blendshapes=True,
                                        output_facial_transformation_matrixes=True,
                                        num_faces=10,
                                        min_face_detection_confidence=0.5)
    detector = vision.FaceLandmarker.create_from_options(options)


    # Load RetinaFace model
    model_file = "./Pytorch_RetinaFace_resnet50-640-640.onnx"
    retinaface = facedet.RetinaFace(model_file, runtime_option=None, model_format=ModelFormat.ONNX)

    # Load the image
    image_path = "./test_lite_face_detector_3.jpg"
    directory, filename = os.path.split(image_path)
    image = cv2.imread(image_path)

    # Detect faces with RetinaFace
    results = retinaface.predict(image)

    created_files = []

    for i in range(len(results.boxes)):
        box = results.boxes[i]
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        
        # Crop the face region
        face_image = image[y1:y2, x1:x2]
        face_label = f'Face{i + 1}'
        output_face_path = f'Face{i + 1}' + filename
        cv2.imwrite(output_face_path, face_image)

        face_input = mp.Image.create_from_file(output_face_path)
        directory_face, filename_face = os.path.split(output_face_path)
        new_filename = "landmark_" + filename_face
        face_output_path = os.path.join(directory, new_filename)

        detection_result = detector.detect(face_input)
        annotated_image = draw_landmarks_on_image(face_input.numpy_view(), detection_result)

        cv2.imwrite(face_output_path, cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

        # created_files.append(output_face_path)
        created_files.append(face_output_path)

        # plot_face_blendshapes_bar_graph(detection_result.face_blendshapes[0])

        # print(detection_result.facial_transformation_matrixes)

        image[y1:y2, x1:x2] = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)


        # Before replacing the face in the original image, add the label
        label_position = (x1, y1 - 10)  # Positioning the label above the face bounding box
        font_scale = 0.3
        font_color = (0, 255, 0)  # White color
        line_type = 1

        cv2.putText(image, face_label, label_position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, line_type)
        plot_face_blendshapes_bar_graph(detection_result.face_blendshapes[0], output_face_path)


    cv2.imwrite("finalOutput2.jpg", image)


    # Now delete the intermediate files
    for file_path in created_files:
        try:
            os.remove(file_path)
            # print(f"Deleted {file_path}")
        except Exception as e:
            # print(f"Error deleting {file_path}: {e}")
           pass
