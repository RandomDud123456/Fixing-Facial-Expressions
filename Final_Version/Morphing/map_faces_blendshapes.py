import os
import mediapipe as mp
import cv2
import numpy as np

class FaceMapper:
    def __init__(self, method1_faces_dir, method2_faces_dir):
        self.method1_faces_dir = method1_faces_dir
        self.method2_faces_dir = method2_faces_dir
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)
    
    def calculate_distance(self, blendshapes1, blendshapes2):
        blendshapes1 = np.array(blendshapes1)
        blendshapes2 = np.array(blendshapes2)
        distance = np.linalg.norm(blendshapes1 - blendshapes2)
        return distance
    
    def extract_blendshapes(self, image_path):
        image = cv2.imread(image_path)
        results = self.face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            blendshapes = [landmark.x for landmark in landmarks.landmark]
            return blendshapes
        else:
            return None
    
    def match_faces(self, method1_blendshapes, method1_filenames, method2_blendshapes, method2_filenames):
        mappings = {}
        for i, (blendshapes1, filename1) in enumerate(zip(method1_blendshapes, method1_filenames)):
            min_distance = float('inf')
            matching_face = None
            for j, (blendshapes2, filename2) in enumerate(zip(method2_blendshapes, method2_filenames)):
                distance = self.calculate_distance(blendshapes1, blendshapes2)
                if distance < min_distance:
                    min_distance = distance
                    matching_face = j
            mappings[filename1] = method2_filenames[matching_face]
        return mappings
    
    def map_faces(self):
        method1_filenames = os.listdir(self.method1_faces_dir)
        method2_filenames = os.listdir(self.method2_faces_dir)
        
        method1_blendshapes = []
        for filename in method1_filenames:
            blendshapes = self.extract_blendshapes(os.path.join(self.method1_faces_dir, filename))
            if blendshapes:
                method1_blendshapes.append(blendshapes)
        
        method2_blendshapes = []
        for filename in method2_filenames:
            blendshapes = self.extract_blendshapes(os.path.join(self.method2_faces_dir, filename))
            if blendshapes:
                method2_blendshapes.append(blendshapes)
        
        mappings = self.match_faces(method1_blendshapes, method1_filenames, method2_blendshapes, method2_filenames)
        return mappings

# Usage
method1_faces_dir = '/home/saketh/IIITH Sem 4/DASS/Project/Modular Code/Mini_Dataset/292A2172/faces_extracted'
method2_faces_dir = '/home/saketh/IIITH Sem 4/DASS/Project/Modular Code/Mini_Dataset/292A2172/faces_dataset'

face_mapper = FaceMapper(method1_faces_dir, method2_faces_dir)
mappings = face_mapper.map_faces()

for method1_filename, method2_filename in mappings.items():
    print(f"{method1_filename} maps to {method2_filename}")