import face_recognition
import os
import numpy as np

class FaceMapper:
    def __init__(self, method1_faces_dir, method2_faces_dir):
        self.method1_faces_dir = method1_faces_dir
        self.method2_faces_dir = method2_faces_dir
    
    def detect_embeddings(self, image_path):
        image = face_recognition.load_image_file(image_path)
        face_locations = face_recognition.face_locations(image)
        face_embeddings = face_recognition.face_encodings(image, face_locations)
        return face_embeddings
    
    def match_faces(self, method1_faces, method1_filenames, method2_faces, method2_filenames):
        mappings = {}
        for i, (face1, filename1) in enumerate(zip(method1_faces, method1_filenames)):
            min_distance = float('inf')
            matching_face = None
            for j, (face2, filename2) in enumerate(zip(method2_faces, method2_filenames)):
                distance = face_recognition.face_distance(np.array([face1]), np.array(face2))[0]
                if distance < min_distance:
                    min_distance = distance
                    matching_face = j
            mappings[filename1] = method2_filenames[matching_face]
        return mappings
    
    def map_faces(self):
        method1_filenames = os.listdir(self.method1_faces_dir)
        method2_filenames = os.listdir(self.method2_faces_dir)
        
        method1_embeddings = []
        for filename in method1_filenames:
            embeddings = self.detect_embeddings(os.path.join(self.method1_faces_dir, filename))
            if embeddings:
                method1_embeddings.append(embeddings[0])
        
        method2_embeddings = []
        for filename in method2_filenames:
            embeddings = self.detect_embeddings(os.path.join(self.method2_faces_dir, filename))
            if embeddings:
                method2_embeddings.append(embeddings[0])
        
        mappings = self.match_faces(method1_embeddings, method1_filenames, method2_embeddings, method2_filenames)
        return mappings

# Usage
method1_faces_dir = '/home/saketh/IIITH Sem 4/DASS/Project/Modular Code/Mini_Dataset/292A2172/faces_extracted'
method2_faces_dir = '/home/saketh/IIITH Sem 4/DASS/Project/Modular Code/Mini_Dataset/292A2172/faces_dataset'

face_mapper = FaceMapper(method1_faces_dir, method2_faces_dir)
mappings = face_mapper.map_faces()

for method1_filename, method2_filename in mappings.items():
    print(f"{method1_filename} maps to {method2_filename}")