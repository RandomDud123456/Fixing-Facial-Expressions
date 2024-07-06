# Facial Expression correction using Patch and Blend.
## Group photo

1. Ensure that the `path to group image` and `path to clusters` for each face which you plan to edit are available.

2. Following the usage of `driver.py`, run. Usage for `driver.py`:
```
usage: driver.py [-h] group_photo_path

Edit group photo with multiple faces using clusters.

positional arguments:
  group_photo_path  Path to the group photo

options:
  -h, --help        show this help message and exit
```

3. Each face present in the group is shown, and user is given option to choose whether to edit or not. If choosen to edit, provide the `path to cluster` of the corresponding face. And the `upper limit` of for the eye blink upto which you want to consider eyes. Rest are filtered out. 

4. Loop through each face and edit accordingly.

5. Final edited group photo in `edited_group_photo.jpg`.

## Base and Targets are manually choosen.

1. Run `patch_blend.py` according to the below usage:
```
usage: patch_blend.py [-h] base_path eyes_target_path mouth_target_path eyes_landmarks_path mouth_landmarks_path

Patching and Blending

positional arguments:
  base_path             Path to the base image.
  eyes_target_path      Path to the eyes target image
  mouth_target_path     Path to the mouth target image
  eyes_landmarks_path   Path to the text file containing landmarks of eye
  mouth_landmarks_path  Path to the text file containing landmarks of mouth

options:
  -h, --help            show this help message and exit
```

# General details of the codes and files

1. `driver.py`: Usage:
```
usage: driver.py [-h] group_photo_path

Edit group photo with multiple faces using clusters.

positional arguments:
  group_photo_path  Path to the group photo

options:
  -h, --help        show this help message and exit
```

2. `edit_face.py`: Usage:
```
usage: edit_face.py [-h] base_image_path cluster_folder_path

Edit base image with single face using cluster corresponding to it.

positional arguments:
  base_image_path      Path to the base image
  cluster_folder_path  Path to the folder containing cluster images

options:
  -h, --help           show this help message and exit
```

3. `patch_blend.py`: Usage:
```
usage: patch_blend.py [-h] base_path eyes_target_path mouth_target_path eyes_landmarks_path mouth_landmarks_path

Patching and Blending

positional arguments:
  base_path             Path to the base image.
  eyes_target_path      Path to the eyes target image
  mouth_target_path     Path to the mouth target image
  eyes_landmarks_path   Path to the text file containing landmarks of eye
  mouth_landmarks_path  Path to the text file containing landmarks of mouth

options:
  -h, --help            show this help message and exit
```

4. `pick_best_filter_eye.py`: Usage:
```
usage: pick_best_filter_eye.py [-h] [--threshold THRESHOLD] base_image_path cluster_folder_path

Sort the best and filter out the eyes

positional arguments:
  base_image_path       Path to the base image
  cluster_folder_path   Path to the folder containing cluster images

options:
  -h, --help            show this help message and exit
  --threshold THRESHOLD
                        Threshold for eyeBlinkLeft and eyeBlinkRight sum
```

5. `pick_best_filter_mouth.py`: Usage:
```
usage: pick_best_filter_mouth.py [-h] [--topK TOPK] base_image_path cluster_folder_path

Sort the best and filter out the eyes

positional arguments:
  base_image_path      Path to the base image
  cluster_folder_path  Path to the folder containing cluster images

options:
  -h, --help           show this help message and exit
  --topK TOPK          Number of entries in sorted order to sort according to smile scores.
```

6. `eyes.txt`: Contains the manually selected coordinates for extracting the patch of eyes. Coordinates are selected so as to give the best patch for realistic patch and blend later.

7. `mouth2.txt`: Contains manually selected coordinates for extracting the patch of mouth. Coordinates are selected so as to give the best patch for realistic patch and blend later.

8. `requirements.txt`: Contains the requirements for running the codes.

9. `models`: Folder containing the pretrained models for RetinaFace face detection and Mediapipe landmark detector.

10. `Datasets`: Contains datasets corresponding to different use cases. Each use case contains a `group photo` and `clusters of faces` corresponding to faces which we want to edit.

11. `file_structure.txt`: contains the file structure of current folder for reference.

12. `models.txt`: contains the link for pretrained models. Make sure 

13. `Datasets.txt`: contains the link for Datasets folder link in case `Datasets` folder is not present.
