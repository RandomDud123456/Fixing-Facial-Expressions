import os
import subprocess

def run_webcam_main_py(folder_path):
    # Ensure the path ends with a '/'
    if not folder_path.endswith('/'):
        folder_path += '/'

    # List all files in the given directory
    files = os.listdir(folder_path)

    # Filter out the target and frame files
    target_files = [file for file in files if '_frame' not in file and file.endswith('.jpg')]

    # Sort the files to ensure matching pairs are processed together
    target_files.sort()

    # Execute main.py for each pair of frame and target files
    for target_file in target_files:
        target_path = os.path.join(folder_path, target_file)
        
        # Constructing the command to run main.py with the appropriate arguments
        command = f'python3 webcam_main.py --target "{target_path}"'
        print(f"Executing: {command}")
        
        # Use subprocess to execute the command
        subprocess.run(command, shell=True)

# Example usage
folder_path = "/home/saketh/IIITH Sem 4/DASS/Project/Modular Code/Mini_Dataset/292A2175"
run_webcam_main_py(folder_path)
