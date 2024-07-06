"""
    run: python3 categorize.py {group_photo_ID}
    Input: group photo name
    Output: folder with folder_name as group_photo_name containing group photo and corresponding faces
"""
import os
import shutil
import argparse

def search_and_organize_files(starting_filename, search_directories):
    """
        Arguments:
        arg1: group photo ID
        arg2: directories that are needed to be searched.
        List of directories is also accepted
        Function: creates a new folder named with same ID, and
        copies the group photo and extracted faces into it.
    """
    # Create the target directory for the found files, if it doesn't exist
    target_directory = os.path.join(os.getcwd(), starting_filename)
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)

    for directory in search_directories:
        # Walk through all directories and subdirectories
        # for root, dirs, files in os.walk(directory):
        for root, _, files in os.walk(directory):
            for filename in files:
                if filename.startswith(starting_filename):
                    source_path = os.path.join(root, filename)
                    destination_path = os.path.join(target_directory, filename)

                    # Copy the file to the new directory
                    shutil.copy(source_path, destination_path)
                    print(f"Copied '{filename}' to '{target_directory}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""Search for files starting with a given filename
                                     across all directories and subdirectories,
                                     and organize them into a new folder named after the
                                     filename by copying them.""")
    parser.add_argument('filename', type=str, help='The starting filename to search for')
    args = parser.parse_args()

    search_directories_list = ['faces_rec', 'upload']  # Directories to search in
    search_and_organize_files(args.filename, search_directories_list)
