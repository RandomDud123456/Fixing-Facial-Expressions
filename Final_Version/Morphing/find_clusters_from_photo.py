"""
    Run: python3 find_clusters_from_photo.py {group_photo_ID}
    Input: Group Photo name
    Output: faces_filename.jpg found in cluster_Y
"""

import os
import argparse
import re

def find_files_and_print_clusters(starting_pattern, search_directory):
    """
        map and print the clusters belonging to each face so that they
        can be used for morphing and stitching.
    """
    # Case insensitive search for .JPG files
    pattern = re.compile(rf"^{starting_pattern}_\d+\.JPG$", re.IGNORECASE)

    # for root, dirs, files in os.walk(search_directory):
    for root, _, files in os.walk(search_directory):
        for filename in files:
            if pattern.match(filename):
                # Extract "cluster_Y" from the path
                cluster_folder = os.path.basename(root)
                print(f"{filename} found in {cluster_folder}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""Search for files matching a specific pattern
                                     within a directory structure and print their names along
                                     with the corresponding 'cluster_Y' folder.""")
    parser.add_argument('filename', type=str, help='''The starting filename pattern
                        to search for, without the _X part.''')
    args = parser.parse_args()

    # Assuming 'faces_rec' is the main directory to search in
    # search_directory_str = 'faces_rec'
    SEARCH_DIRECTORY_STR = 'faces_rec'
    find_files_and_print_clusters(args.filename, SEARCH_DIRECTORY_STR)
