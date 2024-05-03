import os
import shutil
import glob

def copy_directory(source, destination):
    """
    Copy all files and folders from source directory to destination directory.

    Parameters:
    source (str): The path to the source directory.
    destination (str): The path to the destination directory.
    """
    # Check if the source directory exists
    if not os.path.exists(source):
        print("Source directory does not exist.")
        return

    # Check if the destination directory exists; if not, create it
    if not os.path.exists(destination):
        os.makedirs(destination)

    print(f"Folder {source} copied to/as {destination}")
    # Iterate over all directories and files in the source directory
    for item in os.listdir(source):
        src_path = os.path.join(source, item)
        dst_path = os.path.join(destination, item)

        # If item is a directory, recursively call the function
        if os.path.isdir(src_path):
            copy_directory(src_path, dst_path)
        else:
            # If item is a file, copy it
            shutil.copy2(src_path, dst_path)


def copy_file(source_file, destination_folder):
    """
    Copies a single file from source_file to destination_folder.

    Parameters:
    source_file (str): The path to the source file to be copied.
    destination_folder (str): The path to the destination folder where the file should be copied.
    """
    # Import the os module to handle path operations

    # Create the destination folder if it doesn't exist
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Determine the destination file path
    destination_file = os.path.join(destination_folder, os.path.basename(source_file))

    # Copy the source file to the destination file
    shutil.copy2(source_file, destination_file)
    print(f"File copied from {source_file} to {destination_file}")

# Example usage
folder_path = sorted(glob.glob("/media/juy220/My Passport/Geo-Registration/Floor_2/20*"))
for source_directory in folder_path:
    destination_directory = '/data/juy220/LU Student Dropbox/Jun Yu/_Vinod/Indoor_Navi/Localization/data/HST_video/Jun/Floor_2/' + source_directory.split('/')[-1]

    copy_directory(os.path.join(source_directory, "sparse", "geo"), os.path.join(destination_directory, "sparse", "geo"))

    # copy_file(os.path.join(source_directory, "camera2world_6DoF.txt"), destination_directory)
    # copy_file(os.path.join(source_directory, "cameras.txt"), destination_directory)
    copy_file(os.path.join(source_directory, "images.txt"), destination_directory)
    # copy_file(os.path.join(source_directory, "points3D.txt"), destination_directory)
    copy_file(os.path.join(source_directory, "geo_coord.txt"), destination_directory)