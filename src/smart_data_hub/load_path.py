import os


def get_path_in_dir(dir_name):
    """Get the absolute paths of all files in the specified directory and its subdirectories.

    Args:
        dir_name (str): path to a folder.

    Returns:
        list: a list of absolute paths to files.
    """

    real_path = os.path.realpath(os.path.dirname(__file__))
    dir_path = os.path.join(real_path, os.path.join(dir_name))

    file_paths = []
    # Walk through the directories to find files
    for root, _, files in os.walk(dir_path):
        for file_name in files:
            # Get the absolute path to the directory
            abs_path = os.path.realpath(os.path.join(root, file_name))
            file_paths.append(abs_path)

    return file_paths
