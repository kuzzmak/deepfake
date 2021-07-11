import os


def get_file_paths_from_dir(dir: str) -> list[str] or None:
    """Returns list of paths to files in directory. If no files
    are found in directory, None is returned.

    Parameters
    ----------
    dir : str
        directory with files

    Returns
    -------
    list[str] or None
        list of paths 
    """
    if not os.path.exists(dir):
        return None

    files = [f for f in os.listdir(
        dir) if os.path.isfile(os.path.join(dir, f))]
    curr_dir = os.path.abspath(dir)
    file_paths = [os.path.join(curr_dir, x) for x in files]

    return file_paths
