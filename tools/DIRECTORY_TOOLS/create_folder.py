"""
tools/DIRECTORY_TOOLS/create_folder.py

Utility for creating project output directories.
"""

from pathlib import Path


def create_folder(base_path: str, folder_name: str) -> Path:
    """Create a directory at base_path/folder_name if it doesn't already exist.

    Args:
        base_path: Parent directory path.
        folder_name: Name of the folder to create.

    Returns:
        Path object pointing to the created (or existing) directory.

    Raises:
        FileExistsError: If a file (not directory) already exists at the target path.
    """
    folder_path = Path(base_path) / folder_name

    if folder_path.exists():
        if not folder_path.is_dir():
            raise FileExistsError(f"A file exists at: {folder_path}")
    else:
        folder_path.mkdir(parents=True)

    return folder_path