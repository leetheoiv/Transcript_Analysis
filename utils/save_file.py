"""
utils/save_file.py

Generic file-saving utility. Writes content to a specified path with any extension.
Creates parent directories if they don't exist.
"""

from pathlib import Path
from typing import Optional


def save_file(
    content: str,
    filename: str,
    output_dir: str,
    extension: str = ".md",
    overwrite: bool = True,
) -> Path:
    """
    Save text content to a file.

    Args:
        content: The text content to write.
        filename: Name of the file (without extension unless you want to override).
        output_dir: Directory to save the file in.
        extension: File extension (default: ".md"). Include the dot.
        overwrite: If False, appends a numeric suffix to avoid overwriting existing files.

    Returns:
        Path to the saved file.
    """
    if not extension.startswith("."):
        extension = f".{extension}"

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Strip extension from filename if user accidentally included it
    stem = filename.removesuffix(extension)

    file_path = output_path / f"{stem}{extension}"

    if not overwrite:
        counter = 1
        while file_path.exists():
            file_path = output_path / f"{stem}_{counter}{extension}"
            counter += 1

    file_path.write_text(content, encoding="utf-8")
    return file_path
