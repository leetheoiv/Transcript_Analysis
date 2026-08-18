"""
tools/utils/load_schema_model.py

Dynamically loads and imports a Pydantic model class from a file path.
"""

import importlib.util
from pathlib import Path

def load_model_from_path(file_path, class_name):
    """Dynamically import a Pydantic model class from a Python file.

    Args:
        file_path: Path to the .py file containing the model definition.
        class_name: Name of the class to retrieve from the loaded module.

    Returns:
        The imported class object.
    """
    file_path = str(file_path)
    file_path = Path(file_path)

    spec = importlib.util.spec_from_file_location(file_path.stem, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return getattr(module, class_name)