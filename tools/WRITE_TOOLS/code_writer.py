"""
tools/WRITE_TOOLS/code_writer.py

Utilities for writing generated Python code to files and validating imports.
"""

import re
import importlib.util
from pathlib import Path
from typing import Type
from pydantic import BaseModel


def _to_snake_case(name: str) -> str:
    """Convert a CamelCase name to snake_case.

    Args:
        name: CamelCase string to convert.

    Returns:
        snake_case version of the input string.
    """
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def write_code(code: str, model_name: str, output_dir: str) -> Path:
    """Write Python code to a file and validate it is importable.

    Writes the code to {output_dir}/{snake_case(model_name)}.py, then attempts
    to import the module to verify syntactic and semantic correctness.

    Args:
        code: Python source code string to write.
        model_name: CamelCase model name (used to derive the filename).
        output_dir: Directory to write the file into.

    Returns:
        Path to the written file.

    Raises:
        Any exception thrown during import — caller handles retry logic.
    """
    path = Path(output_dir) / f"{_to_snake_case(model_name)}.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(code, encoding="utf-8")

    spec = importlib.util.spec_from_file_location("_validate_module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return path


def load_model(path: Path, model_name: str) -> Type[BaseModel]:
    """Dynamically import a generated Pydantic model class from a file.

    Args:
        path: Path to the Python file containing the model.
        model_name: Class name to retrieve from the loaded module.

    Returns:
        The Pydantic BaseModel subclass.
    """
    spec = importlib.util.spec_from_file_location(model_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, model_name)
