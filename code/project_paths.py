"""Resolve the original code-relative paths independently of the shell cwd.

This module only resolves paths. It does not create directories or change any
model, dataset label mapping, training configuration, or output filename.
"""
import os
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_DIR.parent


def resolve_code_path(path):
    """Resolve a legacy path such as ../weights against the code directory."""
    value = os.fspath(path)
    if os.path.isabs(value):
        return os.path.normpath(value)
    return os.path.normpath(os.path.join(str(CODE_DIR), value))
