"""MemVault — a file-first memory engine over a git repository of Markdown.

The version lives here rather than in `pyproject.toml`: the build backend reads it out of this
module, so a release is one edit in one place and the number a running process reports is the
number the wheel was built from.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
