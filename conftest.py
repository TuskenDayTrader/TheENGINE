"""
pytest configuration – ensures the repo root is on sys.path so that
`packages` and `apps` can be imported without fragile relative path hacks.
"""

import os
import sys

# Insert repo root once, idempotently
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
