"""Put the repository root on sys.path so `src` imports work from anywhere.

Keeps the scripts runnable with a plain `python scripts/<name>.py` and no
install step or PYTHONPATH fiddling.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
