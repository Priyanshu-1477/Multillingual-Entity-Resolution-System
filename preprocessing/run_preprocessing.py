#!/usr/bin/env python3
"""Root wrapper for multilingual_preprocessing/run_preprocessing.py."""

import sys
from pathlib import Path

# Add multilingual_preprocessing to sys.path
BASE_DIR = Path(__file__).resolve().parent / "multilingual_preprocessing"
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from run_preprocessing import main

if __name__ == "__main__":
    main()
