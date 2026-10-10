#!/usr/bin/env python3
"""Portable entry point for the standalone TRA skill."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tra_alpha.cli import main


if __name__ == "__main__":
    main()
