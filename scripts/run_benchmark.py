"""Thin shim so `python scripts/run_benchmark.py` still works."""

from benchmark.__main__ import main

if __name__ == "__main__":
    main()
