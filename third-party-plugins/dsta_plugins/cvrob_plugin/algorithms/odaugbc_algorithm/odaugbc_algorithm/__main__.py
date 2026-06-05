"""
Allow augmentation_by_class_algorithm to be executable through `python3 -m augmentation_by_class_algorithm`.
"""

import sys
from importlib.metadata import version
from pathlib import Path

from augmentation_by_class_algorithm.plugin_init import run


def main() -> None:
    """
    Print the version of test engine core
    """
    print("*" * 20)
    print(version_msg())
    print("*" * 20)
    # invoke algorithm
    run()


def version_msg():
    """
    Return the accumulated_local_effect version, location and Python powering it.
    """
    python_version = sys.version
    location = Path(__file__).resolve().parent.parent

    return f"augmentation_by_class_algorithm - {version('augmentation_by_class_algorithm')} from \
        {location} (Python {python_version})"


if __name__ == "__main__":
    main()
