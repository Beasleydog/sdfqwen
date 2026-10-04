"""Small, dependency-free payload for the remote execution smoke test."""

import json
import os
import platform
import shutil


if __name__ == "__main__":
    print(json.dumps({
        "message": "copyme.py ran successfully on Prime Intellect",
        "system": platform.system(),
        "kernel": platform.release(),
        "architecture": platform.machine(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "disk_free_bytes": shutil.disk_usage("/").free,
        "working_directory": os.getcwd(),
    }, indent=2))
