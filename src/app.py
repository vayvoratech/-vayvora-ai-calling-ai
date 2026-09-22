import os
import sys
from pathlib import Path

# Ensure root directory is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Re-export and execute app logic
from app import *

if __name__ == "__main__" or is_running_in_streamlit():
    if is_running_in_streamlit():
        run_streamlit_mode()
    else:
        run_cli_mode()