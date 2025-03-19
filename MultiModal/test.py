import os
from pathlib import Path

CHROMA_PATH = "chroma_db"

# Method 1: Using os
if os.path.exists(CHROMA_PATH):
    print("Chroma directory exists")

# Method 2: Using Pathlib
if Path(CHROMA_PATH).exists():
    print("Chroma directory exists")