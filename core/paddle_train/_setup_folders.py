"""Setup script to create required folder structure"""
import os
from pathlib import Path

base_dir = Path(__file__).parent

folders = [
    "configs",
    "data",
    "data/det",
    "data/rec",
    "data/rec/crops",
    "data/logs",
    "output",
]

for folder in folders:
    folder_path = base_dir / folder
    folder_path.mkdir(parents=True, exist_ok=True)
    print(f"Created: {folder_path}")

print("\n✅ Folder structure created successfully!")

