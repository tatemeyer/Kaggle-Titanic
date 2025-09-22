# src/utils.py
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUT_DIR = PROJECT_ROOT

def data_path(name: str) -> Path:
    return DATA_DIR / name

def out_path(name: str) -> Path:
    OUT_DIR.mkdir(exist_ok=True, parents=True)
    return OUT_DIR / name
