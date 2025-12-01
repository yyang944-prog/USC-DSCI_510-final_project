"""
I/O utilities: safe CSV read/write with folders, dtype control.
"""
from __future__ import annotations
from typing import Iterable
import os
import pandas as pd

def ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)

def to_csv(df: pd.DataFrame, path: str, index: bool = False) -> None:
    ensure_dir(path); df.to_csv(path, index=index)

def read_csv(path: str, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, **kwargs)
