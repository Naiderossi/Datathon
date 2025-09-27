"""Convenience entrypoint that proxies to the talent matching page."""
from __future__ import annotations

from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parent
TARGET = ROOT / "app" / "pages" / "2_Sugestao_de_Candidatos.py"

if __name__ == "__main__":
    runpy.run_path(str(TARGET), run_name="__main__")
