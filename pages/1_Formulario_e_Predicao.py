from __future__ import annotations

import sys
from pathlib import Path
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from talent_matching import render_app

st.set_page_config(page_title="Formulário e Predição", page_icon=":memo:")

render_app(section="form")


