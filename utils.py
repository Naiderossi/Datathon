"""Utility functions and constants shared across the project."""
from __future__ import annotations

import math
import re
import unicodedata
from typing import Iterable, Sequence, Mapping

from ast import literal_eval

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

LEVELS: list[str] = [
    "Sem conhecimento",
    "Básico",
    "Intermediário",
    "Avançado",
    "Fluente",
    "Nativo",
]

ACADEMICO: list[str] = [
    "Fundamental",
    "Médio",
    "Técnico",
    "Tecnólogo",
    "Graduação",
    "Pós-graduação",
    "Mestrado",
    "Doutorado",
]

_LVL_ALIAS = {
    "nenhum": 0,
    "sem conhecimento": 0,
    "none": 0,
    "basico": 1,
    "básico": 1,
    "basic": 1,
    "intermediario": 2,
    "intermediário": 2,
    "intermediate": 2,
    "avancado": 3,
    "avançado": 3,
    "advanced": 3,
    "fluente": 4,
    "fluency": 4,
    "nativo": 5,
    "native": 5,
}

_ACAD_ALIAS = {
    "fundamental": 0,
    "ensino fundamental": 0,
    "medio": 1,
    "médio": 1,
    "ensino medio": 1,
    "ensino médio": 1,
    "tecnico": 2,
    "técnico": 2,
    "tecnologo": 3,
    "tecnólogo": 3,
    "superior incompleto": 3,
    "graduacao": 4,
    "graduação": 4,
    "ensino superior": 4,
    "ensino superior completo": 4,
    "superior completo": 4,
    "pos": 5,
    "pós": 5,
    "pos-graduacao": 5,
    "pós-graduação": 5,
    "especializacao": 5,
    "especialização": 5,
    "mba": 5,
    "mestrado": 6,
    "doutorado": 7,
    "phd": 7,
}

_SAP_KEYWORDS = {"sap", "s/4hana", "s4hana", "hana", "abap"}

_HASH = HashingVectorizer(
    n_features=2**18,
    alternate_sign=False,
    norm="l2",
    ngram_range=(1, 2),
    lowercase=True,
)


def _norm(text: str | float | None) -> str:
    if text is None or (isinstance(text, float) and math.isnan(text)):
        return ""
    s = unicodedata.normalize("NFKD", str(text))
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def map_level(value: str | float | None) -> int:
    s = _norm(value)
    for key, val in _LVL_ALIAS.items():
        if key in s:
            return val
    for idx, name in enumerate(LEVELS):
        if _norm(name) in s:
            return idx
    if "fluente" in s:
        return 4
    return 0


def map_acad(value: str | float | None) -> int:
    s = _norm(value)
    for key, val in _ACAD_ALIAS.items():
        if key in s:
            return val
    for idx, name in enumerate(ACADEMICO):
        if _norm(name) in s:
            return idx
    if "ensino superior" in s:
        return 4
    return 0


def parse_req_lang(req_text: str, lang: str = "ingles") -> int:
    s = _norm(req_text)
    if lang == "ingles":
        pattern = r"ingles[^\w]?(basico|básico|intermediario|intermediário|avancado|avançado|fluente|nativo)"
    else:
        pattern = r"espanhol[^\w]?(basico|básico|intermediario|intermediário|avancado|avançado|fluente|nativo)"
    match = re.search(pattern, s)
    if match:
        return map_level(match.group(1))
    if (lang == "ingles" and "ingles" in s) or (lang == "espanhol" and "espanhol" in s):
        return 1
    return 0


def parse_req_acad(req_text: str) -> int:
    s = _norm(req_text)
    for key in [
        "doutorado",
        "mestrado",
        "pos",
        "pós",
        "especializacao",
        "especialização",
        "mba",
        "superior completo",
        "ensino superior completo",
        "graduacao",
        "graduação",
        "tecnico",
        "técnico",
        "medio",
        "médio",
        "fundamental",
    ]:
        if key in s:
            return map_acad(key)
    if "ensino superior" in s:
        return 4
    return 0


def parse_req_pcd(req_text: str) -> int:
    s = _norm(req_text)
    keywords = ["pcd", "pessoas com deficiencia", "pessoas com deficiência", "vaga pcd"]
    return int(any(keyword in s for keyword in keywords))


def parse_req_sap(req_text: str, vaga_sap_field: str | float | None = None) -> int:
    if vaga_sap_field is not None and str(vaga_sap_field).strip().lower() in {"sim", "true", "1", "yes"}:
        return 1
    s = _norm(req_text)
    return int(any(keyword in s for keyword in _SAP_KEYWORDS))


def safe_list_parse(values: str | Sequence[str] | None) -> list[str]:
    if values is None or (isinstance(values, float) and math.isnan(values)):
        return []
    if isinstance(values, (list, tuple, set)):
        return [str(v).strip() for v in values if str(v).strip()]
    text = str(values)
    if text in {"[]", ""}:
        return []
    try:
        parsed = literal_eval(text)
        if isinstance(parsed, (list, tuple, set)):
            return [str(v).strip() for v in parsed if str(v).strip()]
    except Exception:
        pass
    return [part.strip() for part in text.split(",") if part.strip()]


def detect_sap(skills: Iterable[str], skills_text: str, cv_text: str) -> int:
    bucket = " ".join(skills) + " " + (skills_text or "") + " " + (cv_text or "")
    return int(any(keyword in _norm(bucket) for keyword in _SAP_KEYWORDS))


def compute_skill_overlap(skills: Iterable[str], job_text: str) -> tuple[int, float]:
    skills_norm = {_norm(skill) for skill in skills if _norm(skill)}
    if not skills_norm:
        return 0, 0.0
    job_norm = _norm(job_text)
    hits = sum(1 for skill in skills_norm if skill and skill in job_norm)
    ratio = hits / max(1, len(skills_norm))
    return hits, ratio


def compute_token_overlap_metrics(cv_text: str, req_text: str) -> tuple[int, float]:
    cv_tokens = {token for token in _norm(cv_text).split() if token}
    req_tokens = {token for token in _norm(req_text).split() if token}
    if not req_tokens:
        return 0, 0.0
    overlap = cv_tokens.intersection(req_tokens)
    ratio = len(overlap) / max(1, len(req_tokens))
    return len(overlap), ratio


def pick_cv_text(row: Mapping[str, str] | dict) -> str:
    for col in ["cv_pt", "cv_pt_clean", "cv_pt_clean_noaccents", "cv_en"]:
        value = row.get(col)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def pick_req_text(row: Mapping[str, str] | dict) -> str:
    for col in ["req_text_clean", "req_text_clean_noaccents", "req_text"]:
        value = row.get(col)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def cosine_similarity_01(text_a: str, text_b: str) -> float:
    a = (text_a or "").strip()
    b = (text_b or "").strip()
    if not a or not b:
        return 0.0
    xa = _HASH.transform([a])
    xb = _HASH.transform([b])
    return float((xa @ xb.T).toarray()[0, 0])


__all__ = [
    'LEVELS',
    'ACADEMICO',
    'map_level',
    'map_acad',
    'parse_req_lang',
    'parse_req_acad',
    'parse_req_pcd',
    'parse_req_sap',
    'safe_list_parse',
    'detect_sap',
    'compute_skill_overlap',
    'compute_token_overlap_metrics',
    'pick_cv_text',
    'pick_req_text',
    'cosine_similarity_01',
]
