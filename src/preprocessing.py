"""Dataset loading and preprocessing utilities."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Set
import gdown
import json
import pandas as pd

from .utils import (
    map_level,
    parse_req_acad,
    parse_req_lang,
    parse_req_pcd,
    parse_req_sap,
    safe_list_parse,
)


# Caminhos para arquivos locais
FILE_APPLICANTS = Path("datasets/df_applicants.parquet")
FILE_JOBS = Path("datasets/df_jobs.parquet")
FILE_PROSPECTS = Path("datasets/df_prospects.parquet")

# ---------------------------
# Carregar Applicants
# ---------------------------
def load_applicants() -> pd.DataFrame:
    if not FILE_APPLICANTS.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {FILE_APPLICANTS}")
    df = pd.read_parquet(FILE_APPLICANTS)
    df = df.drop_duplicates("candidate_id").set_index("candidate_id")
    for col in ["skills_text", "cv_pt", "cv_pt_clean", "cv_pt_clean_noaccents", "cv_en"]:
        if col in df.columns:
            df[col] = df[col].fillna("")
    df["skills_list"] = df.get("skills_list", "[]").apply(safe_list_parse)
    df["cv_len_tokens"] = df.get("cv_len_tokens", 0).fillna(0)
    return df


# ---------------------------
# Carregar Jobs
# ---------------------------
def load_jobs() -> pd.DataFrame:
    if not FILE_JOBS.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {FILE_JOBS}")
    df = pd.read_parquet(FILE_JOBS)
    df = df.drop_duplicates("job_id").set_index("job_id")
    for col in ["req_text_clean", "req_text_clean_noaccents", "req_text"]:
        if col in df.columns:
            df[col] = df[col].fillna("")
    df["req_len_tokens"] = df.get("req_len_tokens", 0).fillna(0)
    return df


# ---------------------------
# Carregar Prospects
# ---------------------------
def load_prospects() -> pd.DataFrame:
    if not FILE_PROSPECTS.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {FILE_PROSPECTS}")
    return pd.read_parquet(FILE_PROSPECTS)

# ---------------------------
# Carregar todos
# ---------------------------
def load_all(extra_jobs: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, pd.DataFrame, Set[int]]:
    apps = load_applicants()
    jobs = load_jobs()
    prospects = load_prospects()
    prospect_ids = set(prospects["candidate_id"])

    if extra_jobs is not None and not extra_jobs.empty:
        extra_jobs = extra_jobs.copy()
        if "job_id" in extra_jobs.columns:
            extra_jobs["job_id"] = pd.to_numeric(extra_jobs["job_id"], errors="coerce")
            extra_jobs = extra_jobs.dropna(subset=["job_id"])
            extra_jobs["job_id"] = extra_jobs["job_id"].astype(int)
            for col in ["req_text_clean", "req_text_clean_noaccents", "req_text"]:
                if col in extra_jobs.columns:
                    extra_jobs[col] = extra_jobs[col].fillna("")
            extra_jobs["req_len_tokens"] = extra_jobs.get("req_len_tokens", 0).fillna(0)
            jobs = pd.concat([jobs.reset_index(), extra_jobs], ignore_index=True, sort=False)
            jobs = jobs.drop_duplicates("job_id").set_index("job_id")

    return apps, jobs, prospect_ids

# ---------------------------
# Carregar Vagas Parquet
# ---------------------------
FILE_VAGAS = Path("datasets/df_vagas.parquet")

def load_vagas_parquet() -> pd.DataFrame:
    if not FILE_VAGAS.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {FILE_VAGAS}")
    
    df = pd.read_parquet(FILE_VAGAS)
    
    # Garantir colunas obrigatórias
    for col in ["titulo", "cliente", "req_text_clean", "vaga_sap_raw"]:
        if col not in df.columns:
            df[col] = ""
    
    # Preencher valores nulos
    df = df.fillna({
        "titulo": "",
        "cliente": "",
        "req_text_clean": "",
        "vaga_sap_raw": ""
    })

    return df
    
__all__ = [
    "load_applicants",
    "load_jobs",
    "load_prospects",
    "load_all",
    "load_vagas_json",
]







