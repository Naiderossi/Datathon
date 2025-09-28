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

# ---------------------------
# IDs dos arquivos no Google Drive
# ---------------------------
ID_APPLICANTS = "18QTiuVFUz3i1xO9bXk9GDZeE6uoX__fb"
ID_JOBS = "1dKkSt5PL-tvCyZOqJvDjTDScFVQgms4c"
ID_PROSPECTS = "1uU3N7XANV_jvHAaWIRFOrqlhkXd-jmRz"
ID_VAGAS_JSON = "1spB6LjvkGBOXQQOOmllV4S5zC1CfWnIw" 
# ---------------------------
# Função auxiliar para baixar arquivos do Google Drive
# ---------------------------
def download_from_drive(file_id: str, filename: str) -> Path:
    path = Path("/tmp") / filename
    if not path.exists():
        url = f"https://drive.google.com/uc?id={file_id}"
        print(f"Baixando {filename} do Google Drive...")
        gdown.download(url, str(path), quiet=False, fuzzy=True)
    return path


# ---------------------------
# Carregar Applicants
# ---------------------------
def load_applicants() -> pd.DataFrame:
    path = download_from_drive(ID_APPLICANTS, "df_applicants.parquet")
    df = pd.read_parquet(path)
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
    path = download_from_drive(ID_JOBS, "df_jobs.parquet")
    df = pd.read_parquet(path)
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
    path = download_from_drive(ID_PROSPECTS, "df_prospects.parquet")
    return pd.read_parquet(path)


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
# Carregar Vagas JSON
# ---------------------------
def load_vagas_json(uploaded_file=None) -> Tuple[pd.DataFrame, dict]:
    if uploaded_file is not None:
        vagas = json.load(uploaded_file)
    else:
        path = download_from_drive(ID_VAGAS_JSON, "vagas.json")
        try:
            vagas = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            return pd.DataFrame(), {"error": f"Erro ao ler o arquivo: {e}"}

    rows = []
    for job_id, data in vagas.items():
        informacoes = data.get("informacoes_basicas", {})
        perfil = data.get("perfil_vaga", {})
        req_text = " ".join([
            str(informacoes.get("titulo_vaga", "")),
            str(informacoes.get("descricao_vaga", "")),
            str(perfil.get("principais_atividades", "")),
            str(perfil.get("competencia_tecnicas_e_comportamentais", "")),
            str(perfil.get("conhecimentos_tecnicos", "")),
            str(perfil.get("requisitos_desejaveis", ""))
        ]).strip()
        rows.append({
            "job_id": str(job_id),
            "titulo": informacoes.get("titulo_vaga", ""),
            "cliente": informacoes.get("cliente", ""),
            "req_text_clean": req_text,
            "vaga_sap_raw": informacoes.get("vaga_sap", ""),
        })

    df = pd.DataFrame(rows)
    return df, {}


__all__ = [
    "load_applicants",
    "load_jobs",
    "load_prospects",
    "load_all",
    "load_vagas_json",
]






