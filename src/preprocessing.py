"""Dataset loading and preprocessing utilities."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Set
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

# Pegar credenciais do secrets
os.environ["KAGGLE_USERNAME"] = st.secrets["kaggle"]["username"]
os.environ["KAGGLE_KEY"] = st.secrets["kaggle"]["key"]


from kaggle.api.kaggle_api_extended import KaggleApi
# Autenticar
api = KaggleApi()
api.authenticate()

# Baixar dataset privado
api.dataset_download_files(
    "naiaraderossi/DatathonDataset",
    path="data",
    unzip=True
)

# ---------------------------
# Baixar dataset privado
# ---------------------------
DATASETS_DIR = Path("data")
DATASETS_DIR.mkdir(exist_ok=True)

# ---------------------------
# Arquivos CSV baixados
# ---------------------------
PATH_APPLICANTS = DATASETS_DIR / "df_applicants.csv"
PATH_JOBS = DATASETS_DIR / "df_jobs.csv"
PATH_PROSPECTS = DATASETS_DIR / "df_prospects.csv"
PATH_VAGAS = DATASETS_DIR/ "vagas.json"
# ---------------------------
# Carregar Applicants
# ---------------------------
def load_applicants() -> pd.DataFrame:
    if not PATH_APPLICANTS.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {PATH_APPLICANTS}")
    df = pd.read_csv(PATH_APPLICANTS)
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
    if not PATH_JOBS.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {PATH_JOBS}")
    df = pd.read_csv(PATH_JOBS)
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
    if not PATH_PROSPECTS.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {PATH_PROSPECTS}")
    return pd.read_csv(PATH_PROSPECTS)

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
# Carregar Vagas 
# ---------------------------
def load_vagas_json(base_dir: Path | str = 'datasets', uploaded_file=None) -> tuple[pd.DataFrame, dict]:
    base_dir = Path(base_dir)
    if uploaded_file is not None:
        vagas = json.load(uploaded_file)
    else:
        vagas_path = PATH_VAGAS
        if not vagas_path.exists():
            return pd.DataFrame(), {'error': f'Arquivo não encontrado: {vagas_path}'}
        vagas = json.loads(vagas_path.read_text(encoding='utf-8'))

    rows = []
    for job_id, data in vagas.items():
        informacoes = data.get('informacoes_basicas', {})
        perfil = data.get('perfil_vaga', {})
        req_text = ' '.join([
            str(informacoes.get('titulo_vaga', '')),
            str(informacoes.get('descricao_vaga', '')),
            str(perfil.get('principais_atividades', '')),
            str(perfil.get('competencia_tecnicas_e_comportamentais', '')),
            str(perfil.get('conhecimentos_tecnicos', '')),
            str(perfil.get('requisitos_desejaveis', ''))
        ]).strip()
        rows.append({
            'job_id': str(job_id),
            'titulo': informacoes.get('titulo_vaga', ''),
            'cliente': informacoes.get('cliente', ''),
            'req_text_clean': req_text,
            'vaga_sap_raw': informacoes.get('vaga_sap', ''),
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











