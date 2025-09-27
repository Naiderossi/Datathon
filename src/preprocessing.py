"""Dataset loading and preprocessing utilities."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple,Set
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
# URLs dos arquivos no Google Drive
# ---------------------------
URL_APPLICANTS = "https://drive.google.com/uc?id=1bsRtUSZaYSScpkDfluP45bHxnoVe8tKm"
URL_JOBS = "https://drive.google.com/uc?id=1cH8Yebtk58xhox7FMypSlEOOXfNMMPFZ"
URL_PROSPECTS = "https://drive.google.com/uc?id=1BeSSet5NhCY5axY6Gr2FLaUVONrFKHJ0"
URL_VAGAS_JSON = "https://drive.google.com/uc?id=1cH8Yebtk58xhox7FMypSlEOOXfNMMPFZ"

DATASETS_DIR = Path('datasets')
DATASETS_DIR.mkdir(exist_ok=True)

# ---------------------------
# Função auxiliar para baixar arquivos se não existirem
# ---------------------------
def download_if_missing(url: str, path: Path):
    if not path.exists():
        print(f"Baixando {path.name} do Google Drive...")
        gdown.download(url, str(path), quiet=False)
    else:
        print(f"Arquivo {path.name} já existe localmente.")

# ---------------------------
# Carregar Applicants
# ---------------------------
def load_applicants(path: Path | str = DATASETS_DIR / 'df_applicants.csv') -> pd.DataFrame:
    download_if_missing(URL_APPLICANTS, Path(path))
    df = pd.read_csv(path)
    df = df.drop_duplicates('candidate_id').set_index('candidate_id')
    for col in ['skills_text', 'cv_pt', 'cv_pt_clean', 'cv_pt_clean_noaccents', 'cv_en']:
        if col in df.columns:
            df[col] = df[col].fillna('')
    df['skills_list'] = df.get('skills_list', '[]').apply(safe_list_parse)
    df['cv_len_tokens'] = df.get('cv_len_tokens', 0).fillna(0)
    return df


# ---------------------------
# Carregar Jobs
# ---------------------------
def load_jobs(path: Path | str = DATASETS_DIR / 'df_jobs.csv') -> pd.DataFrame:
    download_if_missing(URL_JOBS, Path(path))
    df = pd.read_csv(path)
    df = df.drop_duplicates('job_id').set_index('job_id')
    for col in ['req_text_clean', 'req_text_clean_noaccents', 'req_text']:
        if col in df.columns:
            df[col] = df[col].fillna('')
    df['req_len_tokens'] = df.get('req_len_tokens', 0).fillna(0)
    return df

# ---------------------------
# Carregar Prospects
# ---------------------------
def load_prospects(path: Path | str = DATASETS_DIR / 'df_prospects.csv') -> pd.DataFrame:
    download_if_missing(URL_PROSPECTS, Path(path))
    return pd.read_csv(path)

# ---------------------------
# Carregar todos
# ---------------------------
def load_all(extra_jobs: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, pd.DataFrame, Set[int]]:
    apps = load_applicants()
    jobs = load_jobs()
    prospects = load_prospects()
    prospect_ids = set(prospects['candidate_id'])
    return apps, jobs, prospect_ids

    if extra_jobs is not None and not extra_jobs.empty:
        extra_jobs = extra_jobs.copy()
        if 'job_id' in extra_jobs.columns:
            extra_jobs['job_id'] = pd.to_numeric(extra_jobs['job_id'], errors='coerce')
            extra_jobs = extra_jobs.dropna(subset=['job_id'])
            extra_jobs['job_id'] = extra_jobs['job_id'].astype(int)
            for col in ['req_text_clean', 'req_text_clean_noaccents', 'req_text']:
                if col in extra_jobs.columns:
                    extra_jobs[col] = extra_jobs[col].fillna('')
            extra_jobs['req_len_tokens'] = extra_jobs.get('req_len_tokens', 0).fillna(0)
            jobs = pd.concat([jobs.reset_index(), extra_jobs], ignore_index=True, sort=False)
            jobs = jobs.drop_duplicates('job_id').set_index('job_id')

    return apps, jobs, prospect_ids


def load_vagas_json(base_dir: Path | str = 'datasets', uploaded_file=None) -> Tuple[pd.DataFrame, dict]:
    base_dir = Path(base_dir)
    base_dir.mkdir(exist_ok=True)

    if uploaded_file is not None:
        # Se o usuário fez upload do arquivo
        vagas = json.load(uploaded_file)
    else:
        # Caminho local do JSON
        vagas_path = base_dir / 'vagas.json'
        
        # Baixa do Google Drive se não existir
        download_if_missing(URL_VAGAS_JSON, vagas_path)
        
        try:
            vagas = json.loads(vagas_path.read_text(encoding='utf-8'))
        except Exception as e:
            return pd.DataFrame(), {'error': f'Erro ao ler o arquivo: {e}'}

    # Extrai informações e monta DataFrame
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
    'load_applicants',
    'load_jobs',
    'load_prospects',
    'load_all',
    'load_vagas_json',
]


