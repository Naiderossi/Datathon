from __future__ import annotations

from ast import literal_eval
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Home - Dashboard", page_icon=":bar_chart:", layout="wide")
st.title(":bar_chart: Visão Geral do Banco de Talentos")

# Links diretos do Google Drive
URL_APPLICANTS = "https://drive.google.com/uc?id=1Nr1iMwYy-tFqzWpvd2PJuDnYLY1Kv459"
URL_JOBS = "https://drive.google.com/uc?id=1cH8Yebtk58xhox7FMypSlEOOXfNMMPFZ"
URL_PROSPECTS = "https://drive.google.com/uc?id=1BeSSet5NhCY5axY6Gr2FLaUVONrFKHJ0"

ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASETS_DIR = ROOT_DIR / "datasets"

st.sidebar.header("Configurações")
data_dir_input = st.sidebar.text_input(
    "Diretório dos datasets",
    value=str(DEFAULT_DATASETS_DIR),
    help="Caminho para os arquivos df_applicants.csv, df_jobs.csv e df_prospects.csv."
)
data_dir = Path(data_dir_input).expanduser()

if not data_dir.exists():
    st.error(f"Diretório não encontrado: {data_dir}")
    st.stop()


@st.cache_data(show_spinner=False)
def load_datasets(from_drive: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if from_drive:
        apps = pd.read_csv(URL_APPLICANTS)
        jobs = pd.read_csv(URL_JOBS)
        prospects = pd.read_csv(URL_PROSPECTS)
    else:
        ROOT_DIR = Path(__file__).resolve().parent
        DEFAULT_DATASETS_DIR = ROOT_DIR / "datasets"
        apps = pd.read_csv(DEFAULT_DATASETS_DIR / "df_applicants.csv")
        jobs = pd.read_csv(DEFAULT_DATASETS_DIR / "df_jobs.csv")
        prospects = pd.read_csv(DEFAULT_DATASETS_DIR / "df_prospects.csv")
    return apps, jobs, prospects

# Carregar datasets
apps, jobs, prospects = load_datasets(from_drive=use_drive)
st.success("✅ Dados carregados com sucesso do Google Drive!" if use_drive else "✅ Dados carregados do diretório local")


##try:
##    apps, jobs, prospects = load_datasets(str(data_dir))
##except FileNotFoundError as exc:
##    st.error(f"Arquivo não encontrado: {exc}")
##    st.stop()

st.caption(
    "Os dados abaixo utilizam `df_applicants.csv`, `df_jobs.csv` e `df_prospects.csv` no diretório informado."
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Candidatos únicos", f"{apps['candidate_id'].nunique():,}".replace(",", "."))
col2.metric("Vagas mapeadas", f"{jobs['job_id'].nunique():,}".replace(",", "."))
col3.metric("Processos registrados", f"{len(prospects):,}".replace(",", "."))
perc_pcd = apps['pcd'].fillna('').astype(str).str.lower().isin({'sim', 'true', '1'}).mean() * 100
col4.metric("Candidatos PCD (%)", f"{perc_pcd:0.1f}%")

st.divider()

apps_levels = apps['nivel_academico'].fillna('Não informado').astype(str).str.title()
levels_df = apps_levels.value_counts().reset_index()
levels_df.columns = ['Formação', 'Quantidade']
levels_chart = alt.Chart(levels_df).mark_bar(color='#4b8bbe').encode(
    x=alt.X('Quantidade:Q', title='Quantidade'),
    y=alt.Y('Formação:N', sort='-x', title='Formação acadêmica')
)

jobs_ing = jobs['nivel_ingles_req'].fillna('Não informado').astype(str).str.title()
jobs_ing_df = jobs_ing.value_counts().reset_index()
jobs_ing_df.columns = ['Inglês requerido', 'Vagas']
jobs_chart = alt.Chart(jobs_ing_df).mark_bar(color='#f39c12').encode(
    x=alt.X('Inglês requerido:N', sort='-y'),
    y=alt.Y('Vagas:Q')
)

pros_status = prospects['situacao_norm'].fillna('Não informado').astype(str)
status_df = pros_status.value_counts().head(10).reset_index()
status_df.columns = ['Situação', 'Processos']
status_chart = alt.Chart(status_df).mark_bar(color='#16a085').encode(
    x=alt.X('Processos:Q'),
    y=alt.Y('Situação:N', sort='-x')
)

c1, c2, c3 = st.columns(3)
c1.altair_chart(levels_chart, use_container_width=True)
c2.altair_chart(jobs_chart, use_container_width=True)
c3.altair_chart(status_chart, use_container_width=True)

st.divider()

apps['skills_list'] = apps['skills_list'].fillna('[]').apply(
    lambda x: literal_eval(x) if isinstance(x, str) else x
)
skill_series = apps['skills_list'].explode().dropna().astype(str).str.strip()
skills_df = skill_series[skill_series != ''].str.title().value_counts().head(15).reset_index()
skills_df.columns = ['Skill', 'Candidatos']
skills_chart = alt.Chart(skills_df).mark_bar(color='#9b59b6').encode(
    x=alt.X('Candidatos:Q'),
    y=alt.Y('Skill:N', sort='-x')
)

jobs_clients = jobs['cliente'].fillna('Não informado').astype(str).str.title().value_counts().head(15).reset_index()
jobs_clients.columns = ['Cliente', 'Vagas']
clients_chart = alt.Chart(jobs_clients).mark_bar(color='#2980b9').encode(
    x=alt.X('Vagas:Q'),
    y=alt.Y('Cliente:N', sort='-x')
)

col_a, col_b = st.columns(2)
col_a.altair_chart(skills_chart, use_container_width=True)
col_b.altair_chart(clients_chart, use_container_width=True)

st.divider()

apps['nivel_ingles'] = apps['nivel_ingles'].fillna('Não informado').astype(str).str.title()
apps['nivel_espanhol'] = apps['nivel_espanhol'].fillna('Não informado').astype(str).str.title()
ing_table = apps['nivel_ingles'].value_counts().rename_axis('Nível').reset_index(name='Candidatos')
esp_table = apps['nivel_espanhol'].value_counts().rename_axis('Nível').reset_index(name='Candidatos')

col_ing, col_esp = st.columns(2)
col_ing.subheader('Distribuição de inglês declarada')
col_ing.dataframe(ing_table, hide_index=True, use_container_width=True)
col_esp.subheader('Distribuição de espanhol declarada')
col_esp.dataframe(esp_table, hide_index=True, use_container_width=True)

st.caption(
    'Use este painel como ponto de partida para identificar perfis estratégicos, carências de idiomas e clientes com maior volume de vagas. '
    'Atualize os CSVs em `datasets/` e recarregue a página para refletir os dados mais recentes.'
)

