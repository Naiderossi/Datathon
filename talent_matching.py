# app.py  Formulario padrão + mapeamento de features para o modelo
import json
import os
import streamlit as st
import pandas as pd
import re
import sys
import unicodedata
from html import escape
from ast import literal_eval
from pathlib import Path
import io
from src.mlp_infer import NUM_COLS, cosine_01

from train_mlp import pick_cv_text, pick_req_text
from src.utils import safe_list_parse
from src.preprocessing import load_applicants, load_jobs, load_prospects



ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

def resolve_artifact(rel_path: str) -> str | None:
    """Procura o arquivo em vários lugares plausíveis e retorna o caminho absoluto."""
    here = Path(__file__).resolve()
    roots = [Path.cwd()] + [parent for parent in here.parents]
    for root in roots:
        p = (root / rel_path).resolve()
        if p.exists():
            return str(p)
    return None

HAS_MLP = False
try:
    from src.mlp_infer import MLPArtifact
    HAS_MLP = True
except Exception:
    pass

# -----------------------
# Utilitrios de parsing
# -----------------------
def _norm(s: str) -> str:
    s = "" if s is None else str(s)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s

LEVELS = ['Sem conhecimento','Básico','Intermediário','Avançado','Fluente','Nativo']
ACADEMICO = ['Fundamental','Médio','Técnico','Tecnólogo','Graduação','Pós-graduação','Mestrado','Doutorado']

lvl_alias = {
"nenhum":0, "sem conhecimento":0, "none":0,
"basico":1, "bsico":1, "basic":1,
"intermediario":2, "intermedirio":2, "intermediate":2,
"avancado":3, "avanado":3, "advanced":3,
"fluente":4, "fluency":4,
"nativo":5, "native":5
}
acad_alias = {
"fundamental":0, "ensino fundamental":0,
"medio":1, "mdio":1, "ensino medio":1, "ensino mdio":1,
"tecnico":2, "tecnico":2,
"tecnologo":3, "tecnologo":3,
"superior incompleto":3,
"graduacao":4, "graduacao":4, "ensino superior":4, "ensino superior completo":4, "superior completo":4,
"pos":5, "ps":5, "pos-graduacao":5, "pos-graduacao":5, "especializacao":5, "especialização":5, "mba":5,
"mestrado":6,
"doutorado":7, "phd":7
}

def map_level(x):
    s = _norm(x)
    for k, v in lvl_alias.items():
        if k in s:
            return v
    for i, name in enumerate(LEVELS):
        if _norm(name) in s:
            return i
    return 0

def map_acad(x):
    s = _norm(x)
    for k, v in acad_alias.items():
        if k in s:
            return v
    for i, name in enumerate(ACADEMICO):
        if _norm(name) in s:
            return i
    if "ensino superior" in s:
        return 4
    return 0

def parse_req_lang(req_text, lang="ingles"):
    s = _norm(req_text)
    if lang == "ingles":
        m = re.search(r"ingles[^\w]?(basico|bsico|intermediario|intermedirio|avancado|avanado|fluente|nativo)", s)
    else:
        m = re.search(r"espanhol[^\w]?(basico|bsico|intermediario|intermedirio|avancado|avanado|fluente|nativo)", s)
    if m:
        return map_level(m.group(1))
    if (lang=="ingles" and "ingles" in s) or (lang=="espanhol" and "espanhol" in s):
        return 1
    return 0

def parse_req_acad(req_text):
    s = _norm(req_text)
    for k in ["doutorado","mestrado","pos","ps","especializacao","especialização","mba",
                "superior completo","ensino superior completo","graduacao","graduaçãoo",
                "tecnico","tcnico","medio","mdio","fundamental"]:
        if k in s:
            return map_acad(k)
    if "ensino superior" in s:
        return 4
    return 0

def parse_req_pcd(req_text):
    s = _norm(req_text)
    return int(any(w in s for w in ["pcd","pessoas com deficiencia","pessoas com deficincia","inclusivo pcd","vaga pcd"]))

def parse_req_sap(req_text, vaga_sap_field=None):
    if vaga_sap_field is not None and str(vaga_sap_field).strip().lower() in ("sim","true","1","yes"):
        return 1
    s = _norm(req_text or "")
    return int(("sap" in s) or ("4hana" in s) or ("s/4hana" in s) or ("s4hana" in s))


# Pegar credenciais do secrets apenas quando necessário (não durante importação).
# A configuração do Kaggle deve ser feita dinamicamente (ver função load_jobs).

from kaggle.api.kaggle_api_extended import KaggleApi

# ----------------------------
# Configuração Kaggle
# ----------------------------
KAGGLE_DATASET = "naiaraderossi/DatathonDataset" 
VAGAS_FILENAME = "vagas.json"

# Diretório padrão para os arquivos de dados. Se não existir, `load_jobs` o cria.
base_dir_default = Path("data")

def segmented_or_radio(label, options, index=0):
    """
    Helper para escolher entre o segmented_control (quando disponível) e radio.
    Se a versão do Streamlit não suportar 'default', faz fallback para radio horizontal.
    """
    if hasattr(st, "segmented_control"):
        try:
            return st.segmented_control(label=label, options=options, default=options[index])
        except TypeError:
            # Tentativas para versões com assinaturas diferentes
            try:
                val = st.segmented_control(label=label, options=options)
                return val if val is not None else options[index]
            except Exception:
                pass
    # Fallback: radio horizontal
    return st.radio(label, options, index=index, horizontal=True)

# ----------------------------
# Função para carregar vagas
# ----------------------------
@st.cache_data
def load_jobs(base_dir=None, uploaded_file=None):
    """
    Carrega vagas do Kaggle ou de upload do usuário.
    Se base_dir=None, usa pasta local 'data/'.
    """
    # Definir pasta local
    if base_dir is None:
        base_dir = Path("data")
    else:
        base_dir = Path(base_dir)

    base_dir.mkdir(exist_ok=True)
    path = base_dir / VAGAS_FILENAME

    # Baixar do Kaggle se não existir
    if not path.exists():
        st.info(f"Arquivo {VAGAS_FILENAME} não encontrado localmente. Baixando do Kaggle...")

        # Carregar credenciais do Kaggle de variáveis de ambiente ou st.secrets
        if not (os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY")) and hasattr(st, "secrets"):
            creds = st.secrets.get("kaggle", {})
            os.environ.setdefault("KAGGLE_USERNAME", creds.get("username", ""))
            os.environ.setdefault("KAGGLE_KEY", creds.get("key", ""))

        api = KaggleApi()
        try:
            api.authenticate()
        except Exception as exc:
            return pd.DataFrame(), {"error": f"Falha ao autenticar no Kaggle: {exc}"}

        try:
            api.dataset_download_files(
                KAGGLE_DATASET,
                path=base_dir,
                unzip=True
            )
        except Exception as exc:
            return pd.DataFrame(), {"error": f"Erro ao baixar dataset do Kaggle: {exc}"}

        if not path.exists():
            return pd.DataFrame(), {"error": f"Arquivo não encontrado após download: {path}"}

    # Usar upload do usuário se fornecido
    if uploaded_file is not None:
        vagas = json.load(io.TextIOWrapper(uploaded_file, encoding='utf-8'))
    else:
        with open(path, "r", encoding="utf-8") as f:
            vagas = json.load(f)

    # Transformar JSON em DataFrame
    rows = []
    for jid, j in vagas.items():
        ib = j.get("informacoes_basicas", {})
        pv = j.get("perfil_vaga", {})
        req_text = " ".join([
            str(ib.get("titulo_vaga","")),
            str(ib.get("descricao_vaga","")),
            str(pv.get("principais_atividades","")),
            str(pv.get("competencia_tecnicas_e_comportamentais","")),
            str(pv.get("conhecimentos_tecnicos","")),
            str(pv.get("requisitos_desejaveis",""))
        ]).strip()
        rows.append({
            "job_id": str(jid),
            "titulo": ib.get("titulo_vaga",""),
            "cliente": ib.get("cliente",""),
            "req_text_clean": req_text,
            "vaga_sap_raw": ib.get("vaga_sap","")
        })
    df = pd.DataFrame(rows)

    # Requisitos derivados do texto
    df["req_ing_ord"]  = df["req_text_clean"].apply(lambda t: int(parse_req_lang(t,"ingles") or 0))
    df["req_esp_ord"]  = df["req_text_clean"].apply(lambda t: int(parse_req_lang(t,"espanhol") or 0))
    df["req_acad_ord"] = df["req_text_clean"].apply(lambda t: int(parse_req_acad(t) or 0))
    df["job_pcd_req"]  = df["req_text_clean"].apply(lambda t: int(parse_req_pcd(t) or 0))
    df["job_sap_req"]  = df.apply(lambda r: int(parse_req_sap(r["req_text_clean"], r["vaga_sap_raw"]) or 0), axis=1)

    return df, {}
# ----------------------------
# Renderização principal
# ----------------------------
def render_app(section: str | None = None) -> None:
    st.title("Triagem e Recomendações de Talentos")

    # Carregar as vagas
    df_jobs, err = load_jobs(base_dir_default)
    if err:
        st.error(err["error"])
        st.stop()

    if df_jobs.empty:
        st.warning("Nenhuma vaga carregada.")
        st.stop()

    jobs_indexed = df_jobs.set_index("job_id")
    job_options = jobs_indexed.index.tolist()

    # Função auxiliar para formatar label da vaga
    def job_label(job_id: str) -> str:
        row = jobs_indexed.loc[job_id]
        titulo = str(row.get("titulo", "")).strip()
        cliente = str(row.get("cliente", "")).strip()
        parts = [str(job_id)]
        if titulo:
            parts.append(titulo)
        if cliente:
            parts.append(cliente)
        return " - ".join(parts)

    # Interface da seleção de vagas
    st.subheader("Vaga em análise")
    sel_job = st.selectbox(
        "Selecione a vaga",
        job_options,
        index=0,
        format_func=job_label
    )

    sel_row = jobs_indexed.loc[sel_job]
    req_text_clean = str(sel_row.req_text_clean or "")

    with st.expander("Requisitos estimados da vaga", expanded=False):
        st.markdown(
            f"- Inglês: **{LEVELS[sel_row.req_ing_ord]}**\n"
            f"- Espanhol: **{LEVELS[sel_row.req_esp_ord]}**\n"
            f"- Acadêmico: **{ACADEMICO[sel_row.req_acad_ord]}**\n"
            f"- PCD requerido: **{'Sim' if sel_row.job_pcd_req == 1 else 'Não'}**\n"
            f"- Requer SAP: **{'Sim' if sel_row.job_sap_req == 1 else 'Não'}**"
        )

    if section is None:
        tab1, tab2 = st.tabs(['Formulário e Predição', 'Sugestão de Candidatos'])
        with tab1:
            _render_form(req_text_clean, sel_row, sel_job)
        with tab2:
            _render_sourcing()
    elif section == "form":
        _render_form(req_text_clean, sel_row, sel_job)
    elif section == "sourcing":
        _render_sourcing()
    else:
        st.error(f"Seo desconhecida: {section}")

# -------------------------
# Catlogo rpido de skills
# -------------------------

DEFAULT_SKILLS = [
"SAP", "S/4HANA", "ABAP", "MM", "SD", "FI", "CO", "WM", "PP",
"Python", "SQL", "Power BI", "Excel", "InglÃªs", "Espanhol", "Scrum", "Kanban",
]

REQ_SKILL_KEYWORDS = {
"sap": "SAP",
"s/4hana": "S/4HANA",
"s4hana": "S/4HANA",
"hana": "HANA",
"abap": "ABAP",
"mm": "MM",
"sd": "SD",
"fi": "FI",
"co": "CO",
"pp": "PP",
"wm": "WM",
"python": "Python",
"sql": "SQL",
"power bi": "Power BI",
"excel": "Excel",
"tableau": "Tableau",
"scrum": "Scrum",
"kanban": "Kanban",
"agile": "Metodos ageis",
"analytics": "Analytics",
}

DEFAULT_BEHAVIORAL_TRAITS = [
"Comunicacoo", "Trabalho em equipe", "Lideranca", "Resiliencia",
"Proatividade", "Organizacao", "Pensamento critico", "Empatia",
"Adaptabilidade", "Foco em resultados", "Gestao de conflitos",
]

REQ_BEHAVIOR_KEYWORDS = {
"comunic": "Comunicacao",
"colabor": "Trabalho em equipe",
"lider": "Lideranca",
"resili": "Resiliencia",
"proativ": "Proatividade",
"organiza": "Organizacao",
"critico": "Pensamento critico",
"critico": "Pensamento critico",
"empatia": "Empatia",
"adapt": "Adaptabilidade",
"resultado": "Foco em resultados",
"conflito": "Gestao de conflitos",
"negoci": "Negociacao",
"autonomia": "Autonomia",
"multidisciplinar": "Colaboracao interdisciplinar",
"iniciativa": "Proatividade",
}

def extract_req_behaviors(req_text: str) -> list[str]:
    if not req_text:
        return []
    norm_text = _norm(req_text)
    found: list[str] = []
    for keyword, label in REQ_BEHAVIOR_KEYWORDS.items():
        if keyword in norm_text:
            found.append(label)
    result: list[str] = []
    seen: set[str] = set()
    for item in found:
        if item and item not in seen:
            result.append(item)
            seen.add(item)
    return result

def extract_req_skills(req_text: str) -> list[str]:
    """Extrai competencias mencionadas na descricao da vaga."""
    if not req_text:
        return []
    norm_text = _norm(req_text)
    padded = f" {norm_text} "
    found: list[str] = []
    for keyword, label in REQ_SKILL_KEYWORDS.items():
        if f" {keyword} " in padded:
            found.append(label)
    for skill in DEFAULT_SKILLS:
        skill_norm = _norm(skill)
        if f" {skill_norm} " in padded:
            found.append(skill)
    result: list[str] = []
    seen: set[str] = set()
    for item in found:
        if item and item not in seen:
            result.append(item)
            seen.add(item)
    return result

# -------------------------
# FormulÃ¡rio padrÃ£onizado
# -------------------------

def _render_form(req_text_clean: str, job_row, job_id: str) -> None:
    """
    Renderiza o formulário de entrevista padronizado e realiza a análise do
    candidato em relação à vaga selecionada. Todo o fluxo do formulário, desde
    os campos de entrada até a recomendação de vagas similares, está contido
    nesta função para evitar variáveis fora de escopo ou efeitos colaterais na
    importação.
    """
    st.subheader("Formulário de entrevista (padronizado)")

    levels_display = [
        "Sem conhecimento",
        "Básico",
        "Intermediário",
        "Avançado",
        "Fluente",
        "Nativo",
    ]
    academico_display = [
        "Fundamental",
        "Médio",
        "Técnico",
        "Tecnólogo",
        "Graduação",
        "Pós-graduação",
        "Mestrado",
        "Doutorado",
    ]

    def level_label(idx: int, catalog: list[str]) -> str:
        return catalog[idx] if 0 <= idx < len(catalog) else str(idx)

    # Início do formulário. Widgets devem ser declarados dentro do `with form:`.
    form = st.form("frm_interview")
    with form:
        c1, c2, c3 = st.columns(3)
        with c1:
            nome = st.text_input("Nome *")
            email = st.text_input("E-mail *")
            cidade = st.text_input("Cidade/UF")
        with c2:
            nivel_ingles = segmented_or_radio("Nível de inglês", levels_display, index=2)
            nivel_espanhol = segmented_or_radio("Nível de espanhol", levels_display, index=0)
        with c3:
            nivel_acad = segmented_or_radio("Nível acadêmico", academico_display, index=4)
            pcd_flag_ui = segmented_or_radio("PCD", ["Não", "Sim"], index=0)

        st.caption("Skills (tags - digite para filtrar)")
        skills = st.multiselect(
            "Selecionar skills",
            options=DEFAULT_SKILLS,
            default=[],
            placeholder="Ex.: SAP, MM, Python...",
        )
        skills_extra = st.text_input("Skills extras (vírgulas)")

        suggested_skills = extract_req_skills(req_text_clean)
        if suggested_skills:
            competencias_sugeridas = st.multiselect(
                "Competências sugeridas pela vaga",
                options=suggested_skills,
                default=suggested_skills,
                help="Selecione as competências relevantes para preencher o skill list do candidato.",
            )
        else:
            competencias_sugeridas = []
            st.caption("Nenhuma competência sugerida automaticamente para esta vaga.")

        st.caption("Quesitos comportamentais desejados")
        behavioral_suggestions = extract_req_behaviors(req_text_clean)
        comportamento_opcoes = (
            behavioral_suggestions if behavioral_suggestions else DEFAULT_BEHAVIORAL_TRAITS
        )
        comportamentos_escolhidos = st.multiselect(
            "Competências comportamentais",
            options=comportamento_opcoes,
            default=behavioral_suggestions,
            help="Selecione atributos comportamentais alinhados ao perfil da vaga.",
        )
        comportamentos_extra = st.text_input("Outros quesitos comportamentais (vírgulas)")

        comportamentos_registrados = set(comportamentos_escolhidos)
        if comportamentos_extra:
            comportamentos_registrados.update(
                {item.strip() for item in comportamentos_extra.split(",") if item.strip()}
            )
        st.session_state["form_comportamentos"] = sorted(comportamentos_registrados)

        st.caption("Resumo do CV (texto livre - pode colar trechos do currículo)")
        cv_text = st.text_area(
            "CV (texto livre)",
            height=160,
            placeholder="Cole aqui um resumo do currículo ou os tópicos principais",
        )

        # Botão de submissão do formulário
        submitted = form.form_submit_button("Analisar candidato!")

    # Fora do with form: tratar submissão e executar lógica de inferência
    if not submitted:
        st.stop()

    missing = []
    if not nome.strip():
        missing.append("nome")
    if not email.strip():
        missing.append("e-mail")
    if not cv_text.strip():
        missing.append("currículo (texto livre)")

    if missing:
        st.warning("Preencha os campos obrigatórios: " + ", ".join(missing))
        st.stop()

    # Verificação da disponibilidade do modelo
    if not HAS_MLP:
        st.error("Modelo MLP não disponível. Garanta que os artefatos estejam na pasta `models/`.")
        st.stop()

    try:
        artifact = tab2_get_artifact()
    except FileNotFoundError as exc:
        st.error(f"Artefato do modelo não encontrado: {exc}")
        st.stop()
    except Exception as exc:
        st.error(f"Não foi possível carregar o modelo: {exc}")
        st.stop()

    # Construir lista de skills combinada (evitar duplicatas)
    extras = [item.strip() for item in skills_extra.split(",") if item.strip()]
    combined_skills = skills + competencias_sugeridas + extras
    skill_list: list[str] = []
    for item in combined_skills:
        if item and item not in skill_list:
            skill_list.append(item)

    ing_ord = map_level(nivel_ingles)
    esp_ord = map_level(nivel_espanhol)
    acad_ord = map_acad(nivel_acad)
    pcd_flag = 1 if pcd_flag_ui.lower().startswith("s") else 0
    # Detectar SAP usando função específica (compatível com tab2)
    has_sap = detect_sap_tab2(skill_list, "", cv_text)

    # Requisitos da vaga (podem vir do DataFrame job_row ou ser extraídos do texto)
    req_ing_ord = int(job_row.get("req_ing_ord", parse_req_lang(req_text_clean, "ingles")))
    req_esp_ord = int(job_row.get("req_esp_ord", parse_req_lang(req_text_clean, "espanhol")))
    req_acad_ord = int(job_row.get("req_acad_ord", parse_req_acad(req_text_clean)))
    job_pcd_req = int(job_row.get("job_pcd_req", parse_req_pcd(req_text_clean)))
    job_sap_req = int(job_row.get(
        "job_sap_req",
        parse_req_sap(req_text_clean, job_row.get("vaga_sap_raw") or job_row.get("vaga_sap")),
    ))

    feature_row = artifact.build_feature_row(
        cv_pt_clean=cv_text,
        req_text_clean=req_text_clean,
        ing_ord=ing_ord,
        esp_ord=esp_ord,
        acad_ord=acad_ord,
        req_ing_ord=req_ing_ord,
        req_esp_ord=req_esp_ord,
        req_acad_ord=req_acad_ord,
        pcd_flag=pcd_flag,
        job_pcd_req=job_pcd_req,
        has_sap=has_sap,
        job_sap_req=job_sap_req,
        skills_list=skill_list,
    )

    try:
        probability = float(artifact.predict_proba(feature_row))
    except Exception as exc:
        st.error(f"Erro na inferência do MLP: {exc}")
        st.stop()

    score_percent = probability * 100.0
    st.success(f"Aderência estimada para a vaga {job_id}: {score_percent:.1f}%")

    # Estilização para os cartões e pills
    st.markdown(
        """
        <style>
            .match-grid {
                display: flex;
                flex-wrap: wrap;
                gap: 1rem;
                margin: 1rem 0 0 0;
            }
            .match-card {
                flex: 1 1 220px;
                border-radius: 14px;
                border: 1px solid #e8eef9;
                padding: 1rem 1.2rem;
                background: #f9fbff;
                box-shadow: 0 3px 10px rgba(15, 23, 42, 0.05);
            }
            .match-card.ok {
                border-color: #a7f3d0;
                background: #f1fdf6;
            }
            .match-card.warn {
                border-color: #fecaca;
                background: #fff6f5;
            }
            .match-card .match-label {
                text-transform: uppercase;
                font-size: 0.75rem;
                letter-spacing: 0.08em;
                color: #64748b;
                margin-bottom: 0.35rem;
            }
            .match-card .match-value {
                font-size: 1.55rem;
                font-weight: 600;
                color: #0f172a;
                margin-bottom: 0.35rem;
            }
            .match-card .match-req {
                font-size: 0.85rem;
                color: #475569;
            }
            .match-card .match-status {
                font-size: 0.9rem;
                font-weight: 600;
                margin-top: 0.4rem;
            }
            .match-card.ok .match-status {
                color: #047857;
            }
            .match-card.warn .match-status {
                color: #b91c1c;
            }
            .skill-section {
                margin-top: 1.2rem;
            }
            .skill-title {
                font-size: 0.9rem;
                font-weight: 600;
                color: #334155;
                margin-bottom: 0.4rem;
                display: inline-block;
            }
            .skill-pill-container {
                display: flex;
                flex-wrap: wrap;
                gap: 0.5rem;
            }
            .skill-pill {
                padding: 0.35rem 0.75rem;
                border-radius: 999px;
                background: #eef2ff;
                color: #3730a3;
                font-size: 0.85rem;
                font-weight: 500;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    cards_html: list[str] = []

    def add_card(title: str, current: str, requirement: str, status: str, ok: bool) -> None:
        cls = "ok" if ok else "warn"
        icon = "✅" if ok else "⚠️"
        cards_html.append(
            f"<div class='match-card {cls}'>"
            f"<div class='match-label'>{escape(title)}</div>"
            f"<div class='match-value'>{icon} {escape(current)}</div>"
            f"<div class='match-req'>Requisito: {escape(requirement)}</div>"
            f"<div class='match-status'>{escape(status)}</div>"
            "</div>"
        )

    def format_level_status(actual_ord: int, req_ord: int) -> tuple[bool, str]:
        diff = actual_ord - req_ord
        if diff > 0:
            label = "nível" if diff == 1 else "níveis"
            return True, f"Acima do requisito (+{diff} {label})"
        if diff == 0:
            return True, "Dentro do requisito"
        label = "nível" if abs(diff) == 1 else "níveis"
        return False, f"Abaixo do requisito (-{abs(diff)} {label})"

    # Gera cartões de aderência
    ok_ing, status_ing = format_level_status(ing_ord, req_ing_ord)
    ok_esp, status_esp = format_level_status(esp_ord, req_esp_ord)
    ok_acad, status_acad = format_level_status(acad_ord, req_acad_ord)

    add_card("Inglês", level_label(ing_ord, levels_display), level_label(req_ing_ord, levels_display), status_ing, ok_ing)
    add_card("Espanhol", level_label(esp_ord, levels_display), level_label(req_esp_ord, levels_display), status_esp, ok_esp)
    add_card("Formação", level_label(acad_ord, academico_display), level_label(req_acad_ord, academico_display), status_acad, ok_acad)

    # Critério PCD
    if job_pcd_req == 1:
        pcd_ok = bool(pcd_flag)
        pcd_status = "Critério atendido" if pcd_ok else "Necessita candidato PCD"
    else:
        pcd_ok = True
        pcd_status = "Não obrigatório" if not pcd_flag else "Candidato PCD (diferencial)"
    add_card("Critério PCD", "Sim" if pcd_flag else "Não", "Sim" if job_pcd_req else "Não", pcd_status, pcd_ok)

    # Experiência SAP
    if job_sap_req == 1:
        sap_ok = bool(has_sap)
        sap_status = "Experiência confirmada" if sap_ok else "Necessita experiência em SAP"
    else:
        sap_ok = True
        sap_status = "Não obrigatório" if not has_sap else "Experiência disponível"
    add_card("Experiência SAP", "Sim" if has_sap else "Não", "Sim" if job_sap_req else "Não", sap_status, sap_ok)

    st.markdown(f"<div class='match-grid'>{''.join(cards_html)}</div>", unsafe_allow_html=True)

    # Exibe as skills consideradas
    if skill_list:
        skill_pills = ''.join(f"<span class='skill-pill'>{escape(skill)}</span>" for skill in skill_list)
        st.markdown(
            f"<div class='skill-section'><span class='skill-title'>Skills considerados na análise</span><div class='skill-pill-container'>{skill_pills}</div></div>",
            unsafe_allow_html=True,
        )
    else:
        st.caption("Nenhum skill informado até o momento.")

    st.divider()
    st.subheader("Sugestão de vagas com maior aderência")

    # Carrega a base de vagas para recomendação
    try:
        _apps, jobs_base, _prospects = tab2_load_base_data()
    except Exception as exc:
        st.info(f"Não foi possível carregar a base de vagas para recomendação: {exc}")
        st.stop()

    suggestions: list[dict] = []
    job_id_str = str(job_id)
    for jid, row in jobs_base.iterrows():
        job_text = pick_req_text(row)
        if not isinstance(job_text, str) or not job_text.strip():
            continue
        req_ing = map_level(row.get("nivel_ingles_req"))
        req_esp = map_level(row.get("nivel_espanhol_req"))
        req_acad = parse_req_acad(job_text)
        job_pcd = parse_req_pcd(job_text)
        job_sap = parse_req_sap(job_text, row.get("vaga_sap"))

        row_features = artifact.build_feature_row(
            cv_pt_clean=cv_text,
            req_text_clean=job_text,
            ing_ord=ing_ord,
            esp_ord=esp_ord,
            acad_ord=acad_ord,
            req_ing_ord=req_ing,
            req_esp_ord=req_esp,
            req_acad_ord=req_acad,
            pcd_flag=pcd_flag,
            job_pcd_req=job_pcd,
            has_sap=has_sap,
            job_sap_req=job_sap,
            skills_list=skill_list,
        )
        try:
            score = float(artifact.predict_proba(row_features)) * 100.0
        except Exception:
            continue
        suggestions.append({
            "job_id": str(jid),
            "Vaga": row.get("titulo_vaga", ""),
            "Cliente": row.get("cliente", ""),
            "Score (%)": round(score, 2),
        })

    if not suggestions:
        st.info("Nenhuma outra vaga foi encontrada para recomendação.")
        st.stop()

    suggestions_df = pd.DataFrame(suggestions)
    suggestions_df = suggestions_df[suggestions_df["job_id"] != job_id_str]
    if suggestions_df.empty:
        st.info("Nenhuma outra vaga atinge score relevante neste momento.")
        st.stop()

    top_suggestions = suggestions_df.sort_values("Score (%)", ascending=False).head(5)
    st.dataframe(top_suggestions[["job_id", "Vaga", "Cliente", "Score (%)"]], width="stretch")

def _render_sourcing() -> None:
    st.subheader('Sugestão de candidatos livres')
    uploaded_jobs_file = st.file_uploader('Upload de novas vagas (CSV opcional)', type='csv', help='O arquivo deve conter ao menos job_id e campos de descrio.')
    extra_jobs_df = None
    if uploaded_jobs_file is not None:
        try:
            extra_jobs_df = pd.read_csv(uploaded_jobs_file)
            st.success(f'{len(extra_jobs_df)} linhas carregadas do CSV.')
        except Exception as exc:
            st.error(f'Não foi possível ler o CSV: {exc}')
            extra_jobs_df = None

    apps_tab2, jobs_tab2, prospect_ids_tab2 = tab2_load_base_data()
    if extra_jobs_df is not None and not extra_jobs_df.empty:
        extra_jobs = tab2_prepare_jobs(extra_jobs_df)
        if not extra_jobs.empty:
            jobs_tab2 = pd.concat([jobs_tab2.reset_index(), extra_jobs.reset_index()], ignore_index=True, sort=False)
            jobs_tab2 = jobs_tab2.drop_duplicates('job_id').set_index('job_id')

    available_candidates = apps_tab2.index.difference(prospect_ids_tab2)
    if available_candidates.empty:
        st.info('Nenhum candidato livre encontrado.')
    else:
        job_options_tab2 = jobs_tab2.index.tolist()
        job_id_tab2 = st.selectbox(
            'Selecione a vaga (Base CSV)',
            job_options_tab2,
            format_func=lambda j: f"{j}  {jobs_tab2.loc[j, 'titulo_vaga']}"
        )
        threshold_default_tab2 = json.loads(Path('models/thresholds.json').read_text(encoding='utf-8'))['mlp_thresholds']['thr_f1']
        threshold_tab2 = st.slider('Threshold (probabilidade mínima)', 0.0, 1.0, float(threshold_default_tab2), 0.01)
        max_results_tab2 = st.slider('Quantidade máxima de candidatos', 10, 200, 50, 10)

        if st.button('Buscar candidatos sugeridos', type='primary'):
            job_row = jobs_tab2.loc[job_id_tab2]
            req_text = pick_req_text(job_row)
            req_ing_ord = map_level(job_row.get('nivel_ingles_req'))
            req_esp_ord = map_level(job_row.get('nivel_espanhol_req'))
            req_acad_ord = parse_req_acad(req_text)
            job_pcd_req = parse_req_pcd(req_text)
            job_sap_req = parse_req_sap(req_text, job_row.get('vaga_sap'))

            with st.spinner('Gerando recomendações...'):
                scored = tab2_score_candidates(job_id_tab2, apps_tab2, jobs_tab2, available_candidates)

            st.markdown(f"### Vaga {job_id_tab2}  {job_row.titulo_vaga} ({job_row.cliente})")
            with st.expander('Requisitos da vaga (CSV/Base)', expanded=True):
                st.markdown(
                    f"- Inglês requerido: **{LEVELS[req_ing_ord]}**\n"
                    f"- Espanhol requerido: **{LEVELS[req_esp_ord]}**\n"
                    f"- Formação mínima: **{ACADEMICO[req_acad_ord]}**\n"
                    f"- PCD requerido: **{'Sim' if job_pcd_req == 1 else 'Não'}**\n"
                    f"- Exige SAP: **{'Sim' if job_sap_req == 1 else 'Não'}**"
                )
                st.text_area('Descrição da vaga', value=req_text or '(Sem descrição)', height=160)

            if scored.empty:
                st.info('Nenhum candidato elegível para esta vaga.')
            else:
                filtered_tab2 = scored[scored['probability'] >= threshold_tab2]
                filtered_tab2 = filtered_tab2.sort_values('probability', ascending=False).head(max_results_tab2)
                if filtered_tab2.empty:
                    st.info('Nenhum candidato atingiu o corte definido. Ajuste o threshold ou aumente o limite.')
                else:
                    rename_map = {
                        'nome': 'Candidato',
                        'nivel_ingles': 'Inglês',
                        'nível_ingles': 'Inglês',
                        'nivel_espanhol': 'Espanhol',
                        'nível_espanhol': 'Espanhol',
                        'nivel_academico': 'Formação',
                        'nível_academico': 'Formação',
                        'prob_percent': 'Score (%)',
                        'pcd_flag': 'PCD',
                        'has_sap': 'SAP',
                        'skill_overlap': 'Skills match',
                        'token_overlap_ratio': 'Token overlap'
                    }
                    table = filtered_tab2.rename(columns=rename_map)
                    display_cols = ['Candidato', 'Inglês', 'Espanhol', 'Formação', 'Score (%)', 'PCD', 'SAP', 'Skills match', 'Token overlap']
                    table = table[[col for col in display_cols if col in table.columns]].copy()
                    if 'PCD' in table.columns:
                        table['PCD'] = table['PCD'].map({0: 'Não', 1: 'Sim'})
                    if 'SAP' in table.columns:
                        table['SAP'] = table['SAP'].map({0: 'Não', 1: 'Sim'})
                    if 'Token overlap' in table.columns:
                        table['Token overlap'] = (table['Token overlap'] * 100).round(1)
                    if 'Score (%)' in table.columns:
                        table = table.round({'Score (%)': 2})
                    # Ajustar largura segundo nova API (use_container_width descontinuado)
                    st.dataframe(table, width="stretch")

                    csv = filtered_tab2.reset_index()[['candidate_id', 'nome', 'email', 'telefone', 'prob_percent'] + NUM_COLS]
                    st.download_button(
                        'Baixar CSV (tab Sugestões)',
                        csv.to_csv(index=False).encode('utf-8'),
                        file_name=f'sugestoes_tab2_vaga_{job_id_tab2}.csv',
                        mime='text/csv'
                    )

                    top5 = filtered_tab2.head(5)
                    if not top5.empty:
                        if st.button('Agendar entrevistas (Top 5)', type='secondary'):
                            agenda_info = []
                            for _, row in top5.iterrows():
                                nome = str(row.get('nome') or 'Sem nome')
                                email = str(row.get('email') or '').strip()
                                agenda_info.append(f"{nome} <{email}>" if email else nome)
                            st.success('Agendamento enviado para: ' + ', '.join(agenda_info))

                    with st.expander('Ver currículo (cv_pt) de um candidato sugerido'):
                        selection = st.selectbox(
                            'Candidato',
                            filtered_tab2.index.tolist(),
                            format_func=lambda cid: f"{cid}  {filtered_tab2.loc[cid, 'nome']}"
                        )
                        selected_cv = filtered_tab2.loc[selection].get('cv_pt') or filtered_tab2.loc[selection].get('cv_text') or '(Currículo não disponível)'
                        st.text_area('currículo', value=selected_cv, height=280)

                    st.caption('Sugestões calculadas pelo MLP reentreinado sobre a base CSV. Ajuste cortes conforme necessário.')
        else:
            st.info('Selecione a vaga e clique em **Buscar candidatos sugeridos** para ver as recomendações.')
import numpy as np


def safe_list_parse_tab2(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    text = str(value)
    if text in {"[]", ""}:
        return []
    try:
        parsed = literal_eval(text)
        if isinstance(parsed, (list, tuple, set)):
            return [str(v).strip() for v in parsed if str(v).strip()]
    except Exception:
        pass
    return [part.strip() for part in text.split(",") if part.strip()]




def detect_sap_tab2(skills, skills_text, cv_text):
    bucket = " ".join(skills) + " " + (skills_text or "") + " " + (cv_text or "")
    bucket_norm = _norm(bucket)
    return int(any(k in bucket_norm for k in {"sap", "s/4hana", "s4hana", "hana", "abap"}))


def compute_skill_overlap_tab2(skills, job_text):
    skills_norm = {_norm(s) for s in skills if _norm(s)}
    if not skills_norm:
        return 0, 0.0
    job_norm = _norm(job_text)
    hits = sum(1 for s in skills_norm if s and s in job_norm)
    ratio = hits / max(1, len(skills_norm))
    return hits, ratio


def compute_token_overlap_metrics_tab2(cv_text, req_text):
    cv_tokens = {tok for tok in _norm(cv_text).split() if tok}
    req_tokens = {tok for tok in _norm(req_text).split() if tok}
    if not req_tokens:
        return 0, 0.0
    overlap = cv_tokens.intersection(req_tokens)
    ratio = len(overlap) / max(1, len(req_tokens))
    return len(overlap), ratio


def forward_batch_tab2(weights, X):
    z0 = X @ weights.w0 + weights.b0
    a0 = np.maximum(z0, 0.0)
    bn0 = weights.gamma0 * (a0 - weights.mean0) / np.sqrt(weights.var0 + weights.eps0) + weights.beta0
    z1 = bn0 @ weights.w1 + weights.b1
    a1 = np.maximum(z1, 0.0)
    bn1 = weights.gamma1 * (a1 - weights.mean1) / np.sqrt(weights.var1 + weights.eps1) + weights.beta1
    z2 = bn1 @ weights.w_out + weights.b_out
    return 1.0 / (1.0 + np.exp(-np.clip(z2, -60.0, 60.0)))

# Configurar credenciais a partir dos secrets do Streamlit
os.environ["KAGGLE_USERNAME"] = st.secrets["kaggle"]["username"]
os.environ["KAGGLE_KEY"] = st.secrets["kaggle"]["key"]

from kaggle.api.kaggle_api_extended import KaggleApi

@st.cache_resource(show_spinner=False)
def tab2_get_artifact():
    # Diretório local para armazenar modelos
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    # Caminhos dos arquivos
    pipeline_path = models_dir / "data_pipeline.joblib"
    model_path = models_dir / "model_mlp_lsa.h5"

    # Se não existir, baixa do Kaggle
    if not pipeline_path.exists() or not model_path.exists():
        with st.spinner("⬇️ Baixando pipeline e modelo do Kaggle..."):
            api = KaggleApi()
            api.authenticate()

            # Download do dataset privado (descompacta arquivos na pasta models/)
            api.dataset_download_files(
                "naiaraderossi/DatathonDataset",  # substitua pelo seu dataset
                path=str(models_dir),
                unzip=True
            )

    return MLPArtifact(str(pipeline_path), str(model_path))

@st.cache_data(show_spinner=False)
def tab2_load_base_data():
    #Carregar candidatos
    apps_raw = load_applicants()
    # Quando load_applicants devolve (df, err), extraia somente o df
    apps_df = apps_raw[0] if isinstance(apps_raw, tuple) else apps_raw
    # Se o índice é candidate_id, trazemos para coluna para evitar perda na limpeza
    if getattr(apps_df.index, "name", None) == "candidate_id" and "candidate_id" not in apps_df.columns:
        apps_df = apps_df.reset_index()
    apps = apps_df.copy()

    #Carregar vagas
    jobs_raw = load_jobs()
    jobs_df = jobs_raw[0] if isinstance(jobs_raw, tuple) else jobs_raw
    if getattr(jobs_df.index, "name", None) == "job_id" and "job_id" not in jobs_df.columns:
        jobs_df = jobs_df.reset_index()
    jobs = jobs_df.copy()

    prospects = load_prospects()[["candidate_id"]]

    #Preparar candidatos
    if "candidate_id" in apps.columns:
        apps = apps.drop_duplicates("candidate_id").set_index("candidate_id")
    else:
        # Se candidate_id não está disponível, apenas deduplicar pelo índice atual
        apps = apps.drop_duplicates().copy()
    for col in ["skills_text", "cv_pt", "cv_pt_clean", "cv_pt_clean_noaccents", "cv_en"]:
        if col in apps.columns:
            apps[col] = apps[col].fillna("")
    apps["skills_list"] = apps["skills_list"].apply(safe_list_parse_tab2)
    apps["cv_len_tokens"] = apps.get("cv_len_tokens", 0).fillna(0)
    #Preparar vagas
    if "job_id" in jobs.columns:
        jobs = jobs.drop_duplicates("job_id").set_index("job_id")
    else:
        jobs = jobs.drop_duplicates().copy()
    for col in ["req_text_clean", "req_text_clean_noaccents", "req_text"]:
        if col in jobs.columns:
            jobs[col] = jobs[col].fillna("")
    jobs["req_len_tokens"] = jobs.get("req_len_tokens", 0).fillna(0)

    prospect_ids = set(prospects["candidate_id"])
    return apps, jobs, prospect_ids

def tab2_prepare_jobs(df):
    df = df.copy()
    if 'job_id' not in df.columns:
        return pd.DataFrame()
    df = df.dropna(subset=['job_id'])
    df['job_id'] = pd.to_numeric(df['job_id'], errors='coerce')
    df = df.dropna(subset=['job_id'])
    df['job_id'] = df['job_id'].astype(int)
    for col in ['req_text_clean', 'req_text_clean_noaccents', 'req_text']:
        if col in df.columns:
            df[col] = df[col].fillna('')
    df['req_len_tokens'] = df.get('req_len_tokens', 0).fillna(0)
    df = df.drop_duplicates('job_id').set_index('job_id')
    return df


def tab2_prepare_candidates(job_row, candidates):
    job_text = pick_req_text(job_row) or ''
    req_ing_ord = map_level(job_row.get('nivel_ingles_req'))
    req_esp_ord = map_level(job_row.get('nivel_espanhol_req'))
    req_acad_ord = parse_req_acad(job_text)
    job_pcd_req = parse_req_pcd(job_text)
    job_sap_req = parse_req_sap(job_text, job_row.get('vaga_sap'))
    req_len_tokens = int(job_row.get('req_len_tokens') or len(_norm(job_text).split()))

    records = []
    info_rows = []
    for cid, cand in candidates.iterrows():
        cv_text = pick_cv_text(cand)
        if not cv_text.strip():
            continue
        skills_list = safe_list_parse_tab2(cand.get('skills_list'))
        skills_text = cand.get('skills_text', '')

        ing_ord = map_level(cand.get('nivel_ingles'))
        esp_ord = map_level(cand.get('nivel_espanhol'))
        acad_ord = map_acad(cand.get('nivel_academico'))

        meets_ing = int(ing_ord >= req_ing_ord)
        meets_esp = int(esp_ord >= req_esp_ord)
        meets_acad = int(acad_ord >= req_acad_ord)

        diff_ing = int(np.clip(ing_ord - req_ing_ord, -3, 3))
        diff_esp = int(np.clip(esp_ord - req_esp_ord, -3, 3))
        diff_acad = int(np.clip(acad_ord - req_acad_ord, -4, 4))

        pcd_flag = int(str(cand.get('pcd', '')).strip().lower() in {'sim', 'true', '1'})
        has_sap = detect_sap_tab2(skills_list, skills_text, cv_text)

        skill_overlap, skill_overlap_ratio = compute_skill_overlap_tab2(skills_list, job_text)
        token_overlap_count, token_overlap_ratio = compute_token_overlap_metrics_tab2(cv_text, job_text)

        pcd_match = int(1 if job_pcd_req != 1 else (pcd_flag == 1))
        sap_match = int(1 if job_sap_req != 1 else (has_sap == 1))

        text_score = cosine_01(cv_text, job_text)
        score_textual = float(text_score * 100.0)

        cv_len_tokens = int(cand.get('cv_len_tokens') or len(_norm(cv_text).split()))
        len_ratio = cv_len_tokens / (req_len_tokens + 1)

        records.append({
            'candidate_id': cid,
            'cv_text': cv_text,
            'req_text': job_text,
            'ing_ord': ing_ord,
            'esp_ord': esp_ord,
            'acad_ord': acad_ord,
            'req_ing_ord': req_ing_ord,
            'req_esp_ord': req_esp_ord,
            'req_acad_ord': req_acad_ord,
            'meets_ing': meets_ing,
            'meets_esp': meets_esp,
            'meets_acad': meets_acad,
            'diff_ing': diff_ing,
            'diff_esp': diff_esp,
            'diff_acad': diff_acad,
            'pcd_match': pcd_match,
            'sap_match': sap_match,
            'stage_score': 0,
            'text_score01': float(text_score),
            'score_textual': score_textual,
            'pcd_flag': pcd_flag,
            'job_pcd_req': job_pcd_req,
            'has_sap': has_sap,
            'job_sap_req': job_sap_req,
            'cv_len_tokens': float(cv_len_tokens),
            'req_len_tokens': float(req_len_tokens),
            'len_ratio': float(len_ratio),
            'skill_overlap': float(skill_overlap),
            'skill_overlap_ratio': float(skill_overlap_ratio),
            'token_overlap_ratio': float(token_overlap_ratio),
            'token_overlap_count': float(token_overlap_count),
        })
        info = {
            'nome': cand.get('nome', ''),
            'email': cand.get('email', ''),
            'telefone': cand.get('telefone', ''),
            'nivel_ingles': cand.get('nivel_ingles', ''),
            'nivel_espanhol': cand.get('nivel_espanhol', ''),
            'nivel_academico': cand.get('nivel_academico', ''),
            'skills_list': skills_list,
            'cv_pt': cand.get('cv_pt', '') or cand.get('cv_pt_clean', '')
        }
        info_rows.append((cid, info))

    if not records:
        return pd.DataFrame()

    features = pd.DataFrame.from_records(records).set_index('candidate_id')
    info_df = pd.DataFrame(dict(info_rows)).T
    info_df.index.name = 'candidate_id'
    merged = features.join(info_df, how='left')
    return merged


def tab2_score_candidates(job_id, apps, jobs, candidate_pool):
    artifact = tab2_get_artifact()
    if job_id not in jobs.index:
        return pd.DataFrame()
    job_row = jobs.loc[job_id]
    candidates = apps.loc[candidate_pool]
    features = tab2_prepare_candidates(job_row, candidates)
    if features.empty:
        return pd.DataFrame()

    tfidf_cv = artifact.tfidf_cv.transform(features['cv_text'].tolist())
    svd_cv = artifact.svd_cv.transform(tfidf_cv)
    tfidf_job = artifact.tfidf_job.transform(features['req_text'].tolist())
    svd_job = artifact.svd_job.transform(tfidf_job)
    numeric = artifact.scaler.transform(features[NUM_COLS].astype(float).values)
    X = np.hstack([svd_cv, svd_job, numeric])
    probs = forward_batch_tab2(artifact.weights, X).ravel()

    result = features.copy()
    result['probability'] = probs
    result['prob_percent'] = (result['probability'] * 100).round(2)
    return result

if __name__ == '__main__':
    render_app()









