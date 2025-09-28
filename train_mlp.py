#!/usr/bin/env python
"""Training script for MLP + LSA candidate-job matching model."""
from __future__ import annotations

import json
import math
import re
import unicodedata
from ast import literal_eval
from pathlib import Path
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix,
                             f1_score, precision_recall_curve)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

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

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

TARGET_STATUSES = {
    "encaminhado ao requisitante",
    "nao aprovado pelo cliente",
    "contratado pela decision",
    "contratado como hunting",
    "aprovado",
}
NEGATIVE_MULTIPLIER = 1

HASH_VECTOR = HashingVectorizer(
    n_features=2**18,
    alternate_sign=False,
    norm="l2",
    ngram_range=(1, 2),
    lowercase=True,
)

NUMERIC_FEATURES = [
    "ing_ord",
    "esp_ord",
    "acad_ord",
    "req_ing_ord",
    "req_esp_ord",
    "req_acad_ord",
    "meets_ing",
    "meets_esp",
    "meets_acad",
    "diff_ing",
    "diff_esp",
    "diff_acad",
    "pcd_match",
    "sap_match",
    "stage_score",
    "text_score01",
    "score_textual",
    "pcd_flag",
    "job_pcd_req",
    "has_sap",
    "job_sap_req",
    "cv_len_tokens",
    "req_len_tokens",
    "len_ratio",
    "skill_overlap",
    "skill_overlap_ratio",
    "token_overlap_ratio",
    "token_overlap_count",
]

LEVELS = [
    "Sem conhecimento",
    "Básico",
    "Intermediário",
    "Avançado",
    "Fluente",
    "Nativo",
]
ACADEMICO = [
    "Fundamental",
    "Médio",
    "Técnico",
    "Tecnólogo",
    "Graduação",
    "Pós-graduação",
    "Mestrado",
    "Doutorado",
]

lvl_alias = {
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

acad_alias = {
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

def _norm(text: str | float | None) -> str:
    if text is None or (isinstance(text, float) and math.isnan(text)):
        return ""
    s = str(text)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s

def map_level(x: str | float | None) -> int:
    s = _norm(x)
    for k, v in lvl_alias.items():
        if k in s:
            return v
    for i, name in enumerate(LEVELS):
        if _norm(name) in s:
            return i
    if "fluente" in s:
        return 4
    return 0

def map_acad(x: str | float | None) -> int:
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
    for k in [
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
        if k in s:
            return map_acad(k)
    if "ensino superior" in s:
        return 4
    return 0

def parse_req_pcd(req_text: str) -> int:
    s = _norm(req_text)
    keywords = ["pcd", "pessoas com deficiencia", "pessoas com deficiência", "vaga pcd"]
    return int(any(k in s for k in keywords))

def parse_req_sap(req_text: str, vaga_sap_field: str | float | None = None) -> int:
    if vaga_sap_field is not None and str(vaga_sap_field).strip().lower() in {"sim", "true", "1", "yes"}:
        return 1
    s = _norm(req_text)
    keywords = ["sap", "s/4hana", "s4hana", "4hana"]
    return int(any(k in s for k in keywords))

def cosine_hash(text_a: pd.Series, text_b: pd.Series) -> np.ndarray:
    xa = HASH_VECTOR.transform(text_a.tolist())
    xb = HASH_VECTOR.transform(text_b.tolist())
    sim = xa.multiply(xb).sum(axis=1)
    return np.asarray(sim).ravel()

def safe_list_parse(value: str | float | None) -> list[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    if value == "[]":
        return []
    if isinstance(value, list):
        return value
    try:
        parsed = literal_eval(str(value))
        if isinstance(parsed, (list, tuple, set)):
            return [str(x) for x in parsed]
    except Exception:
        pass
    return [part.strip() for part in str(value).split(",") if part.strip()]

SAP_KEYWORDS = {"sap", "s/4hana", "s4hana", "hana", "abap"}

def compute_token_overlap_metrics(cv_text: str, req_text: str) -> tuple[int, float]:
    cv_tokens = set(tok for tok in _norm(cv_text).split() if tok)
    job_tokens = set(tok for tok in _norm(req_text).split() if tok)
    if not job_tokens:
        return 0, 0.0
    overlap = cv_tokens.intersection(job_tokens)
    ratio = len(overlap) / max(1, len(job_tokens))
    return len(overlap), ratio


def detect_sap(skills: list[str], skills_text: str, cv_text: str) -> int:
    bucket = " ".join(safe for safe in skills) + " " + (skills_text or "") + " " + (cv_text or "")
    bucket_norm = _norm(bucket)
    return int(any(k in bucket_norm for k in SAP_KEYWORDS))

def compute_skill_overlap(skills: list[str], job_text: str) -> tuple[int, float]:
    skills_norm = {_norm(s) for s in skills if _norm(s)}
    if not skills_norm:
        return 0, 0.0
    job_norm = _norm(job_text)
    hits = sum(1 for s in skills_norm if s in job_norm)
    ratio = hits / max(1, len(skills_norm))
    return hits, ratio

def pick_cv_text(row: pd.Series) -> str:
    for col in ["cv_pt_clean_noaccents", "cv_pt_clean", "cv_pt", "cv_en", "cv_pt_clean_noaccents" ]:
        if col in row and isinstance(row[col], str) and row[col].strip():
            return row[col]
    return ""

def pick_req_text(row: pd.Series) -> str:
    for col in ["req_text_clean_noaccents", "req_text_clean", "req_text"]:
        if col in row and isinstance(row[col], str) and row[col].strip():
            return row[col]
    parts = [row.get("principais_atividades", ""), row.get("competencias", "")]
    return " ".join(p for p in parts if isinstance(p, str))

def load_and_prepare() -> pd.DataFrame:
    apps = pd.read_csvt(PATH_APPLICANTS)
    jobs = pd.read_csv(PATH_JOBS)
    prospects = pd.read_csv(PATH_PROSPECTS)

    df = prospects.merge(apps, on="candidate_id", how="left")
    df = df.merge(jobs, on="job_id", how="left", suffixes=("", "_job"))

    df["cv_text"] = df.apply(pick_cv_text, axis=1)
    df["req_text"] = df.apply(pick_req_text, axis=1)

    df = df[df["cv_text"].str.strip().astype(bool)]
    df = df[df["req_text"].str.strip().astype(bool)]

    df["cv_text"] = df["cv_text"].fillna("")
    df["req_text"] = df["req_text"].fillna("")

    df["ing_ord"] = df["nivel_ingles"].apply(map_level)
    df["esp_ord"] = df["nivel_espanhol"].apply(map_level)
    df["acad_ord"] = df["nivel_academico"].apply(map_acad)
    df["req_ing_ord"] = df["nivel_ingles_req"].apply(map_level)
    df["req_esp_ord"] = df["nivel_espanhol_req"].apply(map_level)
    df["req_acad_ord"] = df["req_text"].apply(parse_req_acad)

    df["meets_ing"] = (df["ing_ord"] >= df["req_ing_ord"]).astype(int)
    df["meets_esp"] = (df["esp_ord"] >= df["req_esp_ord"]).astype(int)
    df["meets_acad"] = (df["acad_ord"] >= df["req_acad_ord"]).astype(int)

    df["diff_ing"] = np.clip(df["ing_ord"] - df["req_ing_ord"], -3, 3)
    df["diff_esp"] = np.clip(df["esp_ord"] - df["req_esp_ord"], -3, 3)
    df["diff_acad"] = np.clip(df["acad_ord"] - df["req_acad_ord"], -4, 4)

    df["pcd_flag"] = df["pcd"].fillna("").astype(str).str.lower().str.contains("sim").astype(int)
    df["job_pcd_req"] = df["req_text"].apply(parse_req_pcd)

    df["job_sap_req"] = df.apply(lambda row: parse_req_sap(row["req_text"], row.get("vaga_sap")), axis=1)

    df["skills_list"] = df["skills_list"].apply(safe_list_parse)
    df["skills_text"] = df["skills_text"].fillna("")

    df["has_sap"] = df.apply(
        lambda row: detect_sap(row["skills_list"], row["skills_text"], row["cv_text"]), axis=1
    )

    skill_hits = df.apply(lambda row: compute_skill_overlap(row["skills_list"], row["req_text"]), axis=1)
    df["skill_overlap"] = [hit for hit, _ in skill_hits]
    df["skill_overlap_ratio"] = [ratio for _, ratio in skill_hits]

    token_hits = df.apply(lambda row: compute_token_overlap_metrics(row["cv_text"], row["req_text"]), axis=1)
    df["token_overlap_count"] = [count for count, _ in token_hits]
    df["token_overlap_ratio"] = [ratio for _, ratio in token_hits]

    df["sap_match"] = df.apply(
        lambda row: int(1 if row["job_sap_req"] != 1 else (row["has_sap"] == 1)), axis=1
    )
    df["pcd_match"] = df.apply(
        lambda row: int(1 if row["job_pcd_req"] != 1 else (row["pcd_flag"] == 1)), axis=1
    )

    df["text_score01"] = cosine_hash(df["cv_text"], df["req_text"])
    df["score_textual"] = df["text_score01"] * 100.0

    df["cv_len_tokens"] = df.get("cv_len_tokens").fillna(df["cv_text"].str.split().str.len())
    df["req_len_tokens"] = df.get("req_len_tokens").fillna(df["req_text"].str.split().str.len())
    df["len_ratio"] = df["cv_len_tokens"] / (df["req_len_tokens"] + 1)

    df["stage_score"] = 0

    mask = df["situacao_norm"].isin(TARGET_STATUSES)
    if mask.any():
        df = df[mask].copy()

    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=NUMERIC_FEATURES + ["cv_text", "req_text"])
    df = df[NUMERIC_FEATURES + ['cv_text', 'req_text', 'status_bin', 'job_id', 'candidate_id', 'situacao_norm']]

    pos_df = df[df['status_bin'] == 1]
    neg_df = df[df['status_bin'] == 0]
    if not pos_df.empty and not neg_df.empty:
        max_neg = min(len(neg_df), NEGATIVE_MULTIPLIER * len(pos_df))
        neg_sample = neg_df.sample(n=max_neg, random_state=SEED)
        df = pd.concat([pos_df, neg_sample], ignore_index=True)
        df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    return df

def train_model(df: pd.DataFrame) -> dict:
    y = df["status_bin"].astype(int).values
    X_numeric = df[NUMERIC_FEATURES].astype(float).values

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        df.index.values, y, test_size=0.2, random_state=SEED, stratify=y
    )

    train_df = df.loc[X_train_full].copy()
    test_df = df.loc[X_test].copy()

    tfidf_cv = TfidfVectorizer(max_features=40000, ngram_range=(1, 2), min_df=2)
    tfidf_job = TfidfVectorizer(max_features=30000, ngram_range=(1, 2), min_df=2)

    svd_cv = TruncatedSVD(n_components=300, random_state=SEED)
    svd_job = TruncatedSVD(n_components=200, random_state=SEED)

    tfidf_cv_train = tfidf_cv.fit_transform(train_df["cv_text"].tolist())
    tfidf_job_train = tfidf_job.fit_transform(train_df["req_text"].tolist())

    svd_cv_train = svd_cv.fit_transform(tfidf_cv_train)
    svd_job_train = svd_job.fit_transform(tfidf_job_train)

    scaler = StandardScaler()
    numeric_train = scaler.fit_transform(train_df[NUMERIC_FEATURES].values.astype(float))

    X_train = np.hstack([svd_cv_train, svd_job_train, numeric_train]).astype(np.float32)
    y_train = y_train_full.astype(np.float32)

    tfidf_cv_test = tfidf_cv.transform(test_df["cv_text"].tolist())
    tfidf_job_test = tfidf_job.transform(test_df["req_text"].tolist())
    svd_cv_test = svd_cv.transform(tfidf_cv_test)
    svd_job_test = svd_job.transform(tfidf_job_test)
    numeric_test = scaler.transform(test_df[NUMERIC_FEATURES].values.astype(float))
    X_test_matrix = np.hstack([svd_cv_test, svd_job_test, numeric_test]).astype(np.float32)
    y_test_array = y_test.astype(np.float32)

    class_weight = None

    tf.keras.backend.clear_session()
    input_dim = X_train.shape[1]
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_dim,), name="input_layer"),
            tf.keras.layers.Dense(512, activation="relu", name="dense"),
            tf.keras.layers.BatchNormalization(name="batch_normalization"),
            tf.keras.layers.Dropout(0.5, name="dropout"),
            tf.keras.layers.Dense(256, activation="relu", name="dense_1"),
            tf.keras.layers.BatchNormalization(name="batch_normalization_1"),
            tf.keras.layers.Dropout(0.4, name="dropout_1"),
            tf.keras.layers.Dense(1, activation="sigmoid", name="dense_2"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(name="auc"), tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall")],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc",
            patience=5,
            restore_best_weights=True,
            mode="max",
        )
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_split=0.1,
        epochs=40,
        batch_size=256,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=2,
    )

    y_proba_test = model.predict(X_test_matrix, batch_size=512).ravel()

    precision, recall, thresholds = precision_recall_curve(y_test_array, y_proba_test)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_threshold = float(thresholds[-1] if best_idx >= len(thresholds) else thresholds[best_idx])
    best_f1 = float(f1_scores[best_idx])

    f1_at_half = float(f1_score(y_test_array, (y_proba_test >= 0.5).astype(int)))

    report = classification_report(y_test_array, (y_proba_test >= best_threshold).astype(int), output_dict=True)
    cm = confusion_matrix(y_test_array, (y_proba_test >= best_threshold).astype(int)).tolist()

    pipeline = {
        "tfidf_cv": tfidf_cv,
        "svd_cv": svd_cv,
        "tfidf_job": tfidf_job,
        "svd_job": svd_job,
        "scaler": scaler,
        "numeric_features": NUMERIC_FEATURES,
    }

    # Feature importance on numeric fields using logistic regression
    scaler_num = StandardScaler()
    X_num_train = scaler_num.fit_transform(train_df[NUMERIC_FEATURES].values)
    X_num_test = scaler_num.transform(test_df[NUMERIC_FEATURES].values)
    log_reg = LogisticRegression(max_iter=1000, class_weight="balanced")
    log_reg.fit(X_num_train, train_df['status_bin'].values)
    coef_importance = {
        feature: float(weight)
        for feature, weight in sorted(
            zip(NUMERIC_FEATURES, log_reg.coef_[0]), key=lambda item: abs(item[1]), reverse=True
        )
    }

    artefacts = {
        "model": model,
        "pipeline": pipeline,
        "history": history.history,
        "metrics": {
            "best_threshold": best_threshold,
            "best_f1": best_f1,
            "f1_at_0.50": f1_at_half,
            "classification_report": report,
            "confusion_matrix": cm,
            "class_weight": class_weight,
        },
        "feature_importance_numeric": coef_importance,
        "test_predictions": y_proba_test.tolist(),
        "test_index": test_df[["job_id", "candidate_id", "situacao_norm"]].to_dict(orient="records"),
    }
    return artefacts

import joblib
try:
    import streamlit as st
    USE_STREAMLIT_SECRETS = True
except ImportError:
    USE_STREAMLIT_SECRETS = False

def download_pipeline_from_kaggle() -> Path:
    api.dataset_download_files(
        "naiaraderossi/DatathonDataset", 
        path=data_dir,
        unzip=True
    )
    return data_dir / "data_pipeline.joblib"

def main() -> None:
    df = load_and_prepare()
    artefacts = train_model(df)

    # ---------------------------
    # Pastas e caminhos
    # ---------------------------
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    pipeline_path = models_dir / "data_pipeline.joblib"
    model_path = models_dir / "model_mlp_lsa.h5"
    thresholds_path = models_dir / "thresholds.json"
    registry_path = models_dir / "registry.json"
    report_path = models_dir / "training_report.json"

    # ---------------------------
    # Salvar pipeline
    # ---------------------------
    if not pipeline_path.exists():
        # Se estiver em ambiente remoto, baixar do Kaggle
        if USE_STREAMLIT_SECRETS:
            print("Pipeline não encontrado. Baixando do Kaggle...")
            pipeline_path = download_pipeline_from_kaggle()

    joblib.dump(artefacts["pipeline"], pipeline_path)

    # ---------------------------
    # Salvar modelo
    # ---------------------------
    artefacts["model"].save(model_path, include_optimizer=False)

    # ---------------------------
    # Salvar thresholds
    # ---------------------------
    best_thr = artefacts["metrics"]["best_threshold"]
    thr_payload = {
        "mlp_thresholds": {
            "thr_f1": best_thr,
            "thr_p60": best_thr,
        }
    }
    thresholds_path.write_text(json.dumps(thr_payload, indent=2), encoding="utf-8")

    # ---------------------------
    # Salvar registry
    # ---------------------------
    registry_payload = {
        "model_type": "mlp_lsa",
        "pipeline": str(pipeline_path),
        "model": str(model_path),
        "thresholds": str(thresholds_path),
        "metrics": {
            "best_f1": artefacts["metrics"]["best_f1"],
            "f1_at_0.50": artefacts["metrics"]["f1_at_0.50"],
        },
    }
    registry_path.write_text(json.dumps(registry_payload, indent=2), encoding="utf-8")

    # ---------------------------
    # Salvar relatório de treino
    # ---------------------------
    artefacts_serializable = artefacts.copy()
    artefacts_serializable.pop("model", None)
    artefacts_serializable.pop("pipeline", None)
    report_path.write_text(json.dumps(artefacts_serializable, indent=2), encoding="utf-8")

    # ---------------------------
    # Mensagens
    # ---------------------------
    print("Artifacts saved:")
    print("-", pipeline_path)
    print("-", model_path)
    print("-", thresholds_path)
    print("-", registry_path)
    print("Best F1: %.4f (threshold=%.4f)" % (artefacts["metrics"]["best_f1"], artefacts["metrics"]["best_threshold"]))


if __name__ == "__main__":
    main()










