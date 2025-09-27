"""Training script for the MLP + LSA model."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .feature_engineering import NUMERIC_FEATURES, prepare_candidates_for_job
from .model_utils import NUMERIC_FEATURES as MODEL_NUMERIC_FEATURES
from .utils import (
    LEVELS,
    compute_skill_overlap,
    compute_token_overlap_metrics,
    cosine_similarity_01,
    detect_sap,
    map_acad,
    map_level,
    parse_req_acad,
    parse_req_pcd,
    parse_req_sap,
    pick_cv_text,
    pick_req_text,
    safe_list_parse,
)
from . import preprocessing

SEED = 42


def _prepare_training_dataframe() -> pd.DataFrame:
    apps, jobs, _ = preprocessing.load_all()
    prospects = preprocessing.load_prospects()

    # Keep relevant statuses
    keep_status = {
        'encaminhado ao requisitante',
        'nao aprovado pelo cliente',
        'contratado pela decision',
        'contratado como hunting',
        'aprovado',
    }
    filtered = prospects[prospects['situacao_norm'].isin(keep_status)].copy()

    merged = filtered.merge(apps.reset_index(), on='candidate_id', how='left', suffixes=('', '_cand'))
    merged = merged.merge(jobs.reset_index(), on='job_id', how='left', suffixes=('', '_job'))
    merged = merged.dropna(subset=['cv_pt', 'req_text'], how='all')

    records = []
    for _, row in merged.iterrows():
        job_row = row[[
            'job_id', 'nivel_ingles_req', 'nivel_espanhol_req', 'vaga_sap',
            'req_text_clean', 'req_text_clean_noaccents', 'req_text', 'req_len_tokens'
        ]]
        cand_row = row[[
            'candidate_id', 'pcd', 'nivel_ingles', 'nivel_espanhol', 'nivel_academico',
            'skills_text', 'skills_list', 'cv_pt', 'cv_pt_clean', 'cv_pt_clean_noaccents', 'cv_en', 'cv_len_tokens'
        ]]
        job_text = pick_req_text(job_row)
        features = build_feature_row_training(job_row, cand_row, job_text)
        features['status_bin'] = int(row.get('status_bin', 0))
        features['job_id'] = row['job_id']
        features['candidate_id'] = row['candidate_id']
        features['situacao_norm'] = row['situacao_norm']
        records.append(features)

    df = pd.DataFrame(records)
    pos_df = df[df['status_bin'] == 1]
    neg_df = df[df['status_bin'] == 0]
    if pos_df.empty or neg_df.empty:
        return df
    neg_sample = neg_df.sample(n=len(pos_df), random_state=SEED)
    balanced = pd.concat([pos_df, neg_sample], ignore_index=True)
    balanced = balanced.sample(frac=1, random_state=SEED).reset_index(drop=True)
    return balanced


def build_feature_row_training(job_row: pd.Series, cand_row: pd.Series, job_text: str) -> dict:
    req_ing_ord = map_level(job_row.get('nivel_ingles_req'))
    req_esp_ord = map_level(job_row.get('nivel_espanhol_req'))
    req_acad_ord = parse_req_acad(job_text)
    job_pcd_req = parse_req_pcd(job_text)
    job_sap_req = parse_req_sap(job_text, job_row.get('vaga_sap'))
    req_len_tokens = int(job_row.get('req_len_tokens') or len(job_text.split()))

    cv_text = pick_cv_text(cand_row)
    skills_list = safe_list_parse(cand_row.get('skills_list'))
    skills_text = cand_row.get('skills_text', '')

    ing_ord = map_level(cand_row.get('nivel_ingles'))
    esp_ord = map_level(cand_row.get('nivel_espanhol'))
    acad_ord = map_acad(cand_row.get('nivel_academico'))

    meets_ing = int(ing_ord >= req_ing_ord)
    meets_esp = int(esp_ord >= req_esp_ord)
    meets_acad = int(acad_ord >= req_acad_ord)

    diff_ing = int(np.clip(ing_ord - req_ing_ord, -3, 3))
    diff_esp = int(np.clip(esp_ord - req_esp_ord, -3, 3))
    diff_acad = int(np.clip(acad_ord - req_acad_ord, -4, 4))

    pcd_flag = int(str(cand_row.get('pcd', '')).strip().lower() in {'sim', 'true', '1'})
    has_sap = detect_sap(skills_list, skills_text, cv_text)

    skill_overlap, skill_overlap_ratio = compute_skill_overlap(skills_list, job_text)
    token_overlap_count, token_overlap_ratio = compute_token_overlap_metrics(cv_text, job_text)

    pcd_match = int(1 if job_pcd_req != 1 else (pcd_flag == 1))
    sap_match = int(1 if job_sap_req != 1 else (has_sap == 1))

    text_score = cosine_similarity_01(cv_text, job_text)
    score_textual = float(text_score * 100.0)

    cv_len_tokens = int(cand_row.get('cv_len_tokens') or len(cv_text.split()))
    len_ratio = cv_len_tokens / (req_len_tokens + 1)

    return {
        'cv_pt_clean': cv_text,
        'req_text_clean': job_text,
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
    }


def train_model(output_dir: Path = Path('models')) -> Tuple[float, float]:
    output_dir.mkdir(exist_ok=True)
    df = _prepare_training_dataframe()
    if df.empty:
        raise RuntimeError('Treinamento: dataset vazio após filtragem.')

    y = df['status_bin'].astype(int).values
    X_train_idx, X_test_idx, y_train, y_test = train_test_split(df.index.values, y, test_size=0.2, stratify=y, random_state=SEED)
    train_df = df.loc[X_train_idx]
    test_df = df.loc[X_test_idx]

    tfidf_cv = TfidfVectorizer(max_features=40000, ngram_range=(1, 2), min_df=2)
    tfidf_job = TfidfVectorizer(max_features=30000, ngram_range=(1, 2), min_df=2)

    svd_cv = TruncatedSVD(n_components=300, random_state=SEED)
    svd_job = TruncatedSVD(n_components=200, random_state=SEED)

    tfidf_cv_train = tfidf_cv.fit_transform(train_df['cv_pt_clean'])
    tfidf_job_train = tfidf_job.fit_transform(train_df['req_text_clean'])
    svd_cv_train = svd_cv.fit_transform(tfidf_cv_train)
    svd_job_train = svd_job.fit_transform(tfidf_job_train)

    scaler = StandardScaler()
    numeric_train = scaler.fit_transform(train_df[NUMERIC_FEATURES].values.astype(float))
    X_train = np.hstack([svd_cv_train, svd_job_train, numeric_train]).astype(np.float32)

    tfidf_cv_test = tfidf_cv.transform(test_df['cv_pt_clean'])
    tfidf_job_test = tfidf_job.transform(test_df['req_text_clean'])
    svd_cv_test = svd_cv.transform(tfidf_cv_test)
    svd_job_test = svd_job.transform(tfidf_job_test)
    numeric_test = scaler.transform(test_df[NUMERIC_FEATURES].values.astype(float))
    X_test = np.hstack([svd_cv_test, svd_job_test, numeric_test]).astype(np.float32)

    tf.keras.backend.clear_session()
    input_dim = X_train.shape[1]
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,), name='input_layer'),
        tf.keras.layers.Dense(512, activation='relu', name='dense'),
        tf.keras.layers.BatchNormalization(name='batch_normalization'),
        tf.keras.layers.Dropout(0.5, name='dropout'),
        tf.keras.layers.Dense(256, activation='relu', name='dense_1'),
        tf.keras.layers.BatchNormalization(name='batch_normalization_1'),
        tf.keras.layers.Dropout(0.4, name='dropout_1'),
        tf.keras.layers.Dense(1, activation='sigmoid', name='dense_2'),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='binary_crossentropy',
        metrics=[tf.keras.metrics.AUC(name='auc'), tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')],
    )

    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_auc', patience=5, restore_best_weights=True, mode='max')]
    model.fit(X_train, y_train, validation_split=0.1, epochs=40, batch_size=256, callbacks=callbacks, verbose=0)

    y_proba_test = model.predict(X_test, batch_size=512, verbose=0).ravel()
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba_test)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_threshold = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5
    best_f1 = float(f1_scores[best_idx])

    metrics = {
        'classification_report': classification_report(y_test, (y_proba_test >= best_threshold).astype(int), output_dict=True),
        'confusion_matrix': confusion_matrix(y_test, (y_proba_test >= best_threshold).astype(int)).tolist(),
        'best_threshold': best_threshold,
        'best_f1': best_f1,
        'f1_at_0.50': float(f1_score(y_test, (y_proba_test >= 0.5).astype(int))),
    }

    pipeline = {
        'tfidf_cv': tfidf_cv,
        'svd_cv': svd_cv,
        'tfidf_job': tfidf_job,
        'svd_job': svd_job,
        'scaler': scaler,
    }

    joblib.dump(pipeline, output_dir / 'data_pipeline.joblib')
    model.save(output_dir / 'model_mlp_lsa.h5', include_optimizer=False)
    (output_dir / 'training_report.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    thresholds_payload = {'mlp_thresholds': {'thr_f1': best_threshold, 'thr_p60': best_threshold}}
    (output_dir / 'thresholds.json').write_text(json.dumps(thresholds_payload, indent=2), encoding='utf-8')

    registry = {
        'model_type': 'mlp_lsa',
        'pipeline': str(output_dir / 'data_pipeline.joblib'),
        'model': str(output_dir / 'model_mlp_lsa.h5'),
        'thresholds': str(output_dir / 'thresholds.json'),
        'metrics': {'best_f1': best_f1, 'best_threshold': best_threshold},
    }
    (output_dir / 'registry.json').write_text(json.dumps(registry, indent=2), encoding='utf-8')

    # Numeric feature importance via logistic regression
    scaler_num = StandardScaler()
    X_num_train = scaler_num.fit_transform(train_df[NUMERIC_FEATURES])
    logreg = LogisticRegression(max_iter=1000, class_weight='balanced')
    logreg.fit(X_num_train, y_train)
    importance = {feature: float(weight) for feature, weight in zip(NUMERIC_FEATURES, logreg.coef_[0])}
    metrics['feature_importance_numeric'] = importance
    (output_dir / 'training_report.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')

    return best_f1, best_threshold


__all__ = ['train_model']
