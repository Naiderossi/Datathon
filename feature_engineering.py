"""Feature engineering steps used by the application and training."""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from .model_utils import NUMERIC_FEATURES, MLPArtifact
from .utils import (
    ACADEMICO,
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


def build_candidate_features(job_row: pd.Series, candidate_row: pd.Series) -> dict:
    job_text = pick_req_text(job_row)
    req_ing_ord = map_level(job_row.get('nivel_ingles_req'))
    req_esp_ord = map_level(job_row.get('nivel_espanhol_req'))
    req_acad_ord = parse_req_acad(job_text)
    job_pcd_req = parse_req_pcd(job_text)
    job_sap_req = parse_req_sap(job_text, job_row.get('vaga_sap'))

    cv_text = pick_cv_text(candidate_row)
    skills_list = safe_list_parse(candidate_row.get('skills_list'))
    skills_text_raw = candidate_row.get('skills_text', '')
    if isinstance(skills_text_raw, str):
        skills_text = skills_text_raw
    elif pd.isna(skills_text_raw):
        skills_text = ''
    else:
        skills_text = str(skills_text_raw)

    def _int_or_default(value, default):
        if value is None:
            return default
        if isinstance(value, str) and not value.strip():
            return default
        if pd.isna(value):
            return default
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return default

    ing_ord = map_level(candidate_row.get('nivel_ingles'))
    esp_ord = map_level(candidate_row.get('nivel_espanhol'))
    acad_ord = map_acad(candidate_row.get('nivel_academico'))

    meets_ing = int(ing_ord >= req_ing_ord)
    meets_esp = int(esp_ord >= req_esp_ord)
    meets_acad = int(acad_ord >= req_acad_ord)

    diff_ing = int(np.clip(ing_ord - req_ing_ord, -3, 3))
    diff_esp = int(np.clip(esp_ord - req_esp_ord, -3, 3))
    diff_acad = int(np.clip(acad_ord - req_acad_ord, -4, 4))

    pcd_flag = int(str(candidate_row.get('pcd', '')).strip().lower() in {'sim', 'true', '1'})
    has_sap = detect_sap(skills_list, skills_text, cv_text)

    skill_overlap, skill_overlap_ratio = compute_skill_overlap(skills_list, job_text)
    token_overlap_count, token_overlap_ratio = compute_token_overlap_metrics(cv_text, job_text)

    pcd_match = int(1 if job_pcd_req != 1 else (pcd_flag == 1))
    sap_match = int(1 if job_sap_req != 1 else (has_sap == 1))

    text_score = cosine_similarity_01(cv_text, job_text)
    score_textual = float(text_score * 100.0)

    req_len_tokens = _int_or_default(job_row.get('req_len_tokens'), len(job_text.split()))
    cv_len_tokens = _int_or_default(candidate_row.get('cv_len_tokens'), len(cv_text.split()))
    len_ratio = cv_len_tokens / float(req_len_tokens + 1)

    features = {
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
    return features


def prepare_candidates_for_job(job_row: pd.Series, candidates: pd.DataFrame) -> pd.DataFrame:
    records = []
    info_rows = []
    for cid, candidate_row in candidates.iterrows():
        features = build_candidate_features(job_row, candidate_row)
        if not features['cv_pt_clean'].strip() or not features['req_text_clean'].strip():
            continue
        features['candidate_id'] = cid
        records.append(features)
        info_rows.append({
            'candidate_id': cid,
            'nome': candidate_row.get('nome', ''),
            'email': candidate_row.get('email', ''),
            'telefone': candidate_row.get('telefone', ''),
            'nivel_ingles': candidate_row.get('nivel_ingles', ''),
            'nivel_espanhol': candidate_row.get('nivel_espanhol', ''),
            'nivel_academico': candidate_row.get('nivel_academico', ''),
            'skills_list': safe_list_parse(candidate_row.get('skills_list')),
            'cv_pt': candidate_row.get('cv_pt', '') or candidate_row.get('cv_pt_clean', ''),
        })

    if not records:
        return pd.DataFrame()

    features_df = pd.DataFrame(records).set_index('candidate_id')
    info_df = pd.DataFrame(info_rows).set_index('candidate_id')
    return features_df.join(info_df, how='left')


def score_candidates(
    job_id: int,
    apps: pd.DataFrame,
    jobs: pd.DataFrame,
    candidate_pool: Iterable[int],
    artifact: MLPArtifact,
) -> pd.DataFrame:
    if job_id not in jobs.index:
        return pd.DataFrame()
    job_row = jobs.loc[job_id]
    candidates = apps.loc[candidate_pool]
    features = prepare_candidates_for_job(job_row, candidates)
    if features.empty:
        return pd.DataFrame()

    tfidf_cv = artifact.tfidf_cv.transform(features['cv_pt_clean'].tolist())
    svd_cv = artifact.svd_cv.transform(tfidf_cv)
    tfidf_job = artifact.tfidf_job.transform(features['req_text_clean'].tolist())
    svd_job = artifact.svd_job.transform(tfidf_job)
    numeric = artifact.scaler.transform(features[NUMERIC_FEATURES].astype(float).values)
    stacked = np.hstack([svd_cv, svd_job, numeric])
    probability = artifact.weights.forward(stacked).ravel()

    result = features.copy()
    result['probability'] = probability
    result['prob_percent'] = (probability * 100).round(2)
    return result


__all__ = [
    'NUMERIC_FEATURES',
    'build_candidate_features',
    'prepare_candidates_for_job',
    'score_candidates',
]
