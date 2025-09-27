# mlp_infer.py
# Helper de inferência para o modelo MLP+LSA
# - Carrega pipeline (.joblib) e pesos (.h5) serializados
# - Transforma um dicionário de features em vetor denso


from __future__ import annotations

import json
from pathlib import Path

import h5py
import re
import unicodedata
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer

# Colunas numéricas/flags usadas no treino
NUM_COLS = [
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

# Vetorizar coseno rápido para text_score01 (0..1)

def _norm(text: str | None) -> str:
    if text is None:
        return ""
    s = unicodedata.normalize("NFKD", str(text))
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s

_HASH = HashingVectorizer(
    n_features=2**18,
    alternate_sign=False,
    norm="l2",
    ngram_range=(1, 2),
    lowercase=True,
)


def cosine_01(a: str, b: str) -> float:
    a = (a or "").strip()
    b = (b or "").strip()
    if not a or not b:
        return 0.0
    xa = _HASH.transform([a])
    xb = _HASH.transform([b])
    return float((xa @ xb.T).toarray()[0, 0])  # com norm L2 vira coseno


class _MLPWeights:
    """Carrega pesos do MLP a partir do .h5 e implementa inferência numpy."""

    def __init__(self, model_path: str):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(model_path)

        with h5py.File(model_path, "r") as f:
            cfg_attr = f.attrs.get("model_config")
            eps_by_name: dict[str, float] = {}
            if cfg_attr is not None:
                cfg_json = cfg_attr.decode("utf-8") if isinstance(cfg_attr, bytes) else cfg_attr
                try:
                    cfg = json.loads(cfg_json)
                except Exception:
                    cfg = {}
                layers = cfg.get("config", {}).get("layers", [])
                for layer in layers:
                    if layer.get("class_name") == "BatchNormalization":
                        name = layer.get("config", {}).get("name")
                        eps = layer.get("config", {}).get("epsilon", 1e-3)
                        if name:
                            eps_by_name[name] = float(eps)

            weights_group = f["model_weights"]
            self.w0, self.b0 = self._load_dense(weights_group, "dense")
            self.gamma0, self.beta0, self.mean0, self.var0, self.eps0 = self._load_bn(
                weights_group, "batch_normalization", eps_by_name
            )
            self.w1, self.b1 = self._load_dense(weights_group, "dense_1")
            self.gamma1, self.beta1, self.mean1, self.var1, self.eps1 = self._load_bn(
                weights_group, "batch_normalization_1", eps_by_name
            )
            self.w_out, self.b_out = self._load_dense(weights_group, "dense_2")

    @staticmethod
    def _load_dense(weights_group, name: str) -> tuple[np.ndarray, np.ndarray]:
        grp = weights_group[name][name]
        keys = list(grp.keys())
        kernel_key = next((k for k in keys if "kernel" in k), "kernel")
        bias_key = next((k for k in keys if "bias" in k), "bias")
        return grp[kernel_key][...], grp[bias_key][...]

    @staticmethod
    def _load_bn(weights_group, name: str, eps_by_name: dict[str, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        grp = weights_group[name][name]
        keys = list(grp.keys())
        gamma_key = next((k for k in keys if "gamma" in k), "gamma")
        beta_key = next((k for k in keys if "beta" in k), "beta")
        mean_key = next((k for k in keys if "moving_mean" in k), "moving_mean")
        var_key = next((k for k in keys if "moving_variance" in k), "moving_variance")
        gamma = grp[gamma_key][...]
        beta = grp[beta_key][...]
        mean = grp[mean_key][...]
        var = grp[var_key][...]
        eps = float(eps_by_name.get(name, 1e-3))
        return gamma, beta, mean, var, eps

    @staticmethod
    def _relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(x, 0.0)

    def forward(self, x: np.ndarray) -> float:
        z0 = x @ self.w0 + self.b0
        a0 = self._relu(z0)
        bn0 = self.gamma0 * (a0 - self.mean0) / np.sqrt(self.var0 + self.eps0) + self.beta0
        z1 = bn0 @ self.w1 + self.b1
        a1 = self._relu(z1)
        bn1 = self.gamma1 * (a1 - self.mean1) / np.sqrt(self.var1 + self.eps1) + self.beta1
        z2 = bn1 @ self.w_out + self.b_out
        # Sigmóide numérico estável
        return float(1.0 / (1.0 + np.exp(-np.clip(z2, -60.0, 60.0))))


class MLPArtifact:
    """
    Wrapper de inferência:
      - pipeline_path: models/data_pipeline.joblib (tfidf_cv, svd_cv, tfidf_job, svd_job, scaler)
      - model_path:    models/model_mlp_lsa.h5     (pesos Keras)

   
    """

    def __init__(self, pipeline_path: str = "models/data_pipeline.joblib", model_path: str = "models/model_mlp_lsa.h5"):
        self.pre = joblib.load(pipeline_path)
        self.tfidf_cv = self.pre["tfidf_cv"]
        self.svd_cv = self.pre["svd_cv"]
        self.tfidf_job = self.pre["tfidf_job"]
        self.svd_job = self.pre["svd_job"]
        self.scaler = self.pre["scaler"]
        self.weights = _MLPWeights(model_path)

    # -------- features prontas --------
    def transform_row(self, row_dict: dict) -> np.ndarray:
        """Transforma um row_dict (com colunas esperadas) em vetor denso para o MLP."""
        df = pd.DataFrame([row_dict])
        zcv = self.svd_cv.transform(self.tfidf_cv.transform(df["cv_pt_clean"]))
        zjob = self.svd_job.transform(self.tfidf_job.transform(df["req_text_clean"]))
        xnum = self.scaler.transform(df[NUM_COLS].astype(float).values)
        return np.hstack([zcv.ravel(), zjob.ravel(), xnum.ravel()])

    def predict_proba(self, row_dict: dict) -> float:
        """Retorna probabilidade 0..1 para o row_dict informado."""
        X = self.transform_row(row_dict)
        return self.weights.forward(X)

    # --------constrói features de match a partir de insumos crus --------
    @staticmethod
    def build_feature_row(
        *,
        cv_pt_clean: str,
        req_text_clean: str,
        ing_ord: int,
        esp_ord: int,
        acad_ord: int,
        req_ing_ord: int,
        req_esp_ord: int,
        req_acad_ord: int,
        pcd_flag: int,
        job_pcd_req: int,
        has_sap: int,
        job_sap_req: int,
        # opcionais:
        stage_score: int = 0,
        text_score01: float | None = None,
        score_textual: float | None = None,
        skills_list: list[str] | None = None,
    ) -> dict:
        """
        Monta o dicionário de features no mesmo formato do treino.
        - Calcula meets_*, diff_*, pcd_match, sap_match.
        - Se não passar text_score01/score_textual, calcula coseno rápido.
        """

        meets_ing = int(ing_ord >= req_ing_ord)
        meets_esp = int(esp_ord >= req_esp_ord)
        meets_acad = int(acad_ord >= req_acad_ord)

        diff_ing = int(np.clip(ing_ord - req_ing_ord, -3, 3))
        diff_esp = int(np.clip(esp_ord - req_esp_ord, -3, 3))
        diff_acad = int(np.clip(acad_ord - req_acad_ord, -4, 4))

        pcd_match = int(1 if job_pcd_req != 1 else (pcd_flag == 1))
        sap_match = int(1 if job_sap_req != 1 else (has_sap == 1))

        if text_score01 is None or score_textual is None:
            c = cosine_01(cv_pt_clean or "", req_text_clean or "")
            text_score01 = float(c)
            score_textual = float(c * 100.0)

        skills_list = skills_list or []
        skills_norm = {_norm(item) for item in skills_list if _norm(item)}
        req_norm = _norm(req_text_clean)
        skill_overlap = sum(1 for item in skills_norm if item and item in req_norm)
        skill_overlap_ratio = skill_overlap / max(1, len(skills_norm))

        cv_norm = _norm(cv_pt_clean)
        cv_tokens = [tok for tok in cv_norm.split() if tok]
        req_tokens = [tok for tok in req_norm.split() if tok]
        cv_len_tokens = len(cv_tokens)
        req_len_tokens = len(req_tokens)
        len_ratio = cv_len_tokens / (req_len_tokens + 1)
        token_overlap_count = len(set(cv_tokens).intersection(req_tokens))
        token_overlap_ratio = token_overlap_count / max(1, len(set(req_tokens)))

        return {
            "cv_pt_clean": cv_pt_clean or "",
            "req_text_clean": req_text_clean or "",
            "ing_ord": ing_ord,
            "esp_ord": esp_ord,
            "acad_ord": acad_ord,
            "req_ing_ord": req_ing_ord,
            "req_esp_ord": req_esp_ord,
            "req_acad_ord": req_acad_ord,
            "meets_ing": meets_ing,
            "meets_esp": meets_esp,
            "meets_acad": meets_acad,
            "diff_ing": diff_ing,
            "diff_esp": diff_esp,
            "diff_acad": diff_acad,
            "pcd_match": pcd_match,
            "sap_match": sap_match,
            "stage_score": int(stage_score or 0),
            "text_score01": float(text_score01),
            "score_textual": float(score_textual),
            "pcd_flag": int(pcd_flag),
            "job_pcd_req": int(job_pcd_req),
            "has_sap": int(has_sap),
            "job_sap_req": int(job_sap_req),
            "cv_len_tokens": float(cv_len_tokens),
            "req_len_tokens": float(req_len_tokens),
            "len_ratio": float(len_ratio),
            "skill_overlap": float(skill_overlap),
            "skill_overlap_ratio": float(skill_overlap_ratio),
            "token_overlap_ratio": float(token_overlap_ratio),
            "token_overlap_count": float(token_overlap_count),
        }

    def predict_from_raw(self, **kwargs) -> float:
        """
        Atalho: recebe os argumentos de build_feature_row(...), monta o dict e prediz.
        Exemplo mínimo:
            p = ART.predict_from_raw(
                cv_pt_clean=cv, req_text_clean=job,
                ing_ord=2, esp_ord=1, acad_ord=4,
                req_ing_ord=2, req_esp_ord=0, req_acad_ord=4,
                pcd_flag=0, job_pcd_req=0,
                has_sap=1, job_sap_req=1
            )
        """

        row = self.build_feature_row(**kwargs)
        return self.predict_proba(row)
