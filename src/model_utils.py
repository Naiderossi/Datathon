"""Model loading and inference helpers."""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import joblib
import numpy as np
import pandas as pd

from .utils import cosine_similarity_01

NUMERIC_FEATURES: list[str] = [
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


class _MLPWeights:
    """Helper class that mirrors the TensorFlow MLP in NumPy."""

    def __init__(self, model_path: Path) -> None:
        with h5py.File(model_path, "r") as h5:
            config_attr = h5.attrs.get("model_config")
            eps_by_name: dict[str, float] = {}
            if config_attr is not None:
                config_json = config_attr.decode("utf-8") if isinstance(config_attr, bytes) else config_attr
                try:
                    layers = json.loads(config_json).get("config", {}).get("layers", [])
                except json.JSONDecodeError:
                    layers = []
                for layer in layers:
                    if layer.get("class_name") == "BatchNormalization":
                        cfg = layer.get("config", {})
                        eps_by_name[cfg.get("name", "")] = float(cfg.get("epsilon", 1e-3))

            weights_group = h5["model_weights"]
            self.w0, self.b0 = self._load_dense(weights_group, "dense")
            self.gamma0, self.beta0, self.mean0, self.var0, self.eps0 = self._load_bn(weights_group, "batch_normalization", eps_by_name)
            self.w1, self.b1 = self._load_dense(weights_group, "dense_1")
            self.gamma1, self.beta1, self.mean1, self.var1, self.eps1 = self._load_bn(weights_group, "batch_normalization_1", eps_by_name)
            self.w_out, self.b_out = self._load_dense(weights_group, "dense_2")

    @staticmethod
    def _load_dense(weights_group, name: str) -> tuple[np.ndarray, np.ndarray]:
        grp = weights_group[name][name]
        keys = list(grp.keys())
        kernel_key = next((k for k in keys if "kernel" in k), "kernel")
        bias_key = next((k for k in keys if "bias" in k), "bias")
        return grp[kernel_key][...], grp[bias_key][...]

    @staticmethod
    def _load_bn(weights_group, name: str, eps_lookup: dict[str, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
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
        eps = float(eps_lookup.get(name, 1e-3))
        return gamma, beta, mean, var, eps

    @staticmethod
    def _relu(values: np.ndarray) -> np.ndarray:
        return np.maximum(values, 0.0)

    def forward(self, X: np.ndarray) -> np.ndarray:
        z0 = X @ self.w0 + self.b0
        a0 = self._relu(z0)
        bn0 = self.gamma0 * (a0 - self.mean0) / np.sqrt(self.var0 + self.eps0) + self.beta0
        z1 = bn0 @ self.w1 + self.b1
        a1 = self._relu(z1)
        bn1 = self.gamma1 * (a1 - self.mean1) / np.sqrt(self.var1 + self.eps1) + self.beta1
        z2 = bn1 @ self.w_out + self.b_out
        return 1.0 / (1.0 + np.exp(-np.clip(z2, -60.0, 60.0)))


class MLPArtifact:
    """Loads preprocessing pipeline and dense model for inference."""

    def __init__(self, pipeline_path: str | Path, model_path: str | Path) -> None:
        pipeline_path = Path(pipeline_path)
        model_path = Path(model_path)
        if not pipeline_path.exists():
            raise FileNotFoundError(pipeline_path)
        if not model_path.exists():
            raise FileNotFoundError(model_path)

        self.preprocess = joblib.load(pipeline_path)
        self.tfidf_cv = self.preprocess["tfidf_cv"]
        self.svd_cv = self.preprocess["svd_cv"]
        self.tfidf_job = self.preprocess["tfidf_job"]
        self.svd_job = self.preprocess["svd_job"]
        self.scaler = self.preprocess["scaler"]
        self.weights = _MLPWeights(model_path)

    def transform_row(self, row_dict: dict) -> np.ndarray:
        df = pd.DataFrame([row_dict])
        zcv = self.svd_cv.transform(self.tfidf_cv.transform(df["cv_pt_clean"]))
        zjob = self.svd_job.transform(self.tfidf_job.transform(df["req_text_clean"]))
        xnum = self.scaler.transform(df[NUMERIC_FEATURES].astype(float).values)
        return np.hstack([zcv, zjob, xnum])

    def predict_proba(self, row_dict: dict) -> float:
        vector = self.transform_row(row_dict)
        prob = self.weights.forward(vector).ravel()[0]
        return float(prob)

    def batch_predict(self, features: np.ndarray) -> np.ndarray:
        return self.weights.forward(features).ravel()


__all__ = [
    'MLPArtifact',
    'NUMERIC_FEATURES',
]
