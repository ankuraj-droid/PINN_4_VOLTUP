"""
Inference core for the PINN SOH API.

Loads the trained `Solution_u` network (the SOH estimator inside the PINN) and
reproduces the exact preprocessing used during training in
`dataloader/dataloader.py`, then returns State of Health as a percentage.

Notes on correctness
--------------------
* The model input is 17-dim: the 16 charge-cycle features (in the fixed order 
  below) plus a `cycle index`. The network outputs capacity / nominal_capacity,
  i.e. SOH as a fraction -> *100 gives the percentage.
* Training normalises features PER BATTERY (min-max / z-score over that
  battery's whole life). So the accurate path is to pass a battery's full cycle
  history (`predict_battery`): we re-run the identical per-battery pipeline.
* A single isolated cycle cannot be normalised the way training did. The
  `predict_cycle` path falls back to GLOBAL stats (from the HUST training set,
  see model/norm_stats.json) and is therefore approximate. Prefer batches.
"""

import os
import json
import numpy as np
import pandas as pd
import torch

# Make the project root importable so we can reuse the real model definition.
import sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from Model.Model import Solution_u  # noqa: E402

# Canonical feature order expected in every CSV / request (matches the dataset).
FEATURE_COLUMNS = [
    'voltage mean', 'voltage std', 'voltage kurtosis', 'voltage skewness',
    'CC Q', 'CC charge time', 'voltage slope', 'voltage entropy',
    'current mean', 'current std', 'current kurtosis', 'current skewness',
    'CV Q', 'CV charge time', 'current slope', 'current entropy',
]

# Nominal capacity (Ah) per dataset -> used to turn predicted fraction into SOH.
NOMINAL_CAPACITY = {'HUST': 1.1, 'VOLTUP': 30.0}

_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model')
DEFAULT_MODEL_PATH = os.environ.get('SOH_MODEL_PATH', os.path.join(_MODEL_DIR, 'model.pth'))
DEFAULT_STATS_PATH = os.environ.get('SOH_STATS_PATH', os.path.join(_MODEL_DIR, 'norm_stats.json'))


class SOHPredictor:
    def __init__(self, model_path=DEFAULT_MODEL_PATH, stats_path=DEFAULT_STATS_PATH,
                 dataset='HUST', normalization_method='min-max'):
        self.device = 'cpu'
        self.dataset = dataset
        self.nominal_capacity = NOMINAL_CAPACITY.get(dataset, 1.1)
        self.normalization_method = normalization_method

        # Load only the SOH estimator (solution_u); the physics net F is not
        # needed for forward prediction.
        self.model = Solution_u().to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['solution_u'])
        self.model.eval()
        self.model_path = model_path

        # Global normalization stats for the single-cycle fallback.
        self.stats = None
        if stats_path and os.path.exists(stats_path):
            with open(stats_path) as fp:
                self.stats = json.load(fp)

    # ---- preprocessing helpers (mirror dataloader/dataloader.py) ----------

    @staticmethod
    def _3_sigma_index(series):
        rule = (series.mean() - 3 * series.std() > series) | (series.mean() + 3 * series.std() < series)
        return np.arange(series.shape[0])[rule]

    def _delete_3_sigma(self, df):
        df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
        out_index = []
        for col in df.columns:
            out_index.extend(self._3_sigma_index(df[col]))
        df = df.drop(list(set(out_index)), axis=0).reset_index(drop=True)
        return df

    def _normalize_per_battery(self, f_df):
        if self.normalization_method == 'min-max':
            return 2 * (f_df - f_df.min()) / (f_df.max() - f_df.min()) - 1
        elif self.normalization_method == 'z-score':
            return (f_df - f_df.mean()) / f_df.std()
        return f_df

    def _normalize_global(self, f_df):
        """Normalize using bundled training-set statistics (approximate)."""
        if self.stats is None:
            raise RuntimeError('Global normalization stats not available.')
        cols = self.stats['columns']
        f_df = f_df[cols]
        if self.normalization_method == 'min-max':
            mn = np.array(self.stats['min']); mx = np.array(self.stats['max'])
            return 2 * (f_df.values - mn) / (mx - mn) - 1
        else:
            mean = np.array(self.stats['mean']); std = np.array(self.stats['std'])
            return (f_df.values - mean) / std

    @torch.no_grad()
    def _forward(self, x):
        t = torch.from_numpy(np.asarray(x, dtype=np.float32)).to(self.device)
        soh_fraction = self.model(t).cpu().numpy().reshape(-1)
        return soh_fraction

    # ---- public API -------------------------------------------------------

    def predict_battery(self, df, apply_3sigma=True):
        """
        Accurate path: `df` holds one battery's cycles (rows) with the 16
        feature columns (capacity optional, ignored for input). Reproduces the
        per-battery training pipeline and returns SOH% for each surviving cycle.
        """
        missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f'Missing required feature columns: {missing}')

        work = df[FEATURE_COLUMNS].astype(float).copy()
        work['cycle index'] = np.arange(work.shape[0], dtype=float)

        if apply_3sigma and work.shape[0] > 2:
            work = self._delete_3_sigma(work)

        kept_cycles = work['cycle index'].astype(int).tolist()
        feat = self._normalize_per_battery(work)
        soh = self._forward(feat.values) * 100.0

        return [
            {'cycle_index': int(c), 'soh_percent': round(float(s), 2)}
            for c, s in zip(kept_cycles, soh)
        ]

    def predict_cycle(self, features, cycle_index=0):
        """
        Convenience path: one cycle given as a dict of the 16 features.
        Uses GLOBAL (training-set) normalization -> approximate. For best
        accuracy, send the battery's full history via predict_battery.
        """
        missing = [c for c in FEATURE_COLUMNS if c not in features]
        if missing:
            raise ValueError(f'Missing required features: {missing}')

        row = {c: float(features[c]) for c in FEATURE_COLUMNS}
        row['cycle index'] = float(cycle_index)
        f_df = pd.DataFrame([row])
        feat = self._normalize_global(f_df)
        soh = float(self._forward(feat)[0] * 100.0)
        return round(soh, 2)


# A process-wide singleton so the model loads once.
_PREDICTOR = None


def get_predictor():
    global _PREDICTOR
    if _PREDICTOR is None:
        _PREDICTOR = SOHPredictor()
    return _PREDICTOR
