# PINN4SOH

Physics-Informed Neural Network for lithium-ion battery State of Health (SOH) estimation.  
Based on: [Nature Communications 2024](https://www.nature.com/articles/s41467-024-48779-z)

---

## Setup

```bash
conda create -n pinn python=3.7.10
conda activate pinn
conda install pytorch=1.7.1 scikit-learn=0.24.2 numpy=1.20.3 pandas=1.3.5 matplotlib=3.3.4
pip install scienceplots
```

> **Python 3.12+ users:** The codebase is compatible — pandas version incompatibility is already fixed.

---

## Data Format

Each battery is one CSV file. Required columns (16 features + target):

| Features | Target |
|---|---|
| voltage mean/std/kurtosis/skewness, CC Q, CC charge time, voltage slope/entropy, current mean/std/kurtosis/skewness, CV Q, CV charge time, current slope, current entropy | `capacity` (raw Ah) |

One row = one charge cycle.

---

## Datasets

| Dataset | Type | Nominal Capacity | Folder |
|---|---|---|---|
| HUST | LFP | 1.1 Ah | `data/HUST data/` |
| VoltUp | LFP | 30.0 Ah | `data/VOLTUP DATA/` |

Raw data sources: [HUST](https://data.mendeley.com/datasets/nsc7hnsg4s/2) · [XJTU](https://wang-fujin.github.io/) · [TJU](https://zenodo.org/record/6405084) · [MIT](https://data.matr.io/1/projects/5c48dd2bc625d700019f3204)

---

## Training

```bash
# Train on HUST dataset
python main_HUST.py

# Train on VoltUp dataset (place CSVs in data/VOLTUP DATA/ first)
python main_VOLTUP.py
```

Results saved to `results/` — 10 experiments run by default, each producing `true_label.npy`, `pred_label.npy`, `model.pth`.

---

## Citation

```bibtex
@article{wang2024physics,
  title={Physics-informed neural network for lithium-ion battery degradation stable modeling and prognosis},
  author={Wang, Fujin and Zhai, Zhi and Zhao, Zhibin and Di, Yi and Chen, Xuefeng},
  journal={Nature Communications},
  volume={15}, number={1}, pages={4332}, year={2024},
  publisher={Nature Publishing Group UK London}
}
```
