# Battery SOH API

A small FastAPI service that wraps the trained PINN model and returns a battery's
**State of Health (SOH) as a percentage** from charge-cycle features.

```
give it a charge cycle  ->  SOH = xx %
```

## What's inside

| File | Purpose |
|---|---|
| `app.py` | FastAPI app + HTTP endpoints |
| `inference.py` | Loads `Solution_u`, reproduces the training preprocessing, predicts SOH |
| `model/model.pth` | Trained HUST model (extracted from `results/Ours/HUST.zip`) |
| `model/norm_stats.json` | Global feature statistics from HUST data (for the single-cycle fallback) |
| `requirements.txt` | Python dependencies |
| `run.sh` | Start helper |

The served model is the **HUST** model (nominal capacity 1.1 Ah) — the only trained
checkpoint in the repo. VoltUp has no trained model yet (its data folder is empty).
To serve a different model, set `SOH_MODEL_PATH` (and adjust `dataset` in `inference.py`
for the right nominal capacity).

## Run

```bash
# from the project root
./api/run.sh
# or explicitly:
/opt/anaconda3/envs/voltup_ml/bin/python -m uvicorn api.app:app --reload --port 8000
```

Interactive docs (try requests in the browser): http://127.0.0.1:8000/docs

## Endpoints

### `POST /predict/cycle` — one cycle → SOH % (approximate)
Send the 16 features for a single charge cycle.

```bash
curl -X POST http://127.0.0.1:8000/predict/cycle \
  -H 'Content-Type: application/json' \
  -d '{"cycle_index":0,"features":{
    "voltage mean":3.4557,"voltage std":0.0442,"voltage kurtosis":0.5978,"voltage skewness":1.1054,
    "CC Q":0.2245,"CC charge time":735,"voltage slope":0.000734,"voltage entropy":4.6891,
    "current mean":0.2418,"current std":0.1115,"current kurtosis":-0.8271,"current skewness":0.6026,
    "CV Q":0.01096,"CV charge time":165,"current slope":-0.004194,"current entropy":3.1793}}'
# -> {"soh_percent": 98.21, "cycle_index": 0, "note": "..."}
```

### `POST /predict/battery` — cycle history (JSON) → SOH % per cycle (accurate)
Send a chronological list of cycles (each a dict of the 16 features).

### `POST /predict/csv` — upload a battery CSV → SOH % per cycle (accurate)
```bash
curl -X POST http://127.0.0.1:8000/predict/csv -F 'file=@data/HUST data/1-2.csv'
# -> {"count": 2672, "latest": {"cycle_index": 2671, "soh_percent": 79.6}, "soh": [...]}
```

Other: `GET /health`, `GET /features`, `GET /` (overview).

## Important: why "one cycle" is approximate

Training normalizes features **per battery**, over that battery's entire life
(see `dataloader/dataloader.py`). So the same raw feature maps to a different
normalized value depending on the battery's min/max range.

- `/predict/battery` and `/predict/csv` reproduce that **per-battery** pipeline
  exactly (insert cycle index → 3-sigma outlier removal → min-max/z-score), so
  they are the **accurate** paths. Pass as much of the battery's history as you have.
- `/predict/cycle` has no battery context, so it falls back to **global**
  statistics from the HUST training set. Treat its number as an estimate.

The model outputs capacity ÷ nominal capacity (SOH fraction); the API multiplies
by 100. Early-life LFP cells can read slightly above 100 % (capacity above nominal).
