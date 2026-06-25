"""
FastAPI service for battery State-of-Health (SOH) estimation.

Give it a charging cycle (or a battery's cycle history) and it returns SOH as a
percentage, using the trained PINN `Solution_u` network.

Run:
    cd <project root>
    /opt/anaconda3/envs/voltup_ml/bin/python -m uvicorn api.app:app --reload --port 8000
Then open http://127.0.0.1:8000/docs
"""

import io
from typing import Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from api.inference import FEATURE_COLUMNS, get_predictor

app = FastAPI(
    title="PINN Battery SOH API",
    description="Estimate lithium-ion battery State of Health (SOH %) from charge-cycle features.",
    version="1.0.0",
)


# ----------------------------- schemas -------------------------------------

class CycleFeatures(BaseModel):
    """The 16 features describing one charge cycle."""
    voltage_mean: float = Field(..., alias="voltage mean")
    voltage_std: float = Field(..., alias="voltage std")
    voltage_kurtosis: float = Field(..., alias="voltage kurtosis")
    voltage_skewness: float = Field(..., alias="voltage skewness")
    CC_Q: float = Field(..., alias="CC Q")
    CC_charge_time: float = Field(..., alias="CC charge time")
    voltage_slope: float = Field(..., alias="voltage slope")
    voltage_entropy: float = Field(..., alias="voltage entropy")
    current_mean: float = Field(..., alias="current mean")
    current_std: float = Field(..., alias="current std")
    current_kurtosis: float = Field(..., alias="current kurtosis")
    current_skewness: float = Field(..., alias="current skewness")
    CV_Q: float = Field(..., alias="CV Q")
    CV_charge_time: float = Field(..., alias="CV charge time")
    current_slope: float = Field(..., alias="current slope")
    current_entropy: float = Field(..., alias="current entropy")

    model_config = {"populate_by_name": True}

    def to_feature_dict(self) -> Dict[str, float]:
        return {alias: getattr(self, name)
                for name, f in self.model_fields.items()
                for alias in [f.alias]}


class CycleRequest(BaseModel):
    cycle_index: int = Field(0, description="Cycle number within the battery's life (0-based).")
    features: CycleFeatures


class CycleResponse(BaseModel):
    soh_percent: float
    cycle_index: int
    note: str


class BatteryRequest(BaseModel):
    """A battery's cycle history: list of feature dicts in chronological order."""
    cycles: List[Dict[str, float]]
    normalization_method: Optional[str] = Field(
        "min-max", description="min-max or z-score (must match how the model was trained)."
    )


# ----------------------------- endpoints -----------------------------------

@app.get("/")
def root():
    return {
        "service": "PINN Battery SOH API",
        "endpoints": {
            "POST /predict/cycle": "One cycle (16 features) -> SOH % (approximate, global normalization).",
            "POST /predict/battery": "Battery cycle history (JSON) -> SOH % per cycle (accurate).",
            "POST /predict/csv": "Upload a battery CSV -> SOH % per cycle (accurate).",
            "GET /health": "Model load status.",
            "GET /features": "Required feature column names.",
        },
        "docs": "/docs",
    }


@app.get("/health")
def health():
    p = get_predictor()
    return {"status": "ok", "model_path": p.model_path,
            "dataset": p.dataset, "nominal_capacity_Ah": p.nominal_capacity}


@app.get("/features")
def features():
    return {"feature_columns": FEATURE_COLUMNS, "count": len(FEATURE_COLUMNS)}


@app.post("/predict/cycle", response_model=CycleResponse)
def predict_cycle(req: CycleRequest):
    """Single charge cycle -> SOH %. Approximate (uses global normalization)."""
    p = get_predictor()
    try:
        soh = p.predict_cycle(req.features.to_feature_dict(), cycle_index=req.cycle_index)
    except (ValueError, RuntimeError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    return CycleResponse(
        soh_percent=soh, cycle_index=req.cycle_index,
        note="Approximate: single-cycle uses global training-set normalization. "
             "For accurate SOH, send the battery's full history to /predict/battery.",
    )


@app.post("/predict/battery")
def predict_battery(req: BatteryRequest):
    """Battery cycle history (JSON list) -> SOH % per cycle (accurate path)."""
    p = get_predictor()
    p.normalization_method = req.normalization_method or "min-max"
    try:
        df = pd.DataFrame(req.cycles)
        results = p.predict_battery(df)
    except (ValueError, RuntimeError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not results:
        raise HTTPException(status_code=400, detail="No valid cycles after preprocessing.")
    return {"count": len(results), "latest": results[-1], "soh": results}


@app.post("/predict/csv")
async def predict_csv(file: UploadFile = File(...),
                      normalization_method: str = "min-max"):
    """Upload a battery CSV (16 feature columns) -> SOH % per cycle (accurate)."""
    p = get_predictor()
    p.normalization_method = normalization_method
    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
        results = p.predict_battery(df)
    except (ValueError, RuntimeError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {e}")
    if not results:
        raise HTTPException(status_code=400, detail="No valid cycles after preprocessing.")
    return {"filename": file.filename, "count": len(results),
            "latest": results[-1], "soh": results}
