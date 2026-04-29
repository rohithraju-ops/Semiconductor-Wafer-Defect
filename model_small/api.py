"""
api.py
FastAPI backend for WaferVision dashboard — updated for revamped EnsembleWaferPredictor.

Endpoints:
  GET  /health   → liveness check
  GET  /results  → parses results/comparison_report.md → live F1 JSON
  GET  /stats    → prediction counts per class from PostgreSQL
  GET  /history  → last N predictions from PostgreSQL
  POST /predict  → upload image → EnsembleWaferPredictor → classify + log to DB

Run:
  uvicorn api:app --reload --port 8000
"""

from __future__ import annotations

import re
import os
import sys
from pathlib import Path
from datetime import datetime
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# ── DB URL ────────────────────────────────────────────────────────
# SQLite:   "sqlite+aiosqlite:///./predictions.db"
# Postgres: "postgresql+asyncpg://user:pass@host:5432/waferdb"
# Supabase: "postgresql+asyncpg://postgres:[pw]@db.[ref].supabase.co:5432/postgres"
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite+aiosqlite:///./predictions.db")

# ── Paths ─────────────────────────────────────────────────────────
# WAFER_ROOT should point to the model_small directory (where predict.py lives)
ROOT = Path(os.environ.get("WAFER_ROOT", "."))
RESULTS_DIR   = ROOT / "results"
COMPARISON_MD = RESULTS_DIR / "comparison_report.md"
CHECKPOINTS   = ROOT / "checkpoints"

CLASS_NAMES = ["Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc",
               "Near-full", "Random", "Scratch", "none"]

# ── Lazy predictor ────────────────────────────────────────────────
predictor = None  # EnsembleWaferPredictor, loaded once at startup


# ═══════════════════════════════════════════════════════════════════
# RESULTS PARSER
# ═══════════════════════════════════════════════════════════════════

def parse_comparison_report(path: Path) -> dict:
    """
    Parse results/comparison_report.md and return structured JSON.

    Expected format (from evaluate_both.py output):
      | macro F1 | 0.8834 | 0.8134 | -0.0700 |
      | accuracy | 0.8850 | 0.8155 | -0.0695 |
      | Center   | 0.9359 | 0.9196 | -0.0163 |
      Temperature scaling T = 0.8745
    """
    text = path.read_text()

    result = {
        "stage1": {"macro_f1": None, "accuracy": None, "per_class_f1": {}},
        "stage2": {"macro_f1": None, "accuracy": None, "per_class_f1": {}},
        "temperature": None,
        "gate_passed": True,
        "gate_failed_classes": [],
        "source": str(path.relative_to(ROOT)) if ROOT in path.parents else str(path),
    }

    m = re.search(r"macro F1\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)", text)
    if m:
        result["stage1"]["macro_f1"] = float(m.group(1))
        result["stage2"]["macro_f1"] = float(m.group(2))

    a = re.search(r"accuracy\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)", text)
    if a:
        result["stage1"]["accuracy"] = float(a.group(1))
        result["stage2"]["accuracy"] = float(a.group(2))

    t = re.search(r"Temperature scaling T\s*=\s*([\d.]+)", text)
    if t:
        result["temperature"] = float(t.group(1))

    for cls in CLASS_NAMES:
        pattern = rf"\|\s*{re.escape(cls)}\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)"
        m2 = re.search(pattern, text)
        if m2:
            result["stage1"]["per_class_f1"][cls] = float(m2.group(1))
            result["stage2"]["per_class_f1"][cls] = float(m2.group(2))

    gate_section = re.search(r"FAILED.*?(?=##|\Z)", text, re.DOTALL)
    if gate_section:
        result["gate_passed"] = False
        failed = re.findall(r"-\s*([\w-]+):", gate_section.group())
        result["gate_failed_classes"] = failed
    else:
        result["gate_passed"] = "PASSED" in text

    return result


# ═══════════════════════════════════════════════════════════════════
# APP LIFESPAN — load EnsembleWaferPredictor once at startup
# ═══════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor

    # Ensure model_small is on the path so predict.py imports work
    model_small_dir = ROOT
    if str(model_small_dir) not in sys.path:
        sys.path.insert(0, str(model_small_dir))

    await init_db()

    vit_ckpt    = CHECKPOINTS / "vit_best.pth"
    resnet_ckpt = CHECKPOINTS / "resnet_best.pth"

    if vit_ckpt.exists() and resnet_ckpt.exists():
        try:
            from predict import EnsembleWaferPredictor
            predictor = EnsembleWaferPredictor(
                vit_ckpt=vit_ckpt,
                resnet_ckpt=resnet_ckpt,
                ckpt_dir=CHECKPOINTS,
            )
            print(f"[startup] Loaded EnsembleWaferPredictor (ViT + ResNet)")
        except Exception as e:
            print(f"[startup] Model load failed: {e} — /predict will return 503")
    else:
        missing = []
        if not vit_ckpt.exists():    missing.append("vit_best.pth")
        if not resnet_ckpt.exists(): missing.append("resnet_best.pth")
        print(f"[startup] Missing checkpoints: {missing} — /predict unavailable")

    yield
    print("[shutdown] Bye.")


app = FastAPI(title="WaferVision API", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ═══════════════════════════════════════════════════════════════════
# DATABASE HELPERS
# ═══════════════════════════════════════════════════════════════════

import databases
import sqlalchemy

database = databases.Database(DATABASE_URL)

metadata = sqlalchemy.MetaData()
predictions_table = sqlalchemy.Table(
    "predictions", metadata,
    sqlalchemy.Column("id",              sqlalchemy.Integer, primary_key=True),
    sqlalchemy.Column("wafer_id",        sqlalchemy.String,  nullable=False),
    sqlalchemy.Column("predicted_class", sqlalchemy.String,  nullable=False),
    sqlalchemy.Column("class_idx",       sqlalchemy.Integer, nullable=False),
    sqlalchemy.Column("confidence",      sqlalchemy.Float),
    sqlalchemy.Column("method",          sqlalchemy.String,  default="ensemble"),
    sqlalchemy.Column("vit_pred",        sqlalchemy.String),   # per-model sub-predictions
    sqlalchemy.Column("resnet_pred",     sqlalchemy.String),   # per-model sub-predictions
    sqlalchemy.Column("created_at",      sqlalchemy.DateTime, default=datetime.utcnow),
)


async def init_db():
    engine = sqlalchemy.create_engine(
        DATABASE_URL.replace("+asyncpg", "").replace("+aiosqlite", "")
    )
    metadata.create_all(engine)
    await database.connect()


async def log_prediction(wafer_id: str, result: dict, method: str):
    await database.execute(predictions_table.insert().values(
        wafer_id        = wafer_id,
        predicted_class = result["class_name"],
        class_idx       = result["class_idx"],
        confidence      = result.get("confidence"),
        method          = method,
        vit_pred        = result.get("vit_pred"),
        resnet_pred     = result.get("resnet_pred"),
        created_at      = datetime.utcnow(),
    ))


# ═══════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════════

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": predictor is not None,
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/results")
async def get_results():
    """Parse results/comparison_report.md and return live F1 JSON."""
    if not COMPARISON_MD.exists():
        raise HTTPException(
            status_code=404,
            detail=f"{COMPARISON_MD} not found. Run evaluate_both.py first.",
        )
    try:
        data = parse_comparison_report(COMPARISON_MD)
        return JSONResponse(data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Parse error: {e}")


@app.get("/stats")
async def get_stats():
    """Prediction counts per class from the DB."""
    rows = await database.fetch_all(
        "SELECT predicted_class, COUNT(*) as count FROM predictions GROUP BY predicted_class"
    )
    total = await database.fetch_val("SELECT COUNT(*) FROM predictions")
    return {
        "total": total or 0,
        "by_class": {r["predicted_class"]: r["count"] for r in rows},
    }


@app.get("/history")
async def get_history(limit: int = 20):
    """Last N predictions — serialises datetime to ISO string for JSON."""
    rows = await database.fetch_all(
        f"SELECT * FROM predictions ORDER BY created_at DESC LIMIT {limit}"
    )
    records = []
    for r in rows:
        rec = dict(r)
        if isinstance(rec.get("created_at"), datetime):
            rec["created_at"] = rec["created_at"].isoformat()
        records.append(rec)
    return records


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    method: str = Form(default="ensemble"),
):
    """
    Upload a wafer image → EnsembleWaferPredictor → log to DB → return result.

    method values accepted (maps old + new names):
      - ensemble / direct   → ensemble prediction (ViT + ResNet weighted average)
      - tta                  → ensemble with test-time augmentation (4 rotations)
      - vit                  → ViT only
      - resnet               → ResNet only
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Provide checkpoints.")

    import tempfile, uuid
    wafer_id = str(uuid.uuid4())[:12]

    suffix = Path(file.filename).suffix if file.filename else ".png"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = Path(tmp.name)

    try:
        # Map legacy/new method names → EnsembleWaferPredictor calls
        if method in ("ensemble", "direct"):
            result = predictor.predict_ensemble(tmp_path, use_tta=False)
            used_method = "ensemble"
        elif method == "tta":
            result = predictor.predict_ensemble(tmp_path, use_tta=True)
            used_method = "tta"
        elif method == "vit":
            result = predictor.predict_single(tmp_path, model_name="vit")
            used_method = "vit"
        elif method == "resnet":
            result = predictor.predict_single(tmp_path, model_name="resnet")
            used_method = "resnet"
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown method '{method}'. Use: ensemble, tta, vit, resnet"
            )

        await log_prediction(wafer_id, result, used_method)

        # Normalise softmax_probs key for frontend compatibility:
        #   predict_ensemble → "ensemble_probs"
        #   predict_single   → "probs"
        probs_raw = result.get("ensemble_probs", result.get("probs", []))
        softmax_probs = probs_raw.tolist() if hasattr(probs_raw, "tolist") else list(probs_raw)

        return {
            "wafer_id":        wafer_id,
            "predicted_class": result["class_name"],
            "class_idx":       result["class_idx"],
            "confidence":      result.get("confidence"),
            "method":          used_method,
            "softmax_probs":   softmax_probs,          # frontend expects this key
            "vit_pred":        result.get("vit_pred"),
            "resnet_pred":     result.get("resnet_pred"),
            "timestamp":       datetime.utcnow().isoformat(),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        tmp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
