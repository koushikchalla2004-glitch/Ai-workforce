
# FastAPI worker + predictor for AI Workforce
# Endpoints: /eda, /train, /predict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import io, base64, json, os, requests, traceback
from typing import List, Optional, Dict, Any

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_absolute_error, mean_squared_error
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

app = FastAPI(title="AI Workforce Worker")

def df_from_url(url: str) -> pd.DataFrame:
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    data = r.content
    # try csv
    try:
        df = pd.read_csv(io.BytesIO(data))
        return df
    except Exception:
        pass
    # try excel
    try:
        df = pd.read_excel(io.BytesIO(data))
        return df
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Unsupported file/parse error: {e}")

def chart_to_b64():
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close()
    return base64.b64encode(buf.getvalue()).decode("utf-8")

class EDARequest(BaseModel):
    dataset_url: str

class EDAResponse(BaseModel):
    eda_json: Dict[str, Any]
    eda_html_b64: str
    inferred: Dict[str, Any]

@app.post("/eda", response_model=EDAResponse)
def eda(req: EDARequest):
    try:
        df = df_from_url(req.dataset_url)
        rows, cols = df.shape
        summary = {
            "rows": rows,
            "cols": cols,
            "columns": []
        }
        # simple dtype/nulls/unique
        for c in df.columns:
            col = df[c]
            summary["columns"].append({
                "name": c,
                "dtype": str(col.dtype),
                "nulls": int(col.isna().sum()),
                "unique": int(col.nunique())
            })

        # infer target candidate: last column with low unique count for classification else last column numeric for regression
        inferred_task = None
        inferred_target = None
        for c in reversed(df.columns.tolist()):
            nun = df[c].nunique(dropna=True)
            if nun <= max(10, int(0.05*rows)):
                inferred_task = "classification"
                inferred_target = c
                break
        if inferred_task is None:
            for c in reversed(df.columns.tolist()):
                if pd.api.types.is_numeric_dtype(df[c]):
                    inferred_task = "regression"
                    inferred_target = c
                    break
        if inferred_task is None:
            inferred_task = "regression"
            inferred_target = df.columns[-1]

        # basic plots (up to 4)
        html_parts = [f"<h2>EDA Report</h2><p>Rows: {rows} | Cols: {cols}</p>"]
        plotted = 0
        for c in df.columns[:4]:
            try:
                plt.figure()
                if pd.api.types.is_numeric_dtype(df[c]):
                    df[c].dropna().hist(bins=30)
                    plt.title(f"Histogram: {c}")
                else:
                    df[c].astype(str).value_counts().head(20).plot(kind="bar")
                    plt.title(f"Top categories: {c}")
                b64 = chart_to_b64()
                html_parts.append(f'<h3>{c}</h3><img src="data:image/png;base64,{b64}"/><hr/>')
                plotted += 1
            except Exception:
                pass
            if plotted >= 4:
                break

        eda_html = "\n".join(html_parts)
        eda_html_b64 = base64.b64encode(eda_html.encode("utf-8")).decode("utf-8")

        eda_json = summary
        inferred = {"task": inferred_task, "target": inferred_target}
        return {"eda_json": eda_json, "eda_html_b64": eda_html_b64, "inferred": inferred}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

class TrainRequest(BaseModel):
    dataset_url: str
    target: Optional[str] = None
    task: Optional[str] = "auto"

class TrainResponse(BaseModel):
    metrics_json: Dict[str, Any]
    predictions_csv_b64: str
    model_pkl_b64: str
    model_card_md_b64: str
    chosen_model: str

@app.post("/train", response_model=TrainResponse)
def train(req: TrainRequest):
    try:
        df = df_from_url(req.dataset_url).copy()
        if req.target is None:
            # naive inference: choose last column
            target = df.columns[-1]
        else:
            target = req.target

        y = df[target]
        X = df.drop(columns=[target])

        # minimal preprocessing: drop rows with NA in target, fillna 0 for numeric, mode for categorical
        mask = y.notna()
        X, y = X[mask], y[mask]

        for c in X.columns:
            if pd.api.types.is_numeric_dtype(X[c]):
                X[c] = X[c].fillna(0)
            else:
                X[c] = X[c].astype(str).fillna("")
        # one-hot encode categoricals
        X = pd.get_dummies(X, drop_first=True)

        # infer task
        task = req.task
        if task == "auto":
            task = "classification" if (y.nunique() <= max(10, int(0.05*len(y)))) else "regression"

        # train/test split
        stratify = y if (task=="classification" and y.nunique()>1) else None
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify)

        results = []
        if task == "classification":
            models = [
                ("logreg", LogisticRegression(max_iter=1000)),
                ("rf", RandomForestClassifier(n_estimators=200, random_state=42))
            ]
            for name, m in models:
                m.fit(X_train, y_train)
                y_pred = m.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average="weighted")
                results.append((name, m, {"accuracy": acc, "f1": f1}))
            # pick best by f1
            best = max(results, key=lambda t: t[2]["f1"])
        else:
            models = [
                ("linreg", LinearRegression()),
                ("rf", RandomForestRegressor(n_estimators=300, random_state=42))
            ]
            for name, m in models:
                m.fit(X_train, y_train)
                y_pred = m.predict(X_test)
                r2 = r2_score(y_test, y_pred)
                rmse = mean_squared_error(y_test, y_pred, squared=False)
                mae = mean_absolute_error(y_test, y_pred)
                results.append((name, m, {"r2": r2, "rmse": rmse, "mae": mae}))
            # pick best by r2
            best = max(results, key=lambda t: t[2]["r2"])

        best_name, best_model, best_metrics = best

        # predictions for test set
        y_pred = best_model.predict(X_test)
        preds = pd.DataFrame({"y_true": y_test, "y_pred": y_pred})
        preds_csv = preds.to_csv(index=False).encode("utf-8")
        predictions_csv_b64 = base64.b64encode(preds_csv).decode("utf-8")

        # serialize model
        buf = io.BytesIO()
        joblib.dump(best_model, buf)
        model_pkl_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        # model card
        card = f"""# Model Card
Task: {task}
Target: {target}
Chosen Model: {best_name}

Metrics:
{json.dumps(best_metrics, indent=2)}

Notes:
- Minimal preprocessing + one-hot encoding.
- Baselines only for speed; consider tuning later.
- Do not use for high-stakes decisions without validation.
"""
        model_card_md_b64 = base64.b64encode(card.encode("utf-8")).decode("utf-8")

        # include task/target in metrics
        metrics_json = {"task": task, "target": target, "chosen_model": best_name, "metrics": best_metrics}

        return {
            "metrics_json": metrics_json,
            "predictions_csv_b64": predictions_csv_b64,
            "model_pkl_b64": model_pkl_b64,
            "model_card_md_b64": model_card_md_b64,
            "chosen_model": best_name
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

class PredictRequest(BaseModel):
    model_url: str
    records: List[dict]

class PredictResponse(BaseModel):
    predictions: List[float]

_model_cache = {}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        if req.model_url not in _model_cache:
            r = requests.get(req.model_url, timeout=60)
            r.raise_for_status()
            _model_cache[req.model_url] = joblib.load(io.BytesIO(r.content))
        model = _model_cache[req.model_url]
        df = pd.DataFrame(req.records)
        # note: expects feature engineering same as training (user should send encoded columns)
        yhat = model.predict(df)
        return {"predictions": yhat.tolist()}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
