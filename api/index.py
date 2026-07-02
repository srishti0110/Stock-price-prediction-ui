"""
==================================================================
 STOCK PRICE PREDICTION API  -  Vercel-compatible version
==================================================================
This is the SAME logic as our original main.py, just adjusted to
work as a Vercel Serverless Function:

1. All routes now live under /api/... instead of just /...
   (because Vercel maps everything inside the "api" folder to /api)
2. Vercel may "cold start" a fresh copy of this file on every
   request, so we keep loading fast and simple - no big startup
   delays.
3. We don't use FastAPI's StaticFiles here - the index.html page
   is served separately by Vercel as a plain static file.

Folder structure expected by Vercel:

project/
├── api/
│   └── index.py                  <- this file
│   ├── preprocessor.pkl
│   └── stock_prediction_model.pkl
├── data/
│   └── stock_data_messy.csv
├── index.html                    <- served automatically at "/"
├── vercel.json
└── requirements.txt
==================================================================
"""

import os
import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

# ------------------------------------------------------------------
# STEP 0: Basic setup
# ------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))       # .../api
PROJECT_DIR = os.path.dirname(BASE_DIR)                      # project root

app = FastAPI(title="Stock Prediction API")

KNOWN_COMPANIES = ["AAPL", "AMZN", "GOOGL", "MSFT", "TSLA"]


# ------------------------------------------------------------------
# STEP 1: Load and clean the messy CSV
# (Same cleaning logic as before: fix column names, fix company
# names, parse dates, calculate MA20/MA50.)
# ------------------------------------------------------------------
def load_and_clean_data():
    csv_path = os.path.join(PROJECT_DIR, "data", "stock_data_messy.csv")
    df = pd.read_csv(csv_path)

    # Fix column names the SAME generic way the notebook did:
    # strip spaces, lowercase, then capitalize -> "open"/"Open "/"OPEN" all become "Open"
    df.columns = df.columns.str.strip().str.lower().str.capitalize()

    # Fill missing numeric values with the column mean (matches notebook step 4)
    numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean()).round(2)

    # Clean the Company column (matches notebook step 6)
    df["Company"] = df["Company"].astype(str).str.strip().str.upper()

    # Parse dates with 'mixed' format + dayfirst=True, NOT a rigid single format.
    # The rigid format used previously silently turned mismatched rows into NaT,
    # which then got dropped -> missing companies / not-enough-history errors.
    df["Date"] = pd.to_datetime(df["Date"], format="mixed", dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Date"])

    # Remove exact duplicate Company+Date rows (matches notebook step 5)
    df = df.drop_duplicates(subset=["Company", "Date"], keep="first")

    df = df.sort_values(["Company", "Date"]).reset_index(drop=True)

    # Remove impossible / suspicious values (matches notebook step 8)
    df = df[df["Close"] >= 0]
    df = df[df["High"] < 1000]
    df = df[df["Volume"] != 0]

    # Remove single-day spikes >25% that bounce right back - bad data, not real
    # price moves (matches notebook step 9)
    pct_change_temp = df.groupby("Company")["Close"].pct_change() * 100
    bad_spike_mask = (pct_change_temp < -25) | (pct_change_temp > 25)
    df = df[~bad_spike_mask]

    df = df.sort_values(["Company", "Date"]).reset_index(drop=True)

    df["MA20"] = df.groupby("Company")["Close"].transform(lambda x: x.rolling(20).mean())
    df["MA50"] = df.groupby("Company")["Close"].transform(lambda x: x.rolling(50).mean())

    return df


# ------------------------------------------------------------------
# STEP 2: Load everything once when this function "cold starts".
# Vercel may reuse this same loaded copy for a little while if
# requests come in close together ("warm" function), so this isn't
# always re-run on every single request.
# ------------------------------------------------------------------
stock_data = load_and_clean_data()

model_load_error = None

try:
    preprocessor = joblib.load(os.path.join(BASE_DIR, "preprocessor.pkl"))
    model = joblib.load(os.path.join(BASE_DIR, "stock_prediction_model.pkl"))
except Exception as e:
    import traceback
    preprocessor = None
    model = None
    model_load_error = f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}"
    print(f"Error loading model files: {e}")
    print(traceback.format_exc())


# ------------------------------------------------------------------
# STEP 3: API endpoints
# NOTE: every route starts with /api/... because Vercel routes all
# traffic under /api/* to this file.
# ------------------------------------------------------------------

@app.get("/")
def serve_index():
    """Serve index.html for local dev only. On Vercel, index.html is served
    separately as a static file (from the public/ folder, per Vercel's
    convention) and this route should never actually be hit - but guard it
    anyway just in case."""
    candidates = [
        os.path.join(PROJECT_DIR, "index.html"),
        os.path.join(PROJECT_DIR, "public", "index.html"),
        os.path.join(PROJECT_DIR, "static", "index.html"),
    ]
    for index_path in candidates:
        if os.path.exists(index_path):
            return FileResponse(index_path)
    return {"message": "Stock Prediction API - see /api/companies, /api/history/{company}, /api/predict/{company}"}


@app.get("/api/debug")
def debug_info():
    """TEMPORARY: shows why the model/preprocessor failed to load, directly in the browser."""
    return {
        "model_loaded": model is not None,
        "preprocessor_loaded": preprocessor is not None,
        "base_dir": BASE_DIR,
        "files_in_base_dir": os.listdir(BASE_DIR),
        "load_error": model_load_error,
    }


@app.get("/api/companies")
def get_companies():
    """Return the list of companies the user can choose from."""
    return {"companies": KNOWN_COMPANIES}


@app.get("/api/history/{company}")
def get_history(company: str):
    """Return past closing prices for a company, for the chart."""
    company = company.strip().upper()

    if company not in KNOWN_COMPANIES:
        raise HTTPException(status_code=404, detail=f"Unknown company: {company}")

    company_df = stock_data[stock_data["Company"] == company].copy()

    if company_df.empty:
        raise HTTPException(status_code=404, detail=f"No data found for {company}")

    return {
        "company": company,
        "dates": company_df["Date"].dt.strftime("%Y-%m-%d").tolist(),
        "close": company_df["Close"].round(2).tolist(),
    }


@app.get("/api/predict/{company}")
def predict_next_close(company: str):
    """Predict TOMORROW's closing price for the chosen company."""
    if model is None or preprocessor is None:
        raise HTTPException(status_code=500, detail="Model not loaded on server")

    company = company.strip().upper()
    if company not in KNOWN_COMPANIES:
        raise HTTPException(status_code=404, detail=f"Unknown company: {company}")

    company_df = stock_data[stock_data["Company"] == company].copy()
    company_df = company_df.dropna(subset=["MA20", "MA50"])

    if company_df.empty:
        raise HTTPException(
            status_code=400,
            detail=f"Not enough history for {company} to calculate MA20/MA50",
        )

    latest_row = company_df.iloc[-1]

    input_df = pd.DataFrame([{
        "Company": latest_row["Company"],
        "Open": latest_row["Open"],
        "High": latest_row["High"],
        "Low": latest_row["Low"],
        "Volume": latest_row["Volume"],
        "MA20": latest_row["MA20"],
        "MA50": latest_row["MA50"],
    }])

    # TEMPORARY DEBUG WRAPPER: surfaces the real exception in the response
    # instead of a blank 500, so we can see exactly what's failing.
    # Remove this try/except once the root cause is fixed.
    try:
        input_transformed = preprocessor.transform(input_df)
        predicted_close = float(model.predict(input_transformed)[0])
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(tb)  # also shows up in Vercel's Runtime Logs
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {type(e).__name__}: {e}",
        )

    return {
        "company": company,
        "last_known_date": latest_row["Date"].strftime("%Y-%m-%d"),
        "last_known_close": round(float(latest_row["Close"]), 2),
        "predicted_next_close": round(predicted_close, 2),
    }


@app.get("/api/")
def health_check():
    """Simple endpoint to check the API is alive."""
    return {"message": "Stock Prediction API is live!", "model_loaded": model is not None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("index:app", host="0.0.0.0", port=8000, reload=True)