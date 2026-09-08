# Base Pay — STEM Salary Estimator

An end-to-end ML project that predicts STEM base salaries from role, experience,
education, and location. Started as an EDA notebook, now a full stack: a
reproducible training script, a Flask API, and a small frontend for entering
a role and getting an estimate.

```
notebooks/   original exploratory analysis (kept for reference)
src/         train.py — the reproducible training pipeline
model/       trained model + metadata land here after training (gitignored)
backend/     Flask API that serves predictions + the frontend
data/        put the raw dataset here (gitignored, see data/README.md)
```

## What's inside

- **`notebooks/base_salary_prediction.ipynb`** — the original EDA: missing-value
  analysis, leakage checks, model comparison (Linear/Ridge/Random Forest/XGBoost),
  cross-validation, hyperparameter tuning, residual analysis, feature importance,
  and robustness checks across random seeds. Nothing here was changed — it's the
  paper trail for *why* the pipeline in `src/train.py` looks the way it does.
- **`src/train.py`** — the same cleaning + feature engineering + XGBoost pipeline,
  as a single script instead of 40 notebook cells. Running it produces the two
  files the app actually needs: the trained pipeline and a small metadata file
  (dropdown options for the frontend, plus the model's test-set MAE/RMSE/R²).
- **`backend/app.py`** — a small Flask API (`/api/metadata`, `/api/predict`) that
  also serves the frontend as static files, so it's one process to deploy.
- **`backend/static/`** — plain HTML/CSS/JS frontend. No build step, no framework —
  just open it through the Flask app.

## Running it locally

```bash
git clone <your-repo-url>
cd salary-predictor
python -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt
```

1. **Train the model:**
   ```bash
   python src/train.py --data data/stem-4_25-update-1.xlsx
   ```
   This writes `model/salary_model.pkl` and `model/metadata.json`, and prints
   the held-out test MAE/RMSE/R².

2. **Run the app:**
   ```bash
   python backend/app.py
   ```
   Open http://127.0.0.1:5000 — fill in the form, get an estimate.

## Deploying it

The Flask app serves both the API and the frontend, so it deploys as a single
web service. A `Procfile` is included for Render/Railway/Heroku-style platforms:

```
web: gunicorn --chdir backend app:app
```

Whatever platform you use, you still need `model/salary_model.pkl` and
`model/metadata.json` present at deploy time — either run `train.py` as part
of your build step, or train locally and commit the two files to a private
branch/artifact store (they're gitignored by default since they're derived
data, not source).

## API

**`GET /api/metadata`**
Returns dropdown options (top companies/titles/countries/etc. by frequency),
numeric ranges, and the model's test-set metrics.

**`POST /api/predict`**
```json
{
  "company": "Google",
  "title": "Software Engineer",
  "nation": "United States",
  "gender": "Male",
  "race": "Asian",
  "education": "Master's Degree",
  "yearsofexperience": 6,
  "yearsatcompany": 2
}
```
```json
{
  "predicted_salary": 165230.11,
  "range_low": 135442.9,
  "range_high": 195017.32
}
```
The range is the point estimate ± the model's test-set RMSE — a rough sense of
typical error, not a formal confidence interval.

## On the demographic fields

The dataset includes gender and race, and the original notebook explicitly
treats their relationship to salary as observational, not causal — differences
in the data reflect a snapshot of the labor market, not a claim about what
*should* determine pay. The form still asks for them because the model was
trained on that feature set and removing them silently would make the
tool's behavior inconsistent with what it actually learned. If you're
adapting this for a real use case, worth deciding deliberately whether to
keep them, and being explicit either way.

## Known limitations (inherited from the notebook)

- The data is self-reported (levels.fyi-style survey, 2017–2021), so it skews
  toward large tech companies and doesn't represent the whole STEM workforce.
- Base salary excludes bonus and equity — total comp will read differently.
- Test R² is roughly 0.7 on the original full dataset — useful for a ballpark,
  not precise enough to anchor a specific offer.
- High-cardinality fields (company, title) are typed with autocomplete; values
  outside the model's training vocabulary are treated as "unknown" and mostly
  ignored rather than causing an error.
