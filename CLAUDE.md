# CLAUDE.md

Context for Claude when working in this repository.

## About the project

**Corporate Credit Risk Prediction** is a side project that predicts corporate financial distress or bankruptcy from financial statement data. The long-term goal is a small "Corporate Financial Health & Risk Assessment" tool. A user picks a company and gets its key ratios, a calibrated distress probability and an explanation of what drives it (SHAP).

The author is an AI & Data Science undergraduate. The project is a portfolio piece for roles in data/ML and finance (credit risk, leveraged finance, DCM, ratings/credit research). It should read as a serious research project, not a class assignment. Decisions are justified and documented, and weaknesses are investigated and reported honestly.

## How to help

- Explain the reasoning behind both the ML choices and the finance choices. The author wants to be able to defend every decision in an interview.
- Break math into short steps, one operation per line.
- Prefer simple, readable code over clever code. Put reusable logic in `src/` and keep exploration in `notebooks/`.
- Don't write large amounts of code without explaining what it does. This is a learning project.
- Point out methodological problems (leakage, bad validation, misleading metrics) even when not asked.

## Roadmap

### Phase 1: Benchmark on the UCI Polish Companies Bankruptcy dataset
- Source: https://archive.ics.uci.edu/dataset/365/polish+companies+bankruptcy+data
- Format: 5 ARFF files (`1year` … `5year`). The `Nyear` file holds financial data from N years before the outcome. In total there are about 43,400 rows and 64 numeric attributes (`Attr1`–`Attr64`, ratios defined in the dataset docs). The label is `class` (1 = bankrupt).
- Known issues: heavy class imbalance (~2–7% positive), many missing values, extreme outliers in the ratios.
- Tasks:
  1. Loader for the ARFF files (`scipy.io.arff` or `liac-arff`). Decode byte-string labels.
  2. Map each `AttrN` to its definition and to a category: profitability, liquidity, leverage, activity/efficiency, size.
  3. EDA: missingness per attribute, distributions, class balance per horizon.
  4. Decide which horizon(s) to model and document why.
  5. Baselines: Altman Z-score (approximated from the available ratios), then logistic regression, then gradient boosting (LightGBM/XGBoost).

### Phase 2: Own dataset from SEC EDGAR
- Financials: the XBRL `companyfacts` / `frames` APIs (https://www.sec.gov/search-filings/edgar-application-programming-interfaces). Free and needs no API key, but requests **must** send a descriptive `User-Agent` header with contact info and stay under 10 requests/second. Reliable XBRL coverage starts around 2009.
- Labels: 8-K filings with **Item 1.03 (Bankruptcy or Receivership)** give the event dates.
- Match each event to the last annual filing available *before* it. Use the **filing date**, not the fiscal period end. Distressed firms often stop filing before the event, so handle the gap explicitly.
- Engineer the ratios yourself (D/E, current ratio, ROA, interest coverage, Altman inputs, etc.).

### Phase 3: App
- A small app in `app/` (Streamlit is fine): company lookup, ratios, distress probability, SHAP explanation and comparison with industry peers.

## Methodology rules (non-negotiable)

- **No look-ahead.** Every feature must be available at the prediction date.
- **Chronological validation** wherever time information exists. If a split has to be random (UCI has no dates), say so clearly as a limitation.
- **Metrics:** never report accuracy alone. Use PR-AUC, ROC-AUC, recall at a fixed false-positive rate and the Brier score.
- **Calibration matters.** The output is a probability, so check calibration curves and calibrate if needed.
- **Fit preprocessing only on training folds.** That covers imputation, scaling, winsorising and resampling; use sklearn `Pipeline`s.
- **Class imbalance:** prefer class weights or threshold tuning. If SMOTE is used, apply it inside the CV folds only.
- Fix random seeds and record experiment settings so results are reproducible.

## Repository structure (target)

```
corporate-credit-risk-prediction/
├── data/
│   ├── raw/            # not committed: downloaded by scripts
│   └── processed/      # not committed
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_development.ipynb
├── src/
│   ├── data_processing.py
│   ├── feature_engineering.py
│   └── train.py
├── models/             # not committed if large
├── reports/            # figures and write-ups
├── app/
├── requirements.txt
├── README.md
└── CLAUDE.md
```

## Conventions

- Python 3.11+, with dependencies pinned in `requirements.txt`.
- Data goes in `data/` and is never committed. Provide a download script instead (`src/download_data.py`).
- Type hints and short docstrings on functions in `src/`.
- Notebooks should run top to bottom and import from `src/` rather than duplicating code.
- Make small, frequent commits with clear messages.
- Record decisions and findings in the README or `reports/`, including what *didn't* work.

## Current status

- [x] GitHub repository created (README and .gitignore)
- [x] CLAUDE.md added
- [x] Project folder structure and `requirements.txt`
- [ ] Download script and ARFF loader for the UCI dataset
- [ ] Attribute dictionary (AttrN → definition → category)
- [ ] EDA notebook
- [ ] Prediction target, horizon and validation strategy defined
- [ ] Altman Z-score and logistic regression baselines

Update this checklist as work progresses.
