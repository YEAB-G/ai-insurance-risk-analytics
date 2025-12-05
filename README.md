
# AI Insurance Risk Analytics

End-to-end **auto insurance risk & pricing** project built with modern AI/ML tooling.
This repository is structured for portfolio use and follows good MLOps practices
(Git, CI, DVC, modular code, notebooks, and reports).

## Business Problem

AlphaCare Insurance Solutions (ACIS) wants to:
- Understand risk and profitability across customers, vehicles, and locations.
- Find **low-risk segments** where premiums can be reduced to attract more clients.
- Build **predictive models** for claim risk and premium optimization.

You work as a marketing analytics engineer and use historical claim data
(Feb 2014 – Aug 2015) to answer these questions. fileciteturn0file0

## Project Structure

```text
ai-insurance-risk-analytics/
├── data/
│   ├── raw/          # Place original data here (not tracked by git)
│   ├── processed/    # Cleaned / transformed data
│   └── interim/      # Temporary outputs
├── notebooks/
│   ├── 01_eda.ipynb                 # Exploratory Data Analysis
│   ├── 02_hypothesis_testing.ipynb  # A/B & statistical tests
│   └── 03_modeling.ipynb            # ML models & interpretability
├── src/
│   ├── data/        # Data loading & preparation scripts
│   ├── features/    # Feature engineering
│   └── models/      # Training scripts for ML models
├── models/          # Saved model artefacts (e.g. .pkl)
├── reports/
│   ├── interim_report.md
│   └── final_report_template.md
├── tests/           # Basic tests for CI
├── .github/
│   └── workflows/ci.yml   # GitHub Actions CI pipeline
├── dvc.yaml         # DVC pipeline definition (skeleton)
├── requirements.txt
└── .gitignore
```

## How to Use This Project

1. **Create & activate a virtual environment** (optional but recommended):

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Add the dataset**

Download the ACIS auto insurance dataset from the challenge instructions and save it as:

```text
data/raw/insurance.csv
```

> The code and notebooks assume this file path.  
> Do **not** commit the raw data to GitHub. Use DVC instead.

4. **Run the notebooks**

Open Jupyter:

```bash
jupyter notebook
```

Then run, in order:

1. `notebooks/01_eda.ipynb`
2. `notebooks/02_hypothesis_testing.ipynb`
3. `notebooks/03_modeling.ipynb`

5. **DVC (Data Version Control)**

Once you have the dataset locally:

```bash
dvc init
dvc remote add -d localstorage ../acis_dvc_storage
dvc add data/raw/insurance.csv
dvc push
```

This keeps your data versions reproducible without pushing large files to Git.

## Mapping to Week 3 Tasks

- **Task 1 – Git & EDA**
  - Repo structure, `README`, `.gitignore`, `ci.yml`
  - `01_eda.ipynb` notebook

- **Task 2 – DVC**
  - `dvc.yaml` pipeline skeleton
  - `data/` folder ready for DVC tracking

- **Task 3 – A/B Hypothesis Testing**
  - `02_hypothesis_testing.ipynb`

- **Task 4 – Statistical & ML Modeling**
  - `03_modeling.ipynb`
  - `src/features`, `src/models` modules

## Portfolio Notes

This repo is named **ai-insurance-risk-analytics** to clearly communicate:
- Domain: Insurance / Risk Analytics
- Skillset: AI & Machine Learning
- Tooling: MLOps-style project layout

You can link this project on your CV/LinkedIn as an example of
**end-to-end AI/ML for insurance pricing & risk**.
