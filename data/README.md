# Data Folder

This folder holds all data used in the project.

- `data/raw/` – **Original data** exactly as downloaded from the source.
  - Place the ACIS insurance dataset here, e.g. `insurance.csv`.
  - Example expected path: `data/raw/insurance.csv`.
- `data/processed/` – Cleaned data produced by `src/data/prepare_data.py`.
- `data/interim/` – Temporary files or samples created during EDA.

> Important: Do **not** commit raw data to GitHub.  
> Instead, use **DVC** (`dvc add data/raw/insurance.csv`) so that data
> is versioned without bloating the repository.
