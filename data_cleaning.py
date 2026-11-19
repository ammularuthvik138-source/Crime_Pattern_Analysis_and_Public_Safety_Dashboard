
import pandas as pd
import numpy as np
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent

RAW_FILE = BASE_DIR / "data" / "raw_crime_data.csv"
CLEAN_FILE = BASE_DIR / "data" / "cleaned_crime_data.csv"
REPORT_FILE = BASE_DIR / "outputs" / "analysis" / "data_quality_report.csv"

REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)

# Load original dataset
df = pd.read_csv(RAW_FILE)

original_rows = len(df)

# Standardize column names
df.columns = df.columns.str.strip().str.upper()

# Standardize state names
df["STATE/UT"] = df["STATE/UT"].astype("string").str.strip().str.upper()

# Convert year and crime columns
df["YEAR"] = pd.to_numeric(df["YEAR"], errors="coerce")

crime_columns = [
    col for col in df.columns
    if col not in ["STATE/UT", "YEAR"]
]

for col in crime_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Data quality checks
missing = df.isnull().sum()
duplicates = int(df.duplicated().sum())

duplicate_state_year = int(
    df.duplicated(subset=["STATE/UT", "YEAR"]).sum()
)

negative_values = int((df[crime_columns] < 0).sum().sum())

invalid_years = int(
    (~df["YEAR"].between(2001, 2012)).sum()
)

invalid_states = int(
    (df["STATE/UT"].isna() | (df["STATE/UT"] == "")).sum()
)

# Remove exact duplicates only
cleaned = df.drop_duplicates().copy()

# Remove rows with invalid essential identifiers
cleaned = cleaned[
    cleaned["STATE/UT"].notna()
    & (cleaned["STATE/UT"] != "")
    & cleaned["YEAR"].between(2001, 2012)
].copy()

cleaned["YEAR"] = cleaned["YEAR"].astype(int)

# Create quality report
report = pd.DataFrame({
    "Check": [
        "Original Records",
        "Original Columns",
        "Missing Values",
        "Exact Duplicates",
        "Duplicate State-Year Combinations",
        "Negative Crime Values",
        "Invalid Years",
        "Invalid State Names",
        "Final Records"
    ],
    "Result": [
        original_rows,
        len(df.columns),
        int(missing.sum()),
        duplicates,
        duplicate_state_year,
        negative_values,
        invalid_years,
        invalid_states,
        len(cleaned)
    ]
})

# Export results
cleaned.to_csv(CLEAN_FILE, index=False)
report.to_csv(REPORT_FILE, index=False)

print("\nDATA CLEANING COMPLETED")
print("Original records:", original_rows)
print("Cleaned records:", len(cleaned))
print("Missing values:", int(missing.sum()))
print("Duplicate records:", duplicates)
print("Negative values:", negative_values)
print("\nCleaned dataset saved:", CLEAN_FILE)
print("Quality report saved:", REPORT_FILE)
