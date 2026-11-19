
import pandas as pd
import numpy as np
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_FILE = BASE_DIR / "data" / "cleaned_crime_data.csv"
OUTPUT_DIR = BASE_DIR / "outputs" / "analysis"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load cleaned dataset
df = pd.read_csv(DATA_FILE)

print("=" * 55)
print("CRIME PATTERN STATISTICAL ANALYSIS")
print("=" * 55)

# 1. Descriptive statistics
overall_stats = df["TOTAL IPC CRIMES"].describe()

overall_stats.to_csv(
    OUTPUT_DIR / "descriptive_statistics.csv"
)

print("\n1. Overall Crime Statistics")
print(overall_stats)

# 2. Year-wise crime trends
yearly = (
    df.groupby("YEAR")["TOTAL IPC CRIMES"]
    .sum()
    .reset_index()
)

yearly["YEARLY_CHANGE"] = yearly["TOTAL IPC CRIMES"].diff()

yearly["YEARLY_CHANGE_PERCENT"] = (
    yearly["TOTAL IPC CRIMES"].pct_change() * 100
)

yearly.to_csv(
    OUTPUT_DIR / "yearly_crime_trends.csv",
    index=False
)

print("\n2. Year-wise Crime Trends")
print(yearly.to_string(index=False))

# 3. State-wise crime ranking
state_ranking = (
    df.groupby("STATE/UT")["TOTAL IPC CRIMES"]
    .sum()
    .reset_index()
    .sort_values("TOTAL IPC CRIMES", ascending=False)
)

state_ranking.to_csv(
    OUTPUT_DIR / "state_crime_ranking.csv",
    index=False
)

print("\n3. Top 10 States/UTs")
print(state_ranking.head(10).to_string(index=False))

# 4. Crime category comparison
excluded = ["STATE/UT", "YEAR", "TOTAL IPC CRIMES"]
crime_columns = [
    col for col in df.columns if col not in excluded
]

category_totals = (
    df[crime_columns]
    .sum()
    .sort_values(ascending=False)
    .reset_index()
)

category_totals.columns = [
    "CRIME_CATEGORY", "TOTAL_RECORDS"
]

category_totals.to_csv(
    OUTPUT_DIR / "crime_category_totals.csv",
    index=False
)

print("\n4. Top 10 Crime Categories")
print(category_totals.head(10).to_string(index=False))

# 5. State-wise crime distribution statistics
state_distribution = (
    df.groupby("STATE/UT")["TOTAL IPC CRIMES"]
    .agg(["mean", "median", "min", "max", "std"])
    .reset_index()
)

state_distribution.to_csv(
    OUTPUT_DIR / "state_crime_distribution.csv",
    index=False
)

# 6. Correlation analysis
selected_categories = [
    "MURDER",
    "RAPE",
    "ROBBERY",
    "BURGLARY",
    "THEFT",
    "RIOTS",
    "CHEATING"
]

correlation = df[selected_categories].corr()

correlation.to_csv(
    OUTPUT_DIR / "crime_correlation_matrix.csv"
)

print("\n6. Correlation Matrix")
print(correlation.round(2))

# 7. Final summary
print("\n" + "=" * 55)
print("STATISTICAL ANALYSIS COMPLETED")
print("=" * 55)

print("Total records:", len(df))
print("States/UTs:", df["STATE/UT"].nunique())
print("Years:", df["YEAR"].nunique())
print("Highest recorded year:",
      int(yearly.loc[yearly["TOTAL IPC CRIMES"].idxmax(), "YEAR"]))
print("Highest crime state:",
      state_ranking.iloc[0]["STATE/UT"])

print("\nAnalysis files saved in:", OUTPUT_DIR)
