
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent

DATA_FILE = BASE_DIR / "data" / "cleaned_crime_data.csv"
PLOT_DIR = BASE_DIR / "outputs" / "plots"

PLOT_DIR.mkdir(parents=True, exist_ok=True)

# Load cleaned dataset
df = pd.read_csv(DATA_FILE)

plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 10

print("GENERATING CRIME VISUALIZATIONS...")

# 1. BOX PLOT - Crime distribution by year
years = sorted(df["YEAR"].unique())

box_data = [
    df.loc[df["YEAR"] == year, "TOTAL IPC CRIMES"]
    for year in years
]

plt.figure(figsize=(12, 6))
plt.boxplot(box_data, tick_labels=years, showmeans=True)
plt.title("Distribution of Total IPC Crimes Across States by Year")
plt.xlabel("Year")
plt.ylabel("Total IPC Crimes")
plt.xticks(rotation=45)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(PLOT_DIR / "boxplot.png", dpi=300)
plt.close()

# 2. HISTOGRAM - Crime distribution
plt.figure()
plt.hist(
    df["TOTAL IPC CRIMES"],
    bins=25,
    edgecolor="black"
)
plt.title("Distribution of Total IPC Crimes")
plt.xlabel("Total IPC Crimes")
plt.ylabel("Frequency")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(PLOT_DIR / "histogram.png", dpi=300)
plt.close()

# 3. SCATTER PLOT - Murder vs Rape
plt.figure()
plt.scatter(
    df["MURDER"],
    df["RAPE"],
    alpha=0.65
)
plt.title("Relationship Between Murder and Rape Records")
plt.xlabel("Murder Records")
plt.ylabel("Rape Records")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(PLOT_DIR / "scatterplot.png", dpi=300)
plt.close()

# 4. CORRELATION HEATMAP
selected = [
    "MURDER",
    "RAPE",
    "ROBBERY",
    "BURGLARY",
    "THEFT",
    "RIOTS",
    "CHEATING"
]

correlation = df[selected].corr()

plt.figure(figsize=(10, 8))

image = plt.imshow(
    correlation,
    cmap="coolwarm",
    vmin=-1,
    vmax=1
)

plt.colorbar(image, label="Correlation")

plt.xticks(
    range(len(selected)),
    selected,
    rotation=45,
    ha="right"
)

plt.yticks(range(len(selected)), selected)

for i in range(len(selected)):
    for j in range(len(selected)):
        plt.text(
            j, i,
            f"{correlation.iloc[i, j]:.2f}",
            ha="center",
            va="center",
            fontsize=8
        )

plt.title("Crime Category Correlation Matrix")
plt.tight_layout()
plt.savefig(PLOT_DIR / "correlation_matrix.png", dpi=300)
plt.close()

# 5. YEARLY CRIME TREND
yearly = (
    df.groupby("YEAR")["TOTAL IPC CRIMES"]
    .sum()
)

plt.figure()
plt.plot(
    yearly.index,
    yearly.values,
    marker="o",
    linewidth=2
)

plt.title("Year-wise Total IPC Crime Trend")
plt.xlabel("Year")
plt.ylabel("Total IPC Crimes")
plt.xticks(yearly.index)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(PLOT_DIR / "yearly_trend.png", dpi=300)
plt.close()

print("\nALL VISUALIZATIONS GENERATED!")

for file in PLOT_DIR.iterdir():
    if file.suffix == ".png":
        print("Created:", file.name)
