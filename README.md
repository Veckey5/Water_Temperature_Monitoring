<h1 align="center">🌡️ Building Temperature Monitoring</h1>

<p align="center">
  <i>Weekly hot-water compliance checks and boiler anomaly detection with Python, pandas and scikit-learn.</i>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white"/>
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white"/>
  <img src="https://img.shields.io/badge/Matplotlib-11557c?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white"/>
</p>

<p align="center">
  <a href="https://colab.research.google.com/github/Veckey5/building-temperature-monitoring/blob/main/notebooks/01_hot_water_compliance_analysis.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Notebook 01 in Colab"/></a>
  <a href="https://colab.research.google.com/github/Veckey5/building-temperature-monitoring/blob/main/notebooks/02_boiler_anomaly_detection.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Notebook 02 in Colab"/></a>
</p>

---

## 📑 Contents

1. [Overview](#-overview)
2. [Key Results](#-key-results)
3. [Data & Schema](#-data--schema)
4. [Notebook 01: Hot Water Compliance](#-notebook-01-hot-water-compliance)
5. [Notebook 02: Boiler Anomaly Detection](#-notebook-02-boiler-anomaly-detection)
6. [Limitations & Next Steps](#-limitations--next-steps)
7. [Repository Structure](#-repository-structure)
8. [How to Run](#-how-to-run)

---

## 🎯 Overview

Facilities teams need to know **where hot water is not getting hot enough** and **when boiler readings look abnormal**. This project answers both questions from sensor data. It was originally completed as a technical assessment.

| | Notebook | Question | Approach |
|---|---|---|---|
| **01** | [Hot Water Compliance](notebooks/01_hot_water_compliance_analysis.ipynb) | Does each location reach **more than 55 °C at least once every week**? | Weekly grouping, rule check, compliance charts |
| **02** | [Boiler Anomaly Detection](notebooks/02_boiler_anomaly_detection.ipynb) | Can unusual boiler flow temperatures be flagged automatically? | Lag features, One-Class SVM, tested with synthetic anomalies |

---

## 📈 Key Results

<div align="center">

| 📊 **5,919** | 🚫 **0 of 19 weeks** | 🥇 **8 of 19 weeks** | 🌡️ **65.56 °C** |
|:---:|:---:|:---:|:---:|
| readings across 4 locations | FF Staff Room weeks meeting the 55 °C rule | best location (GF Ladies WC) | highest reading recorded |

</div>

---

## 🗂️ Data & Schema

> The raw CSV files were supplied for the assessment and are **not included** in this repository. Place them in `data/` to re-run the notebooks (see [How to Run](#-how-to-run)).

### `hot-water-pipe-temperature.csv` (Notebook 01)

**5,919 rows · 3 columns · no missing values · readings at irregular intervals · weekly data through 2019**

| Column | Type | Description | Example |
|---|---|---|---|
| `Timestamp` | datetime | Date and time the reading was received | `2019-09-18 09:24:00` |
| `Location` | text (categorical) | Sensor location: `FF Doctors Room`, `FF Staff Room`, `GF Ladies WC`, `GF Treatment Room 4` | `FF Doctors Room` |
| `Value` | float | Pipe temperature in °C | `14.75` |

**Summary of `Value` (°C)**

| count | mean | std | min | 25% | 50% | 75% | max |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 5,919 | 22.32 | 7.14 | 14.75 | 18.81 | 20.43 | 22.43 | 65.56 |

### `boiler-flow-temperature.csv` (Notebook 02)

**8,484 rows · 2 columns · readings about every 5 minutes · 1 to 31 December 2018**

| Column | Type | Description | Example |
|---|---|---|---|
| `timestamp` | integer (epoch milliseconds) | Time of the reading; converted with `pd.to_datetime(..., unit='ms')` | `1543622414000` → `2018-12-01 00:00:14` |
| `value` | float | Boiler flow temperature in °C | `35.5` |

> The raw file has a **leading space in both column names** (`" timestamp"`, `" value"`). Notebook 02 strips the names on load, so no workaround is needed.

**Features built for the model (Notebook 02)**

| Feature | Type | Description |
|---|---|---|
| `value` | float | Current reading |
| `value_lag1` … `value_lag5` | float | The previous 5 readings (the "memory" window) |
| `hour_sin`, `hour_cos` | float | Time of day as a point on a 24-hour circle, so 23:55 and 00:05 are close together |

---

## 🧪 Notebook 01: Hot Water Compliance

**Question:** *Does every location reach more than 55 °C at least once each week?*

**Method**

1. Load the data and check for missing values
2. Convert `Timestamp` to datetime and confirm all readings fall in one year
3. Group readings by `Location` and week of year (`%U`, weeks start on Sunday)
4. Mark each location-week `True` if any reading is above 55 °C
5. Check for location-weeks with no readings at all
6. Visualise compliance

### 📸 Snapshots

**Weeks meeting the requirement, by location**

<p align="center">
  <img src="images/compliance_by_location.png" width="620" alt="Stacked bar chart of weeks that met and did not meet the 55C requirement for each location"/>
</p>

**Week-by-week view**

<p align="center">
  <img src="images/weekly_compliance_heatmap.png" width="900" alt="Heatmap of weekly compliance for each location"/>
</p>

### Findings

| Rank | Location | Weeks met | Compliance |
|:---:|---|:---:|:---:|
| 1 | GF Ladies WC | 8 / 19 | 42% |
| 2 | GF Treatment Room 4 | 5 / 19 | 26% |
| 3 | FF Doctors Room | 3 / 19 | 16% |
| 4 | FF Staff Room | 0 / 19 | **0%** |

- **No location met the rule every week.** The best, GF Ladies WC, met it in fewer than half of the weeks.
- **FF Staff Room never reached 55 °C**, so it is the first place to investigate (distance from the boiler, flow rate or sensor placement).
- **Weeks 33, 37 and 40** were the strongest, with three of the four locations meeting the rule. It is worth finding out what was different in those weeks.

---

## 🤖 Notebook 02: Boiler Anomaly Detection

**Question:** *Can we flag unusual boiler flow temperatures automatically?*

The boiler data has **no labels** saying which readings are faults. The notebook therefore teaches a One-Class SVM what "normal" looks like, then **plants synthetic faults** in data the model has never seen to measure how many it catches.

**Pipeline**

```
raw readings → clean → lag + time-of-day features → time-based split
      → train on first 21 days (scaler + One-Class SVM, fitted on training data only)
      → tune on next 5 days (grid search on nu and gamma, scored with synthetic faults)
      → final test on last 5 days vs Isolation Forest and a rolling z-score rule
```

| Design choice | Detail |
|---|---|
| **Split** | Train: first 21 days · Validation: next 5 days · Test: the rest, always in time order |
| **No leakage** | `StandardScaler` and the SVM sit in one `Pipeline`, fitted on training data only |
| **Synthetic faults** | Spikes, dips and 6-reading level shifts sized at 2 to 4 × the normal spread; different random faults in validation and test |
| **Tuning** | `nu` ∈ {0.01, 0.02, 0.05, 0.10} × `gamma` ∈ {0.01, 0.05, 0.10, 0.50}, chosen by validation F1 |
| **Baselines** | Isolation Forest and a rolling z-score (±3 standard deviations from the previous 12 readings) |
| **Metrics** | Precision, recall, F1, ROC-AUC, average precision, and event recall (share of faults with at least one flagged reading) |

### 📸 Snapshots

> These images are created when you run Notebook 02 (they are saved to `images/`). Commit them to the repo so they appear here.

<p align="center">
  <img src="images/boiler_series_overview.png" width="800" alt="Boiler flow temperature over time and average by hour of day"/>
</p>

<p align="center">
  <img src="images/boiler_anomaly_detections.png" width="900" alt="One-Class SVM detections of injected faults on the test period"/>
</p>

### Results

> **Fill this table in after running Notebook 02 on the real data**, using the "Test-period results" printed in section 8. Then delete this note.

| Method | Precision | Recall | F1 | ROC-AUC | Event recall |
|---|:---:|:---:|:---:|:---:|:---:|
| One-Class SVM | – | – | – | – | – |
| Isolation Forest | – | – | – | – | – |
| Rolling z-score | – | – | – | – | – |

**Chosen settings:** `nu = –`, `gamma = –`

---

## 🔧 Limitations & Next Steps

| Area | Current state | Next step |
|---|---|---|
| **Labels** | Performance is measured on **synthetic** faults, because the data has no real labels | Get a maintenance or fault log and test on real events |
| **Fault realism** | Spikes, dips and level shifts only | Add slow drift and stuck-sensor faults; try smaller fault sizes |
| **History** | One month of boiler data, so seasons are not captured | Use a longer history and retrain periodically |
| **Training data** | May already contain real faults | Remove known fault periods before training |
| **Features** | Current value, 5 lags and time of day | Add rolling mean/standard deviation and rate of change |
| **Notebook 01 weeks** | Week number ignores the year (the notebook checks only one year is present); a location-week with no readings counts as "not met" | Group by year and week, and report "no data" separately |

---

## 📁 Repository Structure

```
building-temperature-monitoring/
├── README.md
├── requirements.txt
├── notebooks/
│   ├── 01_hot_water_compliance_analysis.ipynb
│   └── 02_boiler_anomaly_detection.ipynb
├── images/
│   ├── compliance_by_location.png
│   ├── weekly_compliance_heatmap.png
│   ├── boiler_series_overview.png          # created by notebook 02
│   └── boiler_anomaly_detections.png       # created by notebook 02
└── data/                                   # not committed: add the CSVs here
    ├── hot-water-pipe-temperature.csv
    └── boiler-flow-temperature.csv
```

---

## ▶️ How to Run

```bash
git clone https://github.com/Veckey5/building-temperature-monitoring.git
cd building-temperature-monitoring
pip install -r requirements.txt
jupyter notebook notebooks/
```

Run the notebooks **in order** with *Run All*. Each one finds its CSV automatically in `data/`, or in the current folder.
On **Google Colab**, click a badge above, upload the CSV file when needed, and run all cells.

---

<p align="center">
  <b>Victoria Akinladejo</b> · Business Intelligence Analyst<br>
  <a href="https://www.linkedin.com/in/victoriaakinladejo">LinkedIn</a> ·
  <a href="mailto:akinladejovictoria.a@gmail.com">Email</a> ·
  <a href="https://github.com/Veckey5">GitHub</a>
</p>
