# Determinants and Consequences of Early Marriage in Ethiopia

### Evidence from DHS Data (2000–2016)

---

## 📌 Overview

This project provides a data-driven analysis of early marriage in Ethiopia using nationally representative Demographic and Health Survey (DHS) data from four survey rounds: **2000, 2005, 2011, and 2016**.

The study investigates:

* Trends in early marriage over time
* Socioeconomic and demographic determinants
* Consequences on fertility, education, and employment

This repository is designed as a **reproducible data science and research project**, following professional standards in data analysis and research reporting.

---

## 🎯 Research Objectives

1. Analyze temporal trends in early marriage prevalence
2. Identify key determinants (education, wealth, region, residence, etc.)
3. Examine consequences on:

   * Fertility
   * Educational attainment
   * Employment status
4. Provide evidence-based insights for policy and research

---

## 📊 Data Source

Data comes from the **Ethiopia Demographic and Health Surveys (EDHS)**:

* 2000 EDHS (ETIR41FL)
* 2005 EDHS (ETIR51FL)
* 2011 EDHS (ETIR61FL)
* 2016 EDHS (ETIR71FL)

> Access requires registration at the DHS Program website.

---

## 🧠 Key Variable

**Early Marriage (Target Variable)**

* Defined as: marriage before age 18
* Constructed from DHS variable `v511` (age at first marriage)

---

## 📁 Project Structure

```
ethiopia-early-marriage-dhs/
│
├── data/
│   ├── raw/                  # Original DHS datasets (not included)
│   └── processed/            # Cleaned and merged dataset
│
── notebooks/
│   ├── 01_data_loading.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_exploratory_analysis.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_comprehensive_analysis.ipynb    # ← MAIN ANALYSIS
│   └── 06_advanced_analysis.ipynb         # ← ADVANCED METHODS
│
├── src/
│   ├── config.py
│   ├── data_loader.py
│   └── data_cleaner.py
    └── model.py
    └──visualization.py
    └──wealth_proxy.py
│   └── feature_engineering.py
│
├── reports/
│   ├── figures/              # Visualizations (PNG)
│   ├── tables/               # Statistical tables
│   └── research_paper.pdf
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Methodology

### 1. Data Processing

* Merged DHS datasets across survey years
* Selected relevant variables
* Handled missing values
* Created derived features (education, wealth, media exposure, empowerment)

### 2. Exploratory Data Analysis (EDA)

* Trend analysis over time
* Group comparisons (education, wealth, region, residence)
* Distribution analysis
* Correlation analysis

### 3. Statistical Analysis

* Logistic regression (determinants of early marriage)
* Poisson regression (fertility outcomes)
* Linear regression (education outcomes)

---

## 📈 Outputs

All outputs are generated from the notebooks and saved in:

* `reports/figures/` → visualizations
* `reports/tables/` → statistical tables

Examples include:

* Trend of early marriage over time
* Early marriage by education and wealth
* Regional disparities
* Regression results

---

## 🚀 How to Run

### 1. Clone Repository

```bash
git clone https://github.com/nebekisa/ethiopia-early-marriage-dhs.git
cd ethiopia-early-marriage-dhs
```

### 2. Setup Environment

```bash
python -m venv venv
venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

### 3. Run Analysis

```bash
jupyter notebook
```

Run notebooks in order:
1 → 2 → 3 → 4 → 5

---

## 📌 Notes

* DHS data is not included due to licensing restrictions
* Users must download data from the DHS Program
* Results depend on correct data access and preprocessing

---

## 📄 Research Output

The final research paper is available in:

```
reports/research_paper.pdf
```

---

## 📚 Citation (Draft)

```
Author. (2026). Determinants and Consequences of Early Marriage in Ethiopia: 
Evidence from DHS Data (2000–2016).
```

---

## 📬 Contact

For questions or collaboration:

* GitHub Issues
* Email: libekiger@gmail.com

---


