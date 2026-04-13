# 🌤️ WeatherLens — Weather Pattern Classification Dashboard

A **Python + Streamlit** web application for automated analysis and classification of weather data.  
Upload your weather dataset (CSV/Excel) and instantly get a complete analysis pipeline.

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the app
streamlit run app.py
```

---

## 📋 Features

| Tab | Description |
|-----|-------------|
| **Preview** | Dataset preview, shape, memory, and download |
| **Overview** | Column types, unique values, target detection, class distribution |
| **Quality** | Missing values heatmap, duplicates, IQR outliers, high cardinality |
| **Preprocessing** | sklearn Pipeline: SimpleImputer → StandardScaler → OneHotEncoder → VarianceThreshold → SelectKBest |
| **EDA** | Univariate histograms, bivariate scatter plots, categorical distributions |
| **Visualizations** | Temperature, humidity, rainfall, pressure/wind, seasonal trend charts |
| **Statistics** | Mean, median, std, variance, skewness, kurtosis, normality tests (Shapiro-Wilk), grouped stats |
| **PCA** | Explained variance, scree plot, 2D/3D scatter, loadings heatmap |
| **LDA** | Weather type class separation (LD1 vs LD2), explained variance, 3D view |
| **Factor Analysis** | Scree plot, factor loadings heatmap, communalities, variance explained (varimax rotation) |
| **Clustering** | KMeans elbow curve, silhouette score, cluster visualizations, profiles, vs true labels |
| **Insights** | Auto-generated climate observations, warnings, seasonal trends, modelling recommendations |

---

## 🧰 Tech Stack

- **Frontend:** Streamlit (hospital-style light dashboard)
- **Visualizations:** Plotly (interactive charts)
- **ML Pipeline:** scikit-learn — `Pipeline`, `ColumnTransformer`, `SimpleImputer`, `StandardScaler`, `OneHotEncoder`, `PCA`, `LinearDiscriminantAnalysis`, `FactorAnalysis`, `KMeans`
- **Statistics:** scipy (Shapiro-Wilk normality tests)
- **Data:** pandas, numpy

---

## 📁 Project Structure

```
DAVL_exam/
├── app.py                    # Main Streamlit app
├── requirements.txt          # Python dependencies
├── README.md
└── utils/
    ├── __init__.py
    ├── chart_style.py        # Shared chart palette & layout helpers
    ├── data_loader.py        # CSV/Excel loading, metadata, target detection
    ├── overview.py           # Column info, dtype distribution, class balance
    ├── quality.py            # Missing values, duplicates, outliers, cardinality
    ├── preprocessing.py      # sklearn Pipeline + ColumnTransformer
    ├── eda.py                # Univariate, bivariate, correlation analysis
    ├── visualization.py      # Weather-specific charts & seasonal trends
    ├── stats.py              # Descriptive statistics, normality tests
    ├── pca_analysis.py       # PCA dimensionality reduction
    ├── lda_analysis.py       # Linear Discriminant Analysis
    ├── factor_analysis.py    # Factor Analysis (sklearn FactorAnalysis)
    ├── clustering.py         # KMeans clustering with elbow & silhouette
    └── insights.py           # Auto-generated climate insights
```

---

## 🌦️ Supported Weather Columns (auto-detected)

The app automatically detects columns matching:
- **Temperature:** `temp`, `temperature`
- **Humidity:** `humidity`, `humid`
- **Rainfall:** `rain`, `rainfall`, `precipitation`
- **Pressure:** `pressure`, `baro`
- **Wind:** `wind`
- **Season/Date:** `season`, `month`, `date`, `year`
- **Target/Class:** `weather`, `type`, `condition`, `label`, `class`, `season`

---

## 📊 Problem Context

| Field | Value |
|-------|-------|
| **Title** | Weather Pattern Classification |
| **Domain** | Environmental / Climate Analytics |
| **Objective** | Detect seasonal trends & identify weather patterns |
| **Dataset** | Weather Dataset (CSV/Excel) |

---

## 📤 Outputs & Downloads

- ✅ Processed/cleaned dataset (CSV)
- ✅ PCA loadings matrix (CSV)
- ✅ LDA transformed data (CSV)
- ✅ Factor loadings table (CSV)
- ✅ KMeans clustered dataset (CSV)
- ✅ Statistical summary (CSV)
- ✅ Auto-insights export (TXT)
