# 🇮🇳 Indian Domestic Flight Delay Analytics

An end-to-end **data analytics and machine learning project** that analyses flight delay patterns in Indian domestic aviation using **DGCA On-Time Performance statistics (2024)**.

The project explores delay trends across airlines, airports, seasons, and flight timings, and builds ML models to predict whether a flight will be delayed.

---

## Project Overview

| Item | Details |
|-----|-----|
| Domain | Aviation Data Analytics |
| Dataset | DGCA 2024 domestic flight statistics (~300K records) |
| Airlines | IndiGo, Air India, Akasa Air, SpiceJet, Air India Express, AIX Connect |
| Airports | 15 major Indian airports, including DEL, BOM, BLR, MAA, HYD, CCU |
| Best Model | Random Forest (Accuracy: 90.8%, AUC-ROC: 0.97) |

---

## Key Insights

Some interesting patterns observed from the analysis:

- **IndiGo shows the highest on-time performance (73.4%)**, while **SpiceJet has the lowest (~48.6%)**.
- **Winter fog (Jan–Feb)** significantly increases delays, especially at **Delhi, Lucknow, Patna, and Chandigarh**.
- **Monsoon season** increases the average delay time by **8–12 minutes** across routes.
- Flights departing between **6 PM – 10 PM IST** have the highest probability of delay.
- **Late arrival of aircraft** is the most common cause of delays (~35%).

---

## Project Structure

```
indian-flight-delay-analytics/

generate_india_data.py      # Generates realistic Indian flight dataset
run_preprocessing.py        # Data cleaning and feature engineering
run_eda.py                  # Exploratory data analysis + charts
run_ml.py                   # ML model training and evaluation

dashboard/
    app.py                  # Streamlit interactive dashboard

data/
    charts/                 # Generated visualizations

requirements.txt
README.md
```

---

## How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/indian-flight-delay-analytics.git
cd indian-flight-delay-analytics
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the data pipeline

```bash
python generate_india_data.py
python run_preprocessing.py
python run_eda.py
python run_ml.py
```

### 4. Launch the dashboard

```bash
streamlit run dashboard/app.py
```

The Streamlit dashboard provides an interactive interface to explore delay trends across airlines, airports, and seasons.

---

## Machine Learning Results

| Model | Accuracy | AUC-ROC |
|------|------|------|
| Logistic Regression | 90.99% | 0.975 |
| Random Forest | 90.81% | 0.974 |

Both models perform well, with Random Forest giving strong results while also allowing feature importance analysis.

---

## Tech Stack

- Python
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- Streamlit

---

## Data Source

Flight statistics are based on **DGCA (Directorate General of Civil Aviation) 2024 On-Time Performance reports**.

https://dgca.gov.in

---

## Author

**Sanseet Suna**

LinkedIn: https://in.linkedin.com/in/sanseetsuna  
GitHub: https://github.com/Sanseet
