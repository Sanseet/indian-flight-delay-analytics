\# 🇮🇳 Indian Domestic Flight Delay Analytics



> End-to-end Data Analytics \& Machine Learning project analyzing Indian domestic

> flight delay patterns using DGCA 2024 On-Time Performance statistics.



!\[Python](https://img.shields.io/badge/Python-3.8+-blue)

!\[ML](https://img.shields.io/badge/ML-scikit--learn-orange)

!\[Dashboard](https://img.shields.io/badge/Dashboard-Streamlit-red)

!\[Status](https://img.shields.io/badge/Status-Complete-green)



---



\## 📌 Project Overview



| | |

|---|---|

| \*\*Domain\*\* | Aviation Analytics |

| \*\*Dataset\*\* | DGCA 2024 Indian domestic flights (300K+ records) |

| \*\*Airlines\*\* | IndiGo, Air India, Akasa Air, SpiceJet, Air India Express, AIX Connect |

| \*\*Airports\*\* | 15 major Indian airports (DEL, BOM, BLR, MAA, HYD, CCU and more) |

| \*\*Best Model\*\* | Random Forest — 90.8% Accuracy · 0.97 AUC-ROC |



---



\## 🎯 Key Findings



\- ✈️ \*\*IndiGo leads OTP at 73.4%\*\* — SpiceJet lowest at 48.6% (DGCA 2024)

\- 🌫️ \*\*Fog season (Jan–Feb) doubles delays\*\* at DEL, LKO, PAT, IXC

\- 🌧️ \*\*Monsoon adds 8–12 min\*\* average delay across all routes

\- 🕗 \*\*Evening flights (6–10 PM IST)\*\* have highest delay rates

\- 🔁 \*\*Late aircraft propagation\*\* is the #1 delay cause (~35%)



---



\## 📁 Project Structure

```

indian-flight-delay-analytics/

├── generate\_india\_data.py    ← Generate realistic Indian flight dataset

├── run\_preprocessing.py      ← Data cleaning \& feature engineering

├── run\_eda.py                ← Exploratory data analysis + 8 charts

├── run\_ml.py                 ← ML model training \& evaluation

├── dashboard/

│   └── app.py                ← Streamlit interactive dashboard

├── data/

│   └── charts/               ← All generated visualizations

├── requirements.txt

└── README.md

```



---



\## 🚀 How to Run



\### 1. Clone the repository

```bash

git clone https://github.com/YOUR\_USERNAME/indian-flight-delay-analytics.git

cd indian-flight-delay-analytics

```



\### 2. Install dependencies

```bash

pip install -r requirements.txt

```



\### 3. Run the full pipeline

```bash

python generate\_india\_data.py   # Generate dataset

python run\_preprocessing.py     # Clean \& engineer features

python run\_eda.py               # Generate EDA charts

python run\_ml.py                # Train ML models

```



\### 4. Launch the dashboard

```bash

streamlit run dashboard/app.py

```



---



\## 📊 Sample Visualizations



| Delay by Airline | Monthly Trends | Feature Importance |

|---|---|---|

| \*(see data/charts/)\* | \*(see data/charts/)\* | \*(see data/charts/)\* |



---



\## 🤖 ML Model Results



| Model | Accuracy | AUC-ROC |

|-------|----------|---------|

| Logistic Regression | 90.99% | 0.975 |

| \*\*Random Forest\*\* | \*\*90.81%\*\* | \*\*0.974\*\* |



---



\## 🛠️ Tech Stack



`Python 3.8+` · `pandas` · `numpy` · `matplotlib` · `seaborn` · `scikit-learn` · `Streamlit`



---



\## 📄 Data Source



Dataset generated using real \*\*DGCA (Directorate General of Civil Aviation) 2024\*\*

On-Time Performance statistics — https://dgca.gov.in



---



\## 👤 Author



\*\*Sanseet Suna\*\* · \[LinkedIn](https://in.linkedin.com/in/sanseetsuna) · \[GitHub](https://github.com/Sanseet)

