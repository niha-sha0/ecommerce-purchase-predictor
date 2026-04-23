# E-Commerce Purchase Intent Predictor

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Live-brightgreen)
![ML](https://img.shields.io/badge/Machine%20Learning-XGBoost-orange)
![Status](https://img.shields.io/badge/Status-Deployed-success)

## Live Demo
🌐 **[https://ecommerce-purchase-predictor.streamlit.app/](https://ecommerce-purchase-predictor.streamlit.app/)**

---

## Problem Statement
In e-commerce, only ~15% of website visitors actually make a purchase. This ML system predicts whether a visitor will buy or not — enabling businesses to show targeted discounts, send push notifications, and increase revenue at the right moment.

---

## Project Overview
An end-to-end machine learning project that predicts online shopper purchase intent using session behaviour data. Trained on 12,330 real e-commerce sessions with 18 features, comparing 7 ML algorithms and deployed as a live interactive web application.

---

## Live App
👉 [https://ecommerce-purchase-predictor.streamlit.app/](https://ecommerce-purchase-predictor.streamlit.app/)

---

## Dataset
| Detail | Info |
|--------|------|
| Source | UCI / Kaggle |
| Link | [Online Shoppers Intention](https://www.kaggle.com/datasets/imakash3011/online-shoppers-purchasing-intention-dataset) |
| Size | 12,330 sessions, 18 features |
| Target | Revenue (True = Purchase, False = No Purchase) |
| Class Split | 84.5% No Purchase — 15.5% Purchase |

---

## Tech Stack
| Category | Tools |
|----------|-------|
| Language | Python 3.11 |
| ML Libraries | Scikit-learn, XGBoost |
| Data Analysis | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Explainability | SHAP |
| Deployment | Streamlit Cloud |
| Model Saving | Joblib |

---

## ML Pipeline

Load Data → EDA → Feature Engineering → Split & Scale → Train 7 Models → Evaluate → SHAP Explainability → Deploy

---

## Algorithms Compared

| Algorithm | Type | Role |
|-----------|------|------|
| Logistic Regression | Linear classifier | Baseline |
| Decision Tree | Rule-based splits | Interpretable |
| Random Forest | Bagging ensemble | Strong model |
| Gradient Boosting | Boosting ensemble | Strong model |
| XGBoost | Optimised boosting | Best model |
| SVM | Margin maximiser | Classic ML |
| KNN | Distance-based | Simple ML |

---

## Feature Engineering
11 new features engineered from original 18:

| Feature | Description |
|---------|-------------|
| `total_pages` | Total pages visited across all categories |
| `total_duration` | Total time spent on site in seconds |
| `bounce_exit_diff` | Difference between bounce and exit rates |
| `is_engaged` | Flag if user spent above-median time on site |
| `value_per_page` | Page value score per product page |
| `product_page_ratio` | Fraction of pages that were product pages |
| `high_value_session` | Flag for top 25% page value sessions |
| `duration_per_page` | Average time spent per page |
| `is_returning` | Flag for returning visitor |
| `info_ratio` | Ratio of informational pages visited |
| `weekend_special` | Interaction of weekend visit and special day |

---

## Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | ~92% |
| AUC-ROC | ~0.94 |
| F1-Score | ~0.74 |
| Cross Validation | 5-Fold Stratified |

---

## Handling Class Imbalance
Used `scale_pos_weight` in XGBoost to handle the 84:16 class imbalance — a production-grade approach used in real industry systems instead of SMOTE.

---

## Explainability (SHAP)
- Global feature importance — which features impact purchase decisions most
- Individual prediction explanation — why the model predicted buy or not buy
- Waterfall plots for single customer explanation

---

## Project Structure

ecommerce-purchase-predictor/
│
├── app.py                            ← Streamlit web application
├── ecommerce_model_pipeline.pkl      ← Trained ML pipeline
├── ecommerce_model_pipeline.ipynb    ← Full Jupyter notebook
├── requirements.txt                  ← Python dependencies
├── runtime.txt                       ← Python version
└── README.md                         ← Project documentation

---

## How to Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/ecommerce-purchase-predictor.git
cd ecommerce-purchase-predictor

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py

# 4. Open browser at
http://localhost:8501
```

---

## Resume Bullet Point
> Built a production-grade e-commerce purchase intent prediction system comparing 7 ML algorithms (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost, SVM, KNN) with automated hyperparameter tuning via GridSearchCV. Engineered 11 domain-specific features, achieved 92% accuracy and 0.94 AUC-ROC, added SHAP explainability for model transparency, and deployed as a live Streamlit web application.

---

## Author
M.Tech Student — Machine Learning Project

## License
This project is open source and available under the MIT License.
