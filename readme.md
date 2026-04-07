# 🏠 California Housing Price Prediction

A supervised machine learning project by **Ayesha Javaid** — predicting California house prices using Multiple Linear Regression and other regression models, with a focus on building clean, production-style **Scikit-Learn data pipelines**.

---

## 📌 Project Overview

| Detail | Info |
|---|---|
| **Task** | Regression (Predicting house prices) |
| **Dataset** | California Housing Prices (Kaggle) |
| **Models Trained** | Linear Regression, KNN, Decision Tree, Random Forest, Gradient Boosting |
| **Best Model** | Random Forest (R² = 0.81) |
| **Key Skill** | Sklearn Pipelines + ColumnTransformer |
| **Platform** | Google Colab / Jupyter Notebook |

---

## 🗂️ Repository Structure

```
california-housing-price-prediction/
│
├── notebooks/
│   └── MultipleLinearRegression_CreatingDataPipelines.ipynb
│
├── data/
│   └── housing.csv                  # California Housing dataset
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## 🧠 Problem Statement

A real estate agent needs a machine learning model to predict house prices based on features like location, number of rooms, income level, and ocean proximity.

**Target variable:** `median_house_value` (continuous) → Regression problem

---

## 📊 Dataset Features

| Feature | Description |
|---|---|
| `longitude` | How far west the house is |
| `latitude` | How far north the house is |
| `housing_median_age` | Median age of houses in the block |
| `total_rooms` | Total rooms in the block |
| `total_bedrooms` | Total bedrooms in the block |
| `population` | Total population in the block |
| `households` | Total households in the block |
| `median_income` | Median income (in tens of thousands USD) |
| `ocean_proximity` | Categorical — distance to ocean |
| `median_house_value` | **Target** — median house price (USD) |

Dataset source: [Kaggle - California Housing Prices](https://www.kaggle.com/camnugent/california-housing-prices)

---

## ⚙️ ML Workflow

```
1. Problem Formulation
      ↓
2. Load & Explore Data (20,640 rows × 10 features)
      ↓
3. Exploratory Data Analysis
   - Missing value check
   - Correlation heatmap
   - Geographical scatter plot (lat/lon vs price)
   - Ocean proximity distribution
      ↓
4. Data Preprocessing Pipelines
   - Numerical pipeline: SimpleImputer → StandardScaler
   - Categorical pipeline: OneHotEncoder
   - Combined with ColumnTransformer
      ↓
5. Train-Test Split (80/20)
      ↓
6. Train 5 Models
   - Linear Regression
   - KNN Regressor (with GridSearchCV)
   - Decision Tree Regressor
   - Random Forest Regressor
   - Gradient Boosting Regressor
      ↓
7. Evaluate & Compare (MSE + R² Score)
```

---

## 🔑 Key Highlight — Sklearn Pipelines

One of the main focuses of this project is building **reusable preprocessing pipelines** — a best practice in real ML systems.

```python
# Numerical pipeline
numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
    ('scaler', StandardScaler())
])

# Categorical pipeline
categorical_pipeline = Pipeline([
    ('encoder', OneHotEncoder())
])

# Combined pipeline
final_pipe = ColumnTransformer([
    ('num', numerical_pipeline, numerical_features),
    ('cat', categorical_pipeline, categorical_features)
])
```

This ensures the **same transformations** are applied consistently to both training and test data — preventing data leakage.

---

## 📈 Model Results

| Model | MSE | R² Score |
|---|---|---|
| Linear Regression | 4.81 | 0.64 |
| KNN Regressor | 3.72 | 0.72 |
| Decision Tree | — | — |
| **Random Forest** | **2.52** | **0.81** |
| Gradient Boosting | 3.18 | 0.76 |

**🏆 Random Forest** achieved the best performance with the lowest MSE and highest R² score of **0.81**, meaning it explains 81% of the variance in house prices.

---

## 🚀 How to Run

### Option 1 — Google Colab (Recommended)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/california-housing-price-prediction/blob/main/notebooks/MultipleLinearRegression_CreatingDataPipelines.ipynb)

> Upload `housing.csv` to your Colab session or place it in the same directory.

### Option 2 — Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/california-housing-price-prediction.git
cd california-housing-price-prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch Jupyter
jupyter notebook notebooks/MultipleLinearRegression_CreatingDataPipelines.ipynb
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| [Scikit-Learn](https://scikit-learn.org/) | ML models + Pipelines |
| [Pandas](https://pandas.pydata.org/) | Data manipulation |
| [NumPy](https://numpy.org/) | Numerical operations |
| [Matplotlib](https://matplotlib.org/) | Plotting |
| [Seaborn](https://seaborn.pydata.org/) | Statistical visualization |

---

## 📚 Key Learnings

1. How to build **Sklearn Pipelines** for clean, reusable preprocessing
2. Using **ColumnTransformer** to handle numerical and categorical features separately
3. **Hyperparameter tuning** with GridSearchCV on KNN
4. Comparing multiple regression models using MSE and R² Score
5. Interpreting geographical data with scatter plots

---

## 👩‍💻 Author

**Ayesha Javaid**
Machine Learning Portfolio Project

---

## 📄 License

This project is for educational and portfolio purposes.
