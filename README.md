# The_Rossman_Project

#### This project was made by Bruno Vinicius.

# 1. Business Problem.
The CEO would like a solution with Machine Learning in order to have easy access to the revenue of the next 6 weeks.

Solution Idea:

A Telegram bot were the CEO could suply the store number and receive back the predictions of the sales.

# 2. Solution Strategy

My strategy to solve this challenge was:

**Step 00. Imports and Helper Functions:**

**Step 01. Data Colection:**

**Step 02. Data Cleaning:**

**Step 03. Hypothesis Map:**

**Step 04. Feature Engineering:**

**Step 05. EDA(Exploratory Data Analysis):**

**Step 06. Data Preparation:**

**Step 07. Feature Selection:**

**Step 08. Machine Learning:**

**Step 09. Hyperparameter Tunning:**

**Step 10. Convert Model Performance to Business Values:**

**Step 11. Deploy Model to Production:**

# 4. Top 3 Data Insights

**Hypothesis 01:**
Stores with more assortment should sell more.
**False.**

**Hypothesis 02:**
Stores with competitors near to them sell any less.
**False.**

**Hypothesis 03:**
Stores with more promotions actives should sell more.
**False.**

# 5. Machine Learning Model Applied
In this project was applied the models.
    
    -  Average Model (as baseline model)
    -  Linear Regression
    -  Lasso
    -  Random Forest
    -  Xgboost
    -  Lightgbm

# 6. Machine Learning Modelo Performance
The chosen model was the Lightgbm due to its performance and size.

    - RMSE:1297,29 +/- 183,9
    - MAPE:0,13+/-0,01
    - MAE: 908,11+/-129,11

# 7. Business Results

    - Total sum of predictions R$ 280,123,396.28
    - Total sum of worst scenario R$ 279,191,646.07
    - Total sum of best scenario R$ 281,055,146.48
