# NYC Taxi Trip Duration Prediction

My first Machine Learning project, focused on predicting taxi trip durations in New York City using regression models.

## Project Overview

The goal of this project is to predict the duration of NYC taxi trips based on trip-related features such as pickup and dropoff coordinates, pickup time, passenger count, and trip distance.

The project covers data exploration, preprocessing, feature engineering, model training, and evaluation.

## Exploratory Data Analysis (EDA)

- Analyzed the distribution of trip duration.
- Explored relationships between trip duration and numerical features.
- Investigated outliers and data quality.
- Examined the relationship between trip distance and trip duration.

## Feature Engineering

- Extracted pickup hour, weekday, and month from pickup datetime.
- Created a weekend indicator.
- Calculated geographical distance using the Haversine formula.
- Encoded categorical features.
- Applied `log1p` transformation to the target variable.

## Models

- **Ridge Regression** with polynomial features (degree 3) — final required model.
- **XGBoost Regressor** — used as a comparison model.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- Joblib

## Results

The models were evaluated using:

- R² Score
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)

XGBoost achieved better predictive performance than the Ridge Regression model in this project.

## Project Structure

```text
NYC-Taxi-Trip-Duration-Prediction/
├── DataProcess.py
├── TrainModel.py
├── requirements.txt
├── ridge_model.joblib
├── scaler.joblib
├── README.md
└── ...
```

The structure may vary depending on the files included in the repository.

## How to Run

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd NYC-Taxi-Trip-Duration-Prediction
   ```

2. Install the required libraries:

   ```bash
   pip install -r requirements.txt
   ```

3. Make sure the required dataset files are available in the expected locations.

4. Run the training script:

   ```bash
   python TrainModel.py
   ```

## Learning Objectives

This project helped me practice the fundamentals of Machine Learning, including exploratory data analysis, feature engineering, regression, model evaluation, and comparing different models.

## Author

Mazen Magdy

---

This is my first Machine Learning project and part of my journey toward becoming an AI Engineer.
