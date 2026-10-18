import DataProcess as dp

from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    r2_score,
    root_mean_squared_error,
    mean_absolute_error
)
from sklearn.pipeline import make_pipeline
from xgboost import XGBRegressor
import joblib

# Load datasets
train_data, val_data, test_data = dp.load_data()

# Prepare training and validation datasets
X_train, y_train, X_val, y_val = dp.prepare_datasets(
    train_data, val_data
)


def transform_and_scale(X_train, X_val):
    """
    Scale training and validation features.
    Fit the scaler on training data only.
    """
    scaler = MinMaxScaler()

    X_train_arr = scaler.fit_transform(X_train.to_numpy())
    X_val_arr = scaler.transform(X_val.to_numpy())

    y_train_arr = y_train.to_numpy()
    y_val_arr = y_val.to_numpy()

    return X_train_arr, y_train_arr, X_val_arr, y_val_arr, scaler


def train_model(x_train, y_train, x_val, y_val, degree=3):
    """
    Train Ridge Regression with polynomial features.
    Evaluate it on training and validation datasets.
    """
    model = make_pipeline(
        PolynomialFeatures(degree=degree, include_bias=False),
        Ridge(alpha=1)
    )

    # Train the model
    model.fit(x_train, y_train)

    # Make predictions
    y_train_pred = model.predict(x_train)
    y_val_pred = model.predict(x_val)

    # Calculate evaluation metrics
    results = {
        "Train R2": r2_score(y_train, y_train_pred),
        "Validation R2": r2_score(y_val, y_val_pred),
        "Train RMSE": root_mean_squared_error(y_train, y_train_pred),
        "Validation RMSE": root_mean_squared_error(y_val, y_val_pred),
        "Train MAE": mean_absolute_error(y_train, y_train_pred),
        "Validation MAE": mean_absolute_error(y_val, y_val_pred)
    }

    return model, results


def train_xgboost(x_train, y_train, x_val, y_val):
    """
    Train XGBoost for comparison with Ridge.
    Evaluate it on training and validation datasets.
    """
    model = XGBRegressor(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        tree_method="hist",
        n_jobs=4,
        random_state=42
    )

    # Train the model
    model.fit(x_train, y_train)

    # Make predictions
    y_train_pred = model.predict(x_train)
    y_val_pred = model.predict(x_val)

    # Calculate evaluation metrics
    results = {
        "Train R2": r2_score(y_train, y_train_pred),
        "Validation R2": r2_score(y_val, y_val_pred),
        "Train RMSE": root_mean_squared_error(y_train, y_train_pred),
        "Validation RMSE": root_mean_squared_error(y_val, y_val_pred),
        "Train MAE": mean_absolute_error(y_train, y_train_pred),
        "Validation MAE": mean_absolute_error(y_val, y_val_pred)
    }

    return model, results


def print_results(title, results):
    """
    Print evaluation metrics.
    """
    print(f"\n========== {title} ==========")

    for metric, value in results.items():
        print(f"{metric}: {value:.6f}")


if __name__ == '__main__':

    # --------------------------------------------------
    # 1. Prepare and scale training and validation data
    # --------------------------------------------------

    (
        X_train_arr,
        y_train_arr,
        X_val_arr,
        y_val_arr,
        scaler
    ) = transform_and_scale(X_train, X_val)

    # --------------------------------------------------
    # 2. Train Ridge Regression
    # --------------------------------------------------

    degree = 3

    ridge_model, ridge_results = train_model(
        X_train_arr,
        y_train_arr,
        X_val_arr,
        y_val_arr,
        degree=degree
    )
    # Save the trained model and scaler
    joblib.dump(ridge_model, "ridge_model.joblib")
    joblib.dump(scaler, "scaler.joblib")

    print("Model and scaler saved successfully.")

    print_results(
        f"RIDGE DEGREE {degree} - TRAIN / VALIDATION",
        ridge_results
    )

    # --------------------------------------------------
    # 3. Train XGBoost for comparison
    # --------------------------------------------------

    xgb_model, xgb_results = train_xgboost(
        X_train_arr,
        y_train_arr,
        X_val_arr,
        y_val_arr
    )

    print_results(
        "XGBOOST - TRAIN / VALIDATION",
        xgb_results
    )

    # --------------------------------------------------
    # 4. Prepare the test dataset
    # --------------------------------------------------

    X_test, y_test = dp.optimize_testset(test_data)

    # Match the training feature columns and their order
    X_test = X_test.reindex(
        columns=X_train.columns,
        fill_value=0
    )

    # Use the scaler fitted on training data
    X_test_arr = scaler.transform(X_test.to_numpy())

    # --------------------------------------------------
    # 5. Test Ridge
    # --------------------------------------------------

    y_test_pred_ridge = ridge_model.predict(X_test_arr)

    ridge_test_results = {
        "Test R2": r2_score(y_test, y_test_pred_ridge),
        "Test RMSE": root_mean_squared_error(y_test, y_test_pred_ridge),
        "Test MAE": mean_absolute_error(y_test, y_test_pred_ridge)
    }

    print_results(
        "RIDGE - TEST RESULTS",
        ridge_test_results
    )

    # --------------------------------------------------
    # 6. Test XGBoost
    # --------------------------------------------------

    y_test_pred_xgb = xgb_model.predict(X_test_arr)

    xgb_test_results = {
        "Test R2": r2_score(y_test, y_test_pred_xgb),
        "Test RMSE": root_mean_squared_error(y_test, y_test_pred_xgb),
        "Test MAE": mean_absolute_error(y_test, y_test_pred_xgb)
    }

    print_results(
        "XGBOOST - TEST RESULTS",
        xgb_test_results
    )

    # --------------------------------------------------
    # 7. Compare both models on the test dataset
    # --------------------------------------------------

    print("\n========== TEST SET COMPARISON ==========")

    print(
        f"Ridge Test R2: "
        f"{ridge_test_results['Test R2']:.6f}"
    )
    print(
        f"XGBoost Test R2: "
        f"{xgb_test_results['Test R2']:.6f}"
    )

    print(
        f"Ridge Test RMSE: "
        f"{ridge_test_results['Test RMSE']:.6f}"
    )
    print(
        f"XGBoost Test RMSE: "
        f"{xgb_test_results['Test RMSE']:.6f}"
    )

    print(
        f"Ridge Test MAE: "
        f"{ridge_test_results['Test MAE']:.6f}"
    )
    print(
        f"XGBoost Test MAE: "
        f"{xgb_test_results['Test MAE']:.6f}"
    )