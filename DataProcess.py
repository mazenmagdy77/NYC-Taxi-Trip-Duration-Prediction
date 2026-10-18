import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# Loading our data
def load_data():
    train_data = pd.read_csv('cleaned_train.csv')
    val_data = pd.read_csv('val.csv')
    test_data = pd.read_csv('test.csv')
    return train_data, val_data, test_data


# Clean data
def clean_data(data):
    data = data.copy()

    # Remove trips with extreme duration
    if 'trip_duration' in data.columns:
        data = data[data['trip_duration'] < 10000]

    # Keep pickup coordinates within the selected NYC ranges
    valid_pickup_coordinates = (
        data['pickup_latitude'].between(40, 42)
        & data['pickup_longitude'].between(-75, -72)
    )

    data = data[valid_pickup_coordinates]

    return data.reset_index(drop=True)


# Calculate Haversine distance
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371

    lat1, lon1, lat2, lon2 = np.radians(
        [lat1, lon1, lat2, lon2]
    )

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    c = 2 * np.arctan2(
        np.sqrt(a),
        np.sqrt(1 - a)
    )

    return R * c


# Feature Engineering
def feature_engineering(data):
    data = data.copy()

    data['pickup_datetime'] = pd.to_datetime(
        data['pickup_datetime']
    )

    # Temporarily disabled for testing
    data['pickup_hour'] = data['pickup_datetime'].dt.hour
    data['pickup_weekday'] = data['pickup_datetime'].dt.weekday
    data['pickup_month'] = data['pickup_datetime'].dt.month
    data['is_weekend'] = (
    data['pickup_datetime'].dt.weekday >= 5 ).astype(int)

    # Calculate distance
    data['distance_km'] = haversine_distance(
        data['pickup_latitude'],
        data['pickup_longitude'],
        data['dropoff_latitude'],
        data['dropoff_longitude']
    )

    # One-Hot Encoding
    data = pd.get_dummies(
        data,
        columns=['store_and_fwd_flag'],
        dtype=int
    )

    return data


# Extract features and target
def extract(data):
    X = data.drop(
        columns=['trip_duration', 'id', 'pickup_datetime'],
        errors='ignore'
    )

    if 'trip_duration' in data.columns:
        y = np.log1p(data['trip_duration'])
    else:
        y = None

    return X, y


# Prepare training and validation datasets
def prepare_datasets(train_data, val_data):

    # Train data is already cleaned
    train_data = feature_engineering(train_data)

    # Clean validation data
    val_data = clean_data(val_data)
    val_data = feature_engineering(val_data)

    # Extract features and target
    X_train, y_train = extract(train_data)
    X_val, y_val = extract(val_data)

    # Ensure identical feature columns and order
    X_val = X_val.reindex(
        columns=X_train.columns,
        fill_value=0
    )

    return X_train, y_train, X_val, y_val


# Prepare test dataset when needed for final evaluation
def optimize_testset(test_data):
    test_data = clean_data(test_data)
    test_data = feature_engineering(test_data)

    X_test, y_test = extract(test_data)

    test_data.to_csv('prepared_test.csv', index=False)

    return X_test, y_test


if __name__ == '__main__':
    train_data, val_data, test_data = load_data()

    X_train, y_train, X_val, y_val = prepare_datasets(
        train_data,
        val_data
    )

    print("Train shape:", X_train.shape)
    print("Validation shape:", X_val.shape)

