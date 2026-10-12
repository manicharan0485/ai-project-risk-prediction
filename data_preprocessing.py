import pandas as pd
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
import joblib

def load_data(train_path, test_path):
    print("Loading data...")
    train_data = pd.read_csv(train_path, dtype={'Timestamp': str})
    test_data = pd.read_csv(test_path, dtype={'Timestamp': str})
    print("Data loaded.")
    return train_data, test_data

def preprocess_data(train_data, test_data):
    # Select numeric columns
    print("Preprocessing data...")
    numeric_columns = train_data.select_dtypes(include='number').columns
    
    # Fill missing values using SimpleImputer
    print("Filling missing values...")
    imputer = SimpleImputer(strategy='mean')
    train_data[numeric_columns] = imputer.fit_transform(train_data[numeric_columns])
    test_data[numeric_columns] = imputer.transform(test_data[numeric_columns])
    print("Missing values filled.")
    
    # Save the imputer for future use
    joblib.dump(imputer, 'imputer.pkl')
    
    # Feature Engineering
    print("Creating polynomial features...")
    polynomial_features = PolynomialFeatures(degree=2, include_bias=False)
    X_train_poly = polynomial_features.fit_transform(train_data.drop(columns=['Timestamp', 'SCMstability_category']))
    X_test_poly = polynomial_features.transform(test_data.drop(columns=['Timestamp', 'SCMstability_category']))
    print("Polynomial features created.")

    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_poly)
    X_test_scaled = scaler.transform(X_test_poly)
    print("Features scaled.")
    
    # Remap the class labels
    print("Mapping class labels...")
    y_train = train_data['SCMstability_category'].map({1: 0, 3: 1, 4: 2})
    y_test = test_data['SCMstability_category'].map({1: 0, 3: 1, 4: 2})

    # Check for any unmapped values and handle them
    if y_train.isnull().any():
        print("Warning: Unmapped values found in y_train. Filling with the most frequent class.")
        if not y_train.mode().empty:
            y_train = y_train.fillna(y_train.mode()[0]).astype(int)
        else:
            y_train = y_train.fillna(0).astype(int)
    else:
        y_train = y_train.astype(int)

    if y_test.isnull().any():
        print("Warning: Unmapped values found in y_test. Filling with the most frequent class.")
        if not y_test.mode().empty:
            y_test = y_test.fillna(y_test.mode()[0]).astype(int)
        else:
            y_test = y_test.fillna(0).astype(int)
    
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(polynomial_features, 'polynomial_features.pkl')
    
    print("Preprocessing completed.")
    return X_train_scaled, X_test_scaled, y_train, y_test

if __name__ == "__main__":
    train_data, test_data = load_data('dataset/SCRM_timeSeries_2018_train.csv', 'dataset/SCRM_timeSeries_2018_test.csv')
    X_train, X_test, y_train, y_test = preprocess_data(train_data, test_data)
    joblib.dump((X_train, X_test, y_train, y_test), 'preprocessed_data.pkl')
    print("Preprocessed data saved.")
