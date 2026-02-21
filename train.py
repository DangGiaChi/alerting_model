import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os
import sys

WINDOW_SIZE = 50
HORIZON = 10


def create_sliding_windows(df, window_size=50, horizon=10):
    metric_cols = ['cpu_usage', 'memory_usage', 'request_rate']
    data = df[metric_cols].values
    incidents = df['incident'].values
    
    X = []
    y = []
    
    for i in range(window_size, len(data) - horizon):
        window_features = data[i - window_size:i].flatten()
        X.append(window_features)
        
        future_incidents = incidents[i:i + horizon]
        label = 1 if future_incidents.sum() > 0 else 0
        y.append(label)
    
    return np.array(X), np.array(y)


def add_statistical_features(X, window_size, n_metrics=3):
    n_samples = X.shape[0]
    X_reshaped = X.reshape(n_samples, window_size, n_metrics)
    
    features = []
    
    for i in range(n_metrics):
        metric_data = X_reshaped[:, :, i]
        
        features.append(np.mean(metric_data, axis=1))
        features.append(np.std(metric_data, axis=1))
        features.append(np.min(metric_data, axis=1))
        features.append(np.max(metric_data, axis=1))
        
        features.append(metric_data[:, -1] - metric_data[:, 0])
        
        recent_window = min(10, window_size)
        features.append(np.mean(metric_data[:, -recent_window:], axis=1))
    
    extra_features = np.column_stack(features)
    X_enhanced = np.hstack([X, extra_features])
    
    return X_enhanced


def load_optimal_hyperparams():
    if os.path.exists('models/model_config_tuned.pkl'):
        tuned_config = joblib.load('models/model_config_tuned.pkl')
        print("Found tuned hyperparameters from previous optimization")
        print("Using optimal parameters:")
        for param, value in tuned_config['best_params'].items():
            print(f"    {param}: {value}")
        return tuned_config['best_params']
    else:
        print("Using default hyperparameters")
        print("  (Run tune_hyperparameters.py to optimize)")
        return {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 20,
            'min_samples_leaf': 10,
            'max_features': None,
            'bootstrap': True
        }


def train_model(df, window_size=50, horizon=10):
    print(f"\n=== Training Configuration ===")
    print(f"Window size: {window_size} time steps")
    print(f"Prediction horizon: {horizon} time steps")
    print(f"Problem: Predict if incident occurs in next {horizon} steps")
    
    X, y = create_sliding_windows(df, window_size, horizon)
    
    print(f"Total windows created: {len(X)}")
    print(f"Positive samples (incident): {y.sum()} ({100*y.mean():.2f}%)")
    print(f"Negative samples (normal): {len(y) - y.sum()} ({100*(1-y.mean()):.2f}%)")
    
    X_enhanced = add_statistical_features(X, window_size)
    print(f"Feature dimensions: {X_enhanced.shape}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_enhanced, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTrain set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    params = load_optimal_hyperparams()
    
    print("\nTraining Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=params['n_estimators'],
        max_depth=params['max_depth'],
        min_samples_split=params['min_samples_split'],
        min_samples_leaf=params['min_samples_leaf'],
        max_features=params.get('max_features'),
        bootstrap=params.get('bootstrap', True),
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    
    model.fit(X_train_scaled, y_train)
    
    train_acc = model.score(X_train_scaled, y_train)
    test_acc = model.score(X_test_scaled, y_test)
    
    print(f"\nTraining accuracy: {train_acc:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")
    
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/incident_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    
    config = {
        'window_size': window_size,
        'horizon': horizon,
        'n_features': X_enhanced.shape[1]
    }
    joblib.dump(config, 'models/model_config.pkl')
    
    print("\n=== Model saved ===")
    print("- models/incident_model.pkl")
    print("- models/scaler.pkl")
    print("- models/model_config.pkl")
    
    return model, scaler, X_test_scaled, y_test


if __name__ == "__main__":
    try:
        df = pd.read_csv('timeseries_data.csv')
    except FileNotFoundError:
        print("timeseries_data.csv not found. Maybe you forgot to run generate_data.py first?")
        sys.exit(1)
    
    model, scaler, X_test, y_test = train_model(
        df, 
        window_size=WINDOW_SIZE, 
        horizon=HORIZON
    )
    
    print("\nTraining complete!")
