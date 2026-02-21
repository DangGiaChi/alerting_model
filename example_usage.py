import numpy as np
import pandas as pd
import joblib
import sys
from sklearn.model_selection import train_test_split
from train import add_statistical_features, create_sliding_windows

def load_model():
    try:
        model = joblib.load('models/incident_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        config = joblib.load('models/model_config.pkl')
    except FileNotFoundError:
        print("Model files not found. Maybe you forgot to run train.py first?")
        sys.exit(1)
    return model, scaler, config


def get_test_set_indices(df, config):
    window_size = config['window_size']
    horizon = config['horizon']
    
    X, y = create_sliding_windows(df, window_size, horizon)
    indices = np.arange(window_size, len(df) - horizon)
    _, _, _, _, train_indices, test_indices = train_test_split(
        X, y, indices, test_size=0.2, random_state=42, stratify=y
    )
    
    return test_indices


def predict_incident(model, scaler, config, recent_data):
    window_size = config['window_size']
    if len(recent_data) < window_size:
        raise ValueError(f"Need at least {window_size} data points")
    window = recent_data.tail(window_size)
    
    metric_cols = ['cpu_usage', 'memory_usage', 'request_rate']
    features = window[metric_cols].values.flatten()
    
    features_enhanced = add_statistical_features(
        features.reshape(1, -1), 
        window_size
    )
    
    features_scaled = scaler.transform(features_enhanced)
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0, 1]
    
    return prediction, probability


def main():
    print("Loading trained model...\n")
    model, scaler, config = load_model()
    
    try:
        df = pd.read_csv('timeseries_data.csv')
    except FileNotFoundError:
        print("timeseries_data.csv not found. Maybe you forgot to run generate_data.py first?")
        sys.exit(1)
    
    test_indices = get_test_set_indices(df, config)
    print(f"Using only test set data ({len(test_indices)} windows from unseen data)\n")
    
    print(f"Model configuration:")
    print(f"  Window size (W): {config['window_size']} time steps")
    print(f"  Prediction horizon (H): {config['horizon']} time steps")
    print(f"\nThe model predicts if an incident will occur in the next {config['horizon']} steps")
    print(f"based on the previous {config['window_size']} steps of metrics.\n")
    
    print("=" * 65)
    print("Example 1: Normal period (no incident expected), threshold 0.5")
    print("=" * 65)
    
    normal_idx = test_indices[len(test_indices) // 3]
    normal_window = df.iloc[normal_idx - config['window_size']:normal_idx]
    pred, prob = predict_incident(model, scaler, config, normal_window)
    
    print(f"Prediction: {'INCIDENT PREDICTED' if pred == 1 else 'Normal'}")
    print(f"Probability: {prob:.4f}")
    print(f"Risk level: {'HIGH' if prob > 0.7 else 'MEDIUM' if prob > 0.5 else 'LOW'}")
    
    print("\n" + "=" * 65)
    print("Example 2: High-risk period (from test set), threshold 0.5")
    print("=" * 65)
    
    best_prob = 0
    best_idx = test_indices[0]
    
    for idx in test_indices[::50]:
        test_window = df.iloc[idx - config['window_size']:idx]
        _, temp_prob = predict_incident(model, scaler, config, test_window)
        if temp_prob > best_prob:
            best_prob = temp_prob
            best_idx = idx
    
    high_risk_window = df.iloc[best_idx - config['window_size']:best_idx]
    pred, prob = predict_incident(model, scaler, config, high_risk_window)
    
    future_slice = df.iloc[best_idx:best_idx + config['horizon']]
    actual_incident = future_slice['incident'].sum() > 0
    
    print(f"Prediction: {'INCIDENT PREDICTED' if pred == 1 else 'Normal'}")
    print(f"Probability: {prob:.4f}")
    print(f"Risk level: {'HIGH' if prob > 0.7 else 'MEDIUM' if prob > 0.5 else 'LOW'}")
    print(f"Actual outcome: {'Incident occurred' if actual_incident else 'No incident'}")
    
    print("\n" + "=" * 65)
    print("Example 3: Effect of different alert thresholds")
    print("=" * 65)
    
    thresholds = [0.3, 0.5, 0.7, 0.9]
    print(f"Current probability: {prob:.4f}\n")
    
    for threshold in thresholds:
        would_alert = prob >= threshold
        print(f"Threshold {threshold:.1f}: {'ALERT' if would_alert else 'No alert'}")


if __name__ == "__main__":
    main()
