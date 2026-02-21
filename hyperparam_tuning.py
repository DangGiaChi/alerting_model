import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, make_scorer, f1_score
import joblib
import os
import time
from train import create_sliding_windows, add_statistical_features
import sys

WINDOW_SIZE = 50
HORIZON = 10


def tune_hyperparameters(df, n_iter=20):
    print(f"Randomized Search")
    print(f"Window size: {WINDOW_SIZE}, Horizon: {HORIZON}\n")
    
    X, y = create_sliding_windows(df, WINDOW_SIZE, HORIZON)
    X_enhanced = add_statistical_features(X, WINDOW_SIZE)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_enhanced, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Train samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    param_dist = {
        'n_estimators': [50, 100, 150, 200, 300],
        'max_depth': [5, 10, 15, 20, 25, None],
        'min_samples_split': [5, 10, 20, 30, 40],
        'min_samples_leaf': [2, 5, 10, 15, 20],
        'max_features': ['sqrt', 'log2', None],
        'bootstrap': [True, False]
    }
    
    base_model = RandomForestClassifier(
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    
    f1_scorer = make_scorer(f1_score)
    
    print(f"Testing {n_iter} random combinations")
    search = RandomizedSearchCV(
        base_model,
        param_dist,
        n_iter=n_iter,
        cv=3,
        scoring=f1_scorer,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    start_time = time.time()
    search.fit(X_train_scaled, y_train)
    elapsed_time = time.time() - start_time
    
    print(f"\n=== Tuning Complete ===")
    print(f"Time taken: {elapsed_time:.1f} seconds")
    print(f"\nBest parameters:")
    for param, value in search.best_params_.items():
        print(f"  {param}: {value}")
    
    print(f"\nBest cross-validation F1-score: {search.best_score_:.4f}")
    
    best_model = search.best_estimator_
    train_score = best_model.score(X_train_scaled, y_train)
    test_score = best_model.score(X_test_scaled, y_test)
    
    print(f"\nTrain accuracy: {train_score:.4f}")
    print(f"Test accuracy: {test_score:.4f}")
    
    y_pred = best_model.predict(X_test_scaled)
    print("\nTest set performance:")
    print(classification_report(y_test, y_pred, 
                                target_names=['Normal', 'Incident'],
                                digits=4))
    
    print("\n=== Top 5 Parameter Combinations ===")
    results_df = pd.DataFrame(search.cv_results_)
    top_5 = results_df.nsmallest(5, 'rank_test_score')[
        ['rank_test_score', 'mean_test_score', 'std_test_score', 'params']
    ]
    for idx, row in top_5.iterrows():
        print(f"\nRank {int(row['rank_test_score'])}: F1 = {row['mean_test_score']:.4f} (+/- {row['std_test_score']:.4f})")
        print(f"  Params: {row['params']}")
    
    return best_model, search


def compare_with_baseline():
    print("\n=== Comparison with Baseline ===")
    
    try:
        baseline_config = joblib.load('models/model_config.pkl')
        baseline_model = joblib.load('models/incident_model.pkl')
        print("✓ Baseline model loaded")
        
        print("\nBaseline hyperparameters:")
        print(f"  n_estimators: 100")
        print(f"  max_depth: 10")
        print(f"  min_samples_split: 20")
        print(f"  min_samples_leaf: 10")
        
        if os.path.exists('models/model_config_tuned.pkl'):
            tuned_config = joblib.load('models/model_config_tuned.pkl')
            print("\nTuned hyperparameters:")
            for param, value in tuned_config['best_params'].items():
                print(f"  {param}: {value}")
            print(f"\nCV F1-score: {tuned_config['cv_score']:.4f}")
        
    except FileNotFoundError:
        print("Baseline model not found. Run train_model.py first.")


if __name__ == "__main__":
    try:
        df = pd.read_csv('timeseries_data.csv')
    except FileNotFoundError:
        print("timeseries_data.csv not found. Maybe you forgot to run generate_data.py first?")
        sys.exit(1)
    
    if len(sys.argv) > 1:
        n_iter = int(sys.argv[1])
        save_model = sys.argv[2].lower() == 'y' if len(sys.argv) > 2 else False
    else:
        n_iter = input("Number of iterations (default: 20): ").strip() or "20"
        n_iter = int(n_iter)
        save_model = None
    
    best_model, search = tune_hyperparameters(df, n_iter=n_iter)
    
    compare_with_baseline()
    
    if save_model is None:
        save_choice = input("\n\nSave this tuned model? (y/n): ").lower()
        save_model = (save_choice == 'y')
    
    if save_model:
        os.makedirs('models', exist_ok=True)
        joblib.dump(best_model, 'models/incident_model_tuned.pkl')
        
        scaler = joblib.load('models/scaler.pkl')
        joblib.dump(scaler, 'models/scaler_tuned.pkl')
        
        config = {
            'window_size': WINDOW_SIZE,
            'horizon': HORIZON,
            'n_features': best_model.n_features_in_,
            'best_params': search.best_params_,
            'cv_score': search.best_score_
        }
        joblib.dump(config, 'models/model_config_tuned.pkl')
        
        print("\nTuned model saved:")
        print("- models/incident_model_tuned.pkl")
        print("- models/scaler_tuned.pkl")
        print("- models/model_config_tuned.pkl")
