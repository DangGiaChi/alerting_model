import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)
import joblib
import os
from train import create_sliding_windows, add_statistical_features
from sklearn.model_selection import train_test_split


def evaluate_model(model, scaler, X_test, y_test): 
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Incident'], digits=4))
    
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print("                Predicted")
    print("              Normal  Incident")
    print(f"Actual Normal   {cm[0,0]:5d}    {cm[0,1]:5d}")
    print(f"       Incident {cm[1,0]:5d}    {cm[1,1]:5d}")
    
    tn, fp, fn, tp = cm.ravel()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    print(f"\n=== Key Metrics ===")
    print(f"Precision: {precision:.4f} (of alerts, how many are real incidents)")
    print(f"Recall: {recall:.4f} (of incidents, how many we catch)")
    print(f"F1-Score: {f1:.4f}")
    print(f"False Positive Rate: {false_positive_rate:.4f}")
    
    return y_pred, y_proba


def analyze_thresholds(y_test, y_proba):
    print("\n=== Threshold Analysis ===")
    print("Adjusting the threshold changes the trade-off between false alarms and missed incidents.\n")
    
    thresholds = [0.3, 0.5, 0.7, 0.9]
    results = []
    
    for threshold in thresholds:
        y_pred_thresh = (y_proba >= threshold).astype(int)
        cm = confusion_matrix(y_test, y_pred_thresh)
        tn, fp, fn, tp = cm.ravel()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        results.append({
            'threshold': threshold,
            'precision': precision,
            'recall': recall,
            'fpr': fpr,
            'alerts': tp + fp
        })
        
        print(f"Threshold = {threshold:.1f}:")
        print(f"  Precision: {precision:.4f}  Recall: {recall:.4f}  FPR: {fpr:.4f}  Alerts: {tp + fp}")
    
    print("\nRecommendation: Lower threshold (0.3-0.5) for safety-critical systems")
    print("                Higher threshold (0.7-0.9) to reduce false alarms")
    
    return results


def plot_evaluation_curves(y_test, y_proba, save_path='visualizations/evaluation_curves.png'):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    fpr, tpr, thresholds_roc = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    
    axes[0].plot(fpr, tpr, linewidth=2, label=f'ROC (AUC = {roc_auc:.3f})')
    axes[0].plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    
    examined_thresholds = [0.3, 0.5, 0.7, 0.9]
    threshold_fprs = []
    threshold_tprs = []
    
    for threshold in examined_thresholds:
        y_pred_thresh = (y_proba >= threshold).astype(int)
        cm = confusion_matrix(y_test, y_pred_thresh)
        tn, fp, fn, tp = cm.ravel()
        
        threshold_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        threshold_tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        threshold_fprs.append(threshold_fpr)
        threshold_tprs.append(threshold_tpr)
    
    axes[0].scatter(threshold_fprs, threshold_tprs, c='red', s=100, zorder=5,
                    label='Thresholds: 0.3, 0.5, 0.7, 0.9', marker='o', edgecolors='darkred', linewidths=1.5)
    
    for i, threshold in enumerate(examined_thresholds):
        axes[0].annotate(f'{threshold:.1f}',
                        xy=(threshold_fprs[i], threshold_tprs[i]),
                        xytext=(-20, 5), textcoords='offset points',
                        fontsize=9, color='darkred', weight='bold')
    
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate (Recall)')
    axes[0].set_title('ROC Curve')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    precision, recall, thresholds_pr = precision_recall_curve(y_test, y_proba)
    avg_precision = average_precision_score(y_test, y_proba)
    
    axes[1].plot(recall, precision, linewidth=2, label=f'PR (AP = {avg_precision:.3f})')
    
    threshold_precisions = []
    threshold_recalls = []
    
    for threshold in examined_thresholds:
        y_pred_thresh = (y_proba >= threshold).astype(int)
        cm = confusion_matrix(y_test, y_pred_thresh)
        tn, fp, fn, tp = cm.ravel()
        
        threshold_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        threshold_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        threshold_precisions.append(threshold_precision)
        threshold_recalls.append(threshold_recall)
    
    axes[1].scatter(threshold_recalls, threshold_precisions, c='red', s=100, zorder=5,
                    label='Thresholds: 0.3, 0.5, 0.7, 0.9', marker='o', edgecolors='darkred', linewidths=1.5)
    
    for i, threshold in enumerate(examined_thresholds):
        axes[1].annotate(f'{threshold:.1f}',
                        xy=(threshold_recalls[i], threshold_precisions[i]),
                        xytext=(10, 5), textcoords='offset points',
                        fontsize=9, color='darkred', weight='bold')
    
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].set_title('Precision-Recall Curve')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\nEvaluation curves saved to {save_path}")
    plt.close()
    
    print(f"\nROC AUC: {roc_auc:.4f}")
    print(f"Average Precision: {avg_precision:.4f}")


def analyze_feature_importance(model, window_size):
    print("\n=== Feature Importance ===")
    
    importances = model.feature_importances_
    
    top_indices = np.argsort(importances)[-10:][::-1]
    
    print("Top 10 most important features:")
    
    metric_names = ['cpu_usage', 'memory_usage', 'request_rate']
    n_metrics = len(metric_names)
    n_time_features = window_size * n_metrics
    
    for rank, idx in enumerate(top_indices, 1):
        if idx < n_time_features:
            metric_idx = idx % n_metrics
            time_step = idx // n_metrics
            feature_name = f"{metric_names[metric_idx]}_t-{window_size - time_step}"
        else:
            stat_idx = idx - n_time_features
            stat_names = ['mean', 'std', 'min', 'max', 'trend', 'recent_avg']
            stat_name = stat_names[stat_idx % len(stat_names)]
            metric_idx = stat_idx // len(stat_names)
            feature_name = f"{metric_names[metric_idx]}_{stat_name}"
        
        print(f"{rank:2d}. {feature_name:30s} importance: {importances[idx]:.4f}")


def main():
    model = joblib.load('models/incident_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    config = joblib.load('models/model_config.pkl')
    
    print(f"Window size: {config['window_size']}")
    print(f"Prediction horizon: {config['horizon']}")
    
    df = pd.read_csv('timeseries_data.csv')
    
    X, y = create_sliding_windows(df, config['window_size'], config['horizon'])
    X_enhanced = add_statistical_features(X, config['window_size'])
    
    _, X_test, _, y_test = train_test_split(
        X_enhanced, y, test_size=0.2, random_state=42, stratify=y
    )
    
    X_test_scaled = scaler.transform(X_test)
    
    y_pred, y_proba = evaluate_model(model, scaler, X_test_scaled, y_test)
    
    threshold_results = analyze_thresholds(y_test, y_proba)
    
    plot_evaluation_curves(y_test, y_proba)
    
    analyze_feature_importance(model, config['window_size'])


if __name__ == "__main__":
    main()
