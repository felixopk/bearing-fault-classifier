"""
Train Machine Learning Models for Bearing Fault Classification
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import sys

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score,
    precision_recall_fscore_support
)

import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')
import os 

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(ROOT_DIR, "data", "processed", "features.csv")


def load_and_prepare_data():
    """Load and prepare the dataset"""
    
    print("=" * 70)
    print("  LOADING DATASET")
    print("=" * 70)
    print()
    
    # Load data
    df = pd.read_csv(DATA_PATH)
    print(f"✓ Loaded {len(df)} samples")
    print()
    
    # Separate features and labels
    # Drop label columns
    feature_cols = [col for col in df.columns if col not in ['label', 'label_original', 'fault']]
    
    X = df[feature_cols].values
    y = df['label'].values
    
    print(f"📊 Dataset shape:")
    print(f"  - Features (X): {X.shape}")
    print(f"  - Labels (y): {y.shape}")
    print()
    
    # Class distribution
    unique, counts = np.unique(y, return_counts=True)
    print("🏷️  Class distribution:")
    for label, count in zip(unique, counts):
        print(f"  - {label:15s}: {count:4d} samples ({count/len(y)*100:.1f}%)")
    print()
    return X, y, feature_cols, df['label'].unique()


def split_and_scale_data(X, y, test_size=0.2, random_state=42):
    """Split data and apply feature scaling"""
    
    print("=" * 70)
    print("  DATA PREPROCESSING")
    print("=" * 70)
    print()
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"✓ Split data:")
    print(f"  - Training: {len(X_train)} samples ({(1-test_size)*100:.0f}%)")
    print(f"  - Testing:  {len(X_test)} samples ({test_size*100:.0f}%)")
    print()
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("✓ Applied StandardScaler normalization")
    print()
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def train_random_forest(X_train, y_train):
    """Train Random Forest classifier"""
    
    print("\n" + "─" * 70)
    print("🌲 RANDOM FOREST")
    print("─" * 70)
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    print("Training...")
    model.fit(X_train, y_train)
    print("✓ Training complete")
    
    return model


def train_xgboost(X_train, y_train):
    """Train XGBoost classifier"""
    
    print("\n" + "─" * 70)
    print("🚀 XGBOOST")
    print("─" * 70)
    
    # Encode labels for XGBoost
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1,
        eval_metric='mlogloss'
    )
    
    print("Training...")
    model.fit(X_train, y_train_encoded)
    print("✓ Training complete")
    
    # Store label encoder
    model.label_encoder = le
    
    return model


def train_svm(X_train, y_train):
    """Train SVM classifier"""
    
    print("\n" + "─" * 70)
    print("🎯 SUPPORT VECTOR MACHINE")
    print("─" * 70)
    
    model = SVC(
        kernel='rbf',
        C=10.0,
        gamma='scale',
        random_state=42,
        probability=True
    )
    
    print("Training...")
    model.fit(X_train, y_train)
    print("✓ Training complete")
    
    return model


def train_logistic_regression(X_train, y_train):
    """Train Logistic Regression (baseline)"""
    
    print("\n" + "─" * 70)
    print("📊 LOGISTIC REGRESSION (Baseline)")
    print("─" * 70)
    
    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        n_jobs=-1
    )
    
    print("Training...")
    model.fit(X_train, y_train)
    print("✓ Training complete")
    
    return model


def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate model performance"""
    
    print()
    print("Evaluating...")
    
    # Handle XGBoost encoding
    if hasattr(model, 'label_encoder'):
        y_test_encoded = model.label_encoder.transform(y_test)
        y_pred_encoded = model.predict(X_test)
        y_pred = model.label_encoder.inverse_transform(y_pred_encoded)
    else:
        y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average='weighted', zero_division=0
    )
    
    print()
    print(f"📊 Results:")
    print(f"  - Accuracy:  {accuracy*100:.2f}%")
    print(f"  - Precision: {precision*100:.2f}%")
    print(f"  - Recall:    {recall*100:.2f}%")
    print(f"  - F1-Score:  {f1*100:.2f}%")
    print()
    
    # Classification report
    print("📋 Detailed Classification Report:")
    print()
    print(classification_report(y_test, y_pred, zero_division=0))
    
    # Confusion matrix
    print("🔢 Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    
    # Get unique labels
    labels = sorted(set(y_test) | set(y_pred))
    
    # Print confusion matrix with labels
    print()
    print("         Predicted")
    print("        ", "  ".join(f"{label[:4]:>6s}" for label in labels))
    print("Actual")
    for i, label in enumerate(labels):
        print(f"{label[:10]:10s}", "  ".join(f"{cm[i][j]:6d}" for j in range(len(labels))))
    print()
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'predictions': y_pred
    }


def save_model(model, scaler, model_name, metrics):
    """Save trained model and scaler"""
    
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)
    
    # Save model
    model_path = models_dir / f'{model_name}_model.pkl'
    joblib.dump(model, model_path)
    
    # Save scaler
    scaler_path = models_dir / f'{model_name}_scaler.pkl'
    joblib.dump(scaler, scaler_path)
    
    # Save metrics
    metrics_path = models_dir / f'{model_name}_metrics.txt'
    with open(metrics_path, 'w') as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Accuracy: {metrics['accuracy']*100:.2f}%\n")
        f.write(f"Precision: {metrics['precision']*100:.2f}%\n")
        f.write(f"Recall: {metrics['recall']*100:.2f}%\n")
        f.write(f"F1-Score: {metrics['f1']*100:.2f}%\n")
    
    print(f"✓ Saved model: {model_path}")
    print(f"✓ Saved scaler: {scaler_path}")


def main():
    """Main training pipeline"""
    
    print()
    print("=" * 70)
    print("  BEARING FAULT CLASSIFICATION - MODEL TRAINING")
    print("=" * 70)
    print()
    
    # Load data
    X, y, feature_names, class_names = load_and_prepare_data()
    
    # Split and scale
    X_train, X_test, y_train, y_test, scaler = split_and_scale_data(X, y)
    
    # Train models
    print()
    print("=" * 70)
    print("  TRAINING MODELS")
    print("=" * 70)
    
    models = {}
    results = {}
    
    # 1. Logistic Regression (Baseline)
    models['logistic_regression'] = train_logistic_regression(X_train, y_train)
    results['logistic_regression'] = evaluate_model(
        models['logistic_regression'], X_test, y_test, 'Logistic Regression'
    )
    save_model(models['logistic_regression'], scaler, 'logistic_regression', 
               results['logistic_regression'])
    
    # 2. Random Forest
    models['random_forest'] = train_random_forest(X_train, y_train)
    results['random_forest'] = evaluate_model(
        models['random_forest'], X_test, y_test, 'Random Forest'
    )
    save_model(models['random_forest'], scaler, 'random_forest', 
               results['random_forest'])
    
    # 3. XGBoost
    models['xgboost'] = train_xgboost(X_train, y_train)
    results['xgboost'] = evaluate_model(
        models['xgboost'], X_test, y_test, 'XGBoost'
    )
    save_model(models['xgboost'], scaler, 'xgboost', results['xgboost'])
    
    # 4. SVM
    models['svm'] = train_svm(X_train, y_train)
    results['svm'] = evaluate_model(models['svm'], X_test, y_test, 'SVM')
    save_model(models['svm'], scaler, 'svm', results['svm'])
    
    # Summary
    print()
    print("=" * 70)
    print("  TRAINING COMPLETE - MODEL COMPARISON")
    print("=" * 70)
    print()
    
    print(f"{'Model':<25s} {'Accuracy':>10s} {'Precision':>10s} {'Recall':>10s} {'F1-Score':>10s}")
    print("─" * 70)
    
    for model_name, metrics in results.items():
        print(f"{model_name.replace('_', ' ').title():<25s} "
              f"{metrics['accuracy']*100:>9.2f}% "
              f"{metrics['precision']*100:>9.2f}% "
              f"{metrics['recall']*100:>9.2f}% "
              f"{metrics['f1']*100:>9.2f}%")
    
    # Find best model
    best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
    
    print()
    print(f"🏆 Best Model: {best_model[0].replace('_', ' ').title()}")
    print(f"   Accuracy: {best_model[1]['accuracy']*100:.2f}%")
    print()
    
    print("=" * 70)
    print("  SUCCESS! 🎉")
    print("=" * 70)
    print()
    print("✓ All models trained and saved to models/ directory")
    print()
    print("💡 Next steps:")
    print("  1. Test API: python app/main.py")
    print("  2. Build Docker: docker build -t bearing-classifier .")
    print("  3. Deploy to AWS: (see README.md)")
    print()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)