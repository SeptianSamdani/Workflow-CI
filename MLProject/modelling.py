import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, log_loss, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load data
train = pd.read_csv('engine_fault_preprocessing/train.csv')
test = pd.read_csv('engine_fault_preprocessing/test.csv')

X_train = train.drop('Engine_Condition', axis=1)
y_train = train['Engine_Condition']
X_test = test.drop('Engine_Condition', axis=1)
y_test = test['Engine_Condition']

mlflow.set_experiment("Engine Fault Detection CI")

os.makedirs('artifacts', exist_ok=True)

def save_confusion_matrix(y_true, y_pred, path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Minor Fault', 'Critical Fault'],
                yticklabels=['Normal', 'Minor Fault', 'Critical Fault'])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def save_feature_importance(model, feature_names, path):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(importances)), importances[indices], color='steelblue')
    plt.xticks(range(len(importances)),
               [feature_names[i] for i in indices],
               rotation=45, ha='right')
    plt.title('Feature Importances')
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

# Tidak pakai mlflow.start_run() — MLflow Project sudah buat run otomatis
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight='balanced'
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)

acc       = accuracy_score(y_test, y_pred)
f1        = f1_score(y_test, y_pred, average='weighted', zero_division=0)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall    = recall_score(y_test, y_pred, average='weighted', zero_division=0)
logloss   = log_loss(y_test, y_prob)

mlflow.log_param("n_estimators", 100)
mlflow.log_param("random_state", 42)
mlflow.log_param("class_weight", "balanced")
mlflow.log_metric("accuracy", acc)
mlflow.log_metric("f1_score_weighted", f1)
mlflow.log_metric("precision_weighted", precision)
mlflow.log_metric("recall_weighted", recall)
mlflow.log_metric("log_loss", logloss)

cm_path = 'artifacts/confusion_matrix.png'
fi_path = 'artifacts/feature_importance.png'
cr_path = 'artifacts/classification_report.txt'

save_confusion_matrix(y_test, y_pred, cm_path)
save_feature_importance(model, X_train.columns.tolist(), fi_path)
with open(cr_path, 'w') as f:
    f.write(classification_report(y_test, y_pred,
            target_names=['Normal', 'Minor Fault', 'Critical Fault'],
            zero_division=0))

mlflow.log_artifact(cm_path)
mlflow.log_artifact(fi_path)
mlflow.log_artifact(cr_path)
mlflow.sklearn.log_model(model, artifact_path="model")

print(f"Accuracy  : {acc:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"Log Loss  : {logloss:.4f}")