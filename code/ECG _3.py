# 📦 Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, f1_score, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.model_selection import train_test_split
from flaml import AutoML

# 📌 1. Load 5000 Rows from the Dataset
file_path = "модуль 3 - датасет - практика.csv"  # Replace with your actual CSV filename
df = pd.read_csv(file_path, nrows=5000)

# 📌 2. Select Relevant Columns
columns = ['Count_subj', 'rr_interval', 'p_end', 'qrs_onset', 'qrs_end',
           'p_axis', 'qrs_axis', 't_axis', 'Healthy_Status']
df = df[columns]

# 📌 3. Clean & Prepare Data
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna()
df = df.astype(float)

# Split features and target
X = df.drop('Healthy_Status', axis=1)
y = df['Healthy_Status']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 📌 4. Run FLAML AutoML Classifier
automl = AutoML()
automl.fit(X_train=X_train, y_train=y_train, task="classification", time_budget=60)

# 📌 5. Make Predictions
y_pred = automl.predict(X_test)
y_prob = automl.predict_proba(X_test)[:, 1]  # For ROC curve

# 📌 6. Metrics and Evaluation
# F1-score
print("🔎 F1 Score:", f1_score(y_test, y_pred))

# Classification report
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.show()

# 📌 7. ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], 'k--', alpha=0.6)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 📌 8. Feature Importance (if supported by final model)
model = automl.model.estimator
if hasattr(model, "feature_importances_"):
    importances = model.feature_importances_
    feat_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    plt.figure(figsize=(8, 4))
    sns.barplot(data=feat_df, x="Importance", y="Feature", palette="mako")
    plt.title("Feature Importances")
    plt.tight_layout()
    plt.show()
