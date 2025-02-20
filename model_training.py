import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix
import joblib

# Define the dataset folder path
dataset_folder = r"D:\pythonProject4\datasets\boltzmannbrain\nab\versions\1\artificialWithAnomaly\artificialWithAnomaly"
model_folder = r"D:\pythonProject4\models"

# Ensure model folder exists
os.makedirs(model_folder, exist_ok=True)

# List all CSV files in the dataset folder
csv_files = [f for f in os.listdir(dataset_folder) if f.endswith(".csv")]
print(f"Found {len(csv_files)} CSV files.")

# Loop through each CSV file and process it
for file in csv_files:
    file_path = os.path.join(dataset_folder, file)
    print(f"\n🔹 Processing: {file}")

    # Load dataset
    df = pd.read_csv(file_path)

    # Check and drop timestamp column if present
    for col in ["timestamp", "time", "date"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    # Handle missing values
    df.fillna(df.median(), inplace=True)

    print(df.head())  # Check if there are more feature columns

    # Preprocess data
    scaler = StandardScaler()
    X = scaler.fit_transform(df)

    # Split dataset: 80% training, 20% testing
    split_index = int(0.8 * len(X))
    X_train, X_test = X[:split_index], X[split_index:]

    # Train Isolation Forest model
    model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    model.fit(X_train)

    # Generate labels (1 = normal, -1 = anomaly)
    y_test_pred = model.predict(X_test)
    y_test_true = np.ones(len(y_test_pred))  # Assume all are normal initially
    y_test_true[np.random.choice(len(y_test_true), int(0.05 * len(y_test_true)), replace=False)] = -1  # Random anomalies

    # Convert -1/1 labels to binary (1 for anomaly, 0 for normal)
    y_test_pred_binary = (y_test_pred == -1).astype(int)
    y_test_true_binary = (y_test_true == -1).astype(int)

    # Compute evaluation metrics
    accuracy = accuracy_score(y_test_true_binary, y_test_pred_binary)
    precision = precision_score(y_test_true_binary, y_test_pred_binary)
    recall = recall_score(y_test_true_binary, y_test_pred_binary)
    f1 = f1_score(y_test_true_binary, y_test_pred_binary)

    print(f"✅ Model Performance for {file}:")
    print(f"   - Accuracy:  {accuracy:.4f}")
    print(f"   - Precision: {precision:.4f}")
    print(f"   - Recall:    {recall:.4f}")
    print(f"   - F1-Score:  {f1:.4f}")

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test_true_binary, y_test_pred_binary)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {file}")
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(model_folder, f"roc_curve_{file.replace('.csv', '.png')}"))
    plt.close()

    # Confusion Matrix
    cm = confusion_matrix(y_test_true_binary, y_test_pred_binary)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Normal", "Anomaly"], yticklabels=["Normal", "Anomaly"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {file}")
    plt.savefig(os.path.join(model_folder, f"confusion_matrix_{file.replace('.csv', '.png')}"))
    plt.close()

    # Save the model and scaler with a unique name
    model_filename = os.path.join(model_folder, f"anomaly_model_{file.replace('.csv', '.pkl')}")
    scaler_filename = os.path.join(model_folder, f"scaler_{file.replace('.csv', '.pkl')}")

    joblib.dump(model, model_filename)
    joblib.dump(scaler, scaler_filename)

    print(f"✅ Model and Scaler saved for {file}!")
    print(f"📊 ROC Curve and Confusion Matrix saved!\n")

print("🎉 All models and scalers have been saved successfully!")
