import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

def train_maintenance_model(csv_file_path):
    print("1. Loading dataset...")
    # Load your actual dataset from the data folder
    df = pd.read_csv(csv_file_path)
    
    print("2. Preprocessing and encoding features...")
    # Drop timestamp if present (not a predictive numeric feature)
    if 'timestamp' in df.columns:
        df = df.drop(columns=['timestamp'])
        
    # One-hot encode categorical identifiers and failure types
    categorical_cols = [col for col in ['machine_id', 'failure_type'] if col in df.columns]
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    # Define features (X) and target variable (y)
    # Target: maintenance_required (0 - No, 1 - Yes)
    target = 'maintenance_required'
    if target not in df_encoded.columns:
        raise ValueError(f"Target column '{target}' not found in dataset!")
        
    X = df_encoded.drop(columns=[target])
    y = df_encoded[target]
    
    # Train-test split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Feature scaling for continuous sensor data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("3. Training Random Forest Classifier...")
    rf_model = RandomForestClassifier(n_estimators=150, max_depth=12, random_state=42)
    rf_model.fit(X_train_scaled, y_train)

    print("4. Evaluating model performance...")
    y_pred = rf_model.predict(X_test_scaled)
    y_prob = rf_model.predict_proba(X_test_scaled)[:, 1]
    
    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}")
    
    print("\n--- Confusion Matrix ---")
    print(confusion_matrix(y_test, y_pred))

    print("5. Extracting Feature Importances...")
    importances = rf_model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    print("\nTop 10 Most Important Features contributing to Maintenance Required:")
    print(feature_importance_df.head(10))
    
    # Optional: Plot feature importances
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance_df['Feature'].head(10)[::-1], feature_importance_df['Importance'].head(10)[::-1])
    plt.xlabel("Feature Importance Score")
    plt.ylabel("Features")
    plt.title("Top 10 Sensor & Derived Features in Predictive Maintenance")
    plt.tight_layout()
    plt.savefig('data/feature_importances.png')
    print("\nFeature importance plot saved to 'data/feature_importances.png'")

if __name__ == "__main__":
    # Update with your actual dataset filename in the data/ folder
    data_path = 'data/machine_sensor_data.csv'
    # train_maintenance_model(data_path)
    print("Script template ready! Link your dataset path to execute.")