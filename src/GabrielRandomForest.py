import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

# =====================================================================
# CONFIGURATION & HYPERPARAMETERS (Easily tweak these settings anytime)
# =====================================================================
CONFIG = {
    "data_path": "data/ProjectData/proj_machine_readings_cleaned.csv",
    "target_column": "maintenance_required",
    "test_size": 0.2,            # Proportion of data used for testing (e.g., 0.2 = 20%)
    "n_estimators": 150,         # Number of trees in the forest
    "max_depth": 12,             # Maximum depth of each tree (None for unlimited)
    "random_state": 42           # Ensures reproducibility
}

def run_pipeline(config):
    print("1. Loading dataset...")
    try:
        df = pd.read_csv(config["data_path"])
    except FileNotFoundError:
        print(f"Error: Could not find file at '{config['data_path']}'. Please check your data folder path.")
        return

    print("2. Preprocessing and encoding features...")
    # Drop timestamp if present (not a numeric predictor)
    if 'timestamp' in df.columns:
        df = df.drop(columns=['timestamp'])
        
    # Dynamically one-hot encode any categorical columns present (like machine_id, failure_type)
    categorical_cols = [col for col in ['machine_id', 'failure_type'] if col in df.columns]
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    target = config["target_column"]
    if target not in df_encoded.columns:
        raise ValueError(f"Target column '{target}' not found in dataset columns: {list(df_encoded.columns)}")
        
    X = df_encoded.drop(columns=[target])
    y = df_encoded[target]
    
    # Train-test split using config parameters
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=config["test_size"], 
        random_state=config["random_state"], 
        stratify=y
    )
    
    # Feature scaling for continuous sensor data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"3. Training Random Forest Classifier (n_estimators={config['n_estimators']}, max_depth={config['max_depth']})...")
    rf_model = RandomForestClassifier(
        n_estimators=config["n_estimators"],
        max_depth=config["max_depth"],
        random_state=config["random_state"]
    )
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
    
    # Save feature importance chart
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance_df['Feature'].head(10)[::-1], feature_importance_df['Importance'].head(10)[::-1])
    plt.xlabel("Feature Importance Score")
    plt.ylabel("Features")
    plt.title(f"Top Sensor & Derived Features (n_estimators={config['n_estimators']})")
    plt.tight_layout()
    plt.savefig('data/feature_importances.png')
    print("\nFeature importance plot successfully saved to 'data/feature_importances.png'")

if __name__ == "__main__":
    run_pipeline(CONFIG)