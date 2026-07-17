import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def main():
    print("=" * 50)
    print("Iris Flower Classifier")
    print("=" * 50)
    
    # Load dataset
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names
    
    print(f"\nDataset: {iris.DESCR[:200]}...")
    print(f"\nFeatures: {feature_names}")
    print(f"Classes: {target_names}")
    print(f"Samples: {len(X)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\nTraining samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    
    # Train model
    print("\nTraining Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    print(f"\nAccuracy: {accuracy:.2%}")
    print("\nConfusion Matrix:")
    print(conf_matrix)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    # Sample predictions
    print("Sample Predictions:")
    print("-" * 50)
    for i in range(5):
        sample = X_test[i].reshape(1, -1)
        prediction = model.predict(sample)[0]
        actual = y_test[i]
        status = "[OK]" if prediction == actual else "[FAIL]"
        print(f"Sample {i+1}: Predicted={target_names[prediction]}, "
              f"Actual={target_names[actual]} {status}")

if __name__ == "__main__":
    main()
