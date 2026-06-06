import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn

def main():
    # Mengarah ke dataset di folder root
    data_path = os.path.join(os.path.dirname(__file__), '..', 'telco_preprocessed.csv')
    df = pd.read_csv(data_path)
    
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Memulai run tanpa DagsHub (lokal untuk CI GitHub Actions)
    with mlflow.start_run() as run:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Menyimpan model ke dalam MLflow dan dijadikan Docker Image
        mlflow.sklearn.log_model(model, "model")
        print(f"Model berhasil dilatih. Run ID: {run.info.run_id}")

if __name__ == "__main__":
    main()