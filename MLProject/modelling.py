import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn

def main():
    mlflow.set_experiment("Diabetes_CI_Automated")
    
    # 1. Ambil path relatif ke folder dataset yang satu level dengan script ini
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(BASE_DIR, "diabetes_clean_manual.csv")
    
    print(f"[INFO] Membaca dataset lokal dari: {data_path}")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"[ERROR] Dataset tidak ditemukan di path: {data_path}")
        
    df = pd.read_csv(data_path)
    
    # 2. Split Fitur dan Target
    X = df.drop(columns=['Outcome'])
    y = df['Outcome']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. Proses Retraining di dalam Lingkungan Isolasi MLflow Project / Docker
    with mlflow.start_run(run_name="CI_Retrain_Local_Dataset"):
        model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        model.fit(X_train, y_train)
        
        # Simpan model sebagai komponen artefak wajib
        mlflow.sklearn.log_model(model, "model")
        print("[INFO] CI Retraining dengan dataset lokal sukses dijalankan!")

if __name__ == "__main__":
    main()