import flwr as fl
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import sys
import numpy as np

# --- データの前処理部分 ---
JPY_TO_USD_RATE = 145.0
def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)

    # ステップ1: 基本的な前処理
    if 'japan' in file_path:
        df['Annual_Income(USD)'] = df['Annual_Income(JPY)'] / JPY_TO_USD_RATE
        df['Savings'] = df['Saving(JPY)'] / JPY_TO_USD_RATE
    else:
        df.rename(columns={'Saving(USD)': 'Savings'}, inplace=True)
    
    if 'FICO_Score' not in df.columns:
        df['FICO_Score'] = 0

    # ステップ2
    df['income_per_service'] = df['Annual_Income(USD)'] / (df['Years_of_Service'] + 1)
    df['estimated_asset_score'] = df['Annual_Income(USD)'] * df['Years_of_Service']
    df['total_financial_power'] = df['Annual_Income(USD)'] + df['Savings']

    # ステップ3: モデルが使用する特徴量を選択
    features_df = df[[
        'Age', 'Annual_Income(USD)', 'Years_of_Service', 'Loan_Status', 'FICO_Score',
        'income_per_service', 'estimated_asset_score', 'total_financial_power', 'Savings'
    ]]
    target = df['Payment_Delay']
    
    # ステップ4: データを分割
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        features_df, target, test_size=0.2, random_state=42, stratify=target
    )

    # ステップ5: 標準化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_df)
    X_test = scaler.transform(X_test_df)
    
    return X_train, X_test, y_train, y_test

# --- Flowerクライアント部分 ---
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, X_train, y_train, X_test, y_test):
        # より安定した学習のためのモデル設定
        self.model = LogisticRegression(
            solver='saga',      
            max_iter=1000,      
            class_weight='balanced',
            random_state=42
        )
        self.X_train, self.y_train = X_train, y_train
        self.X_test, self.y_test = X_test, y_test

    def get_parameters(self, config):
        if hasattr(self.model, 'coef_'):
            return [self.model.coef_, self.model.intercept_]
        else:
            n_features = self.X_train.shape[1]
            return [np.zeros((1, n_features)), np.array([0.0])]

    def set_parameters(self, parameters):
        if len(parameters[0].shape) > 1:
            self.model.coef_ = parameters[0]
            self.model.intercept_ = parameters[1]

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.fit(self.X_train, self.y_train)
        print(f"Client training on {len(self.X_train)} samples")
        return self.get_parameters(config={}), len(self.X_train), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        if hasattr(self.model, 'coef_'):
            predictions = self.model.predict(self.X_test)
            accuracy = accuracy_score(self.y_test, predictions)
        else:
            accuracy = 0.0
        
        print(f"✅ Client accuracy: {accuracy:.4f}")
        return float(accuracy), len(self.X_test), {"accuracy": float(accuracy)}

# --- プログラムの実行部分 ---
if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ["usa", "japan"]:
        print("Usage: python client.py [usa|japan]")
        sys.exit(1)
    
    country = sys.argv[1]
    print(f"Starting Flower client for {country}...")

    if country == "usa":
        X_train, X_test, y_train, y_test = load_and_preprocess_data('data_usa.csv')
    else:
        X_train, X_test, y_train, y_test = load_and_preprocess_data('data_japan.csv')

    client = FlowerClient(X_train, y_train, X_test, y_test)
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)