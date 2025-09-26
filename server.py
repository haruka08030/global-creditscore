import flwr as fl

# 学習を管理するサーバーを起動
# strategy: 学習の進め方の戦略 (FedAvgは最も標準的なもの)
# num_rounds: 学習を何回繰り返すか (今回は3回)
strategy = fl.server.strategy.FedAvg()
fl.server.start_server(
    server_address="0.0.0.0:8080", 
    config=fl.server.ServerConfig(num_rounds=3),
    strategy=strategy
)
print("Federated Learning server started.")