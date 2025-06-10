import requests
import json

# APIのベースURL
BASE_URL = "http://localhost:8000"

def test_model_info():
    """モデル情報を取得"""
    response = requests.get(f"{BASE_URL}/model-info")
    print("\n=== Model Info ===")
    print(json.dumps(response.json(), indent=2))

def test_prediction():
    """予測を実行"""
    # テストデータ（California Housingデータセットのサンプル）
    test_data = [
        # サンプル1: 一般的な住宅
        [8.3252, 41.0, 6.984127, 1.023810, 322.0, 2.555556, 37.88, -122.23],
        # サンプル2: 高所得地域の住宅
        [15.0001, 38.0, 7.148438, 1.013422, 226.0, 2.801724, 37.85, -122.25],
        # サンプル3: 低所得地域の住宅
        [3.8462, 52.0, 5.925926, 1.074074, 558.0, 2.675926, 37.87, -122.22]
    ]
    
    for i, data in enumerate(test_data):
        response = requests.post(
            f"{BASE_URL}/predict",
            json=data
        )
        print(f"\n=== Prediction {i+1} ===")
        print(f"Input: {data}")
        print("Output:")
        print(json.dumps(response.json(), indent=2))

if __name__ == "__main__":
    # モデル情報の取得
    test_model_info()
    
    # 予測の実行
    test_prediction() 