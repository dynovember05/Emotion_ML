
import requests
import json
import random

def test_connection():
    url = "http://127.0.0.1:8000"
    print(f"Testing connection to {url}...")
    try:
        response = requests.get(url)
        print(f"Root endpoint status: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Connection failed: {e}")
        return

    # Test prediction with dummy data
    predict_url = f"{url}/predict/landmarks"
    # Create dummy landmarks (1434 floats)
    dummy_landmarks = [random.random() for _ in range(1434)]
    
    payload = {"landmarks": dummy_landmarks}
    
    print(f"\nTesting prediction endpoint: {predict_url}")
    try:
        response = requests.post(predict_url, json=payload)
        print(f"Prediction status: {response.status_code}")
        print(f"Prediction result: {response.json()}")
    except Exception as e:
        print(f"Prediction failed: {e}")

if __name__ == "__main__":
    test_connection()
