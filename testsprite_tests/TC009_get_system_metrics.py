import requests

BASE_URL = "http://localhost:8000"
TIMEOUT = 30

def test_get_system_metrics():
    url = f"{BASE_URL}/api/v1/metrics"
    headers = {
        "Accept": "application/json"
    }
    try:
        response = requests.get(url, headers=headers, timeout=TIMEOUT)
        response.raise_for_status()
        data = response.json()

        # Validate presence of keys
        assert "cpu_usage" in data, "cpu_usage key missing in response"
        assert "memory_usage" in data, "memory_usage key missing in response"
        assert "network_latency" in data, "network_latency key missing in response"
        assert "ai_model_accuracy" in data, "ai_model_accuracy key missing in response"

        # Validate types
        assert isinstance(data["cpu_usage"], (int, float)), "cpu_usage is not a number"
        assert isinstance(data["memory_usage"], (int, float)), "memory_usage is not a number"
        assert isinstance(data["network_latency"], (int, float)), "network_latency is not a number"
        assert isinstance(data["ai_model_accuracy"], (int, float)), "ai_model_accuracy is not a number"

        # Validate values range (optional, generally usage and latency >=0, accuracy 0-1 or 0-100)
        assert data["cpu_usage"] >= 0, "cpu_usage is negative"
        assert data["memory_usage"] >= 0, "memory_usage is negative"
        assert data["network_latency"] >= 0, "network_latency is negative"
        assert 0 <= data["ai_model_accuracy"] <= 100, "ai_model_accuracy out of expected range 0-100"

    except requests.exceptions.RequestException as e:
        assert False, f"Request failed: {e}"

test_get_system_metrics()