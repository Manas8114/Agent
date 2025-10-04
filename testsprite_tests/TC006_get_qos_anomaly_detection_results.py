import requests

BASE_URL = "http://localhost:8000"
TIMEOUT = 30
HEADERS = {
    "Accept": "application/json"
}

def test_get_qos_anomaly_detection_results():
    url = f"{BASE_URL}/api/v1/agents/qos-anomaly"
    try:
        response = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        response.raise_for_status()
    except requests.RequestException as e:
        assert False, f"Request failed: {e}"

    try:
        data = response.json()
    except ValueError:
        assert False, "Response is not valid JSON"

    assert isinstance(data, dict), "Response JSON is not an object"

    # Validate presence and types of keys
    assert "anomalies" in data, "'anomalies' key missing in response"
    assert isinstance(data["anomalies"], list), "'anomalies' is not an array"

    assert "confidence" in data, "'confidence' key missing in response"
    assert isinstance(data["confidence"], (float, int)), "'confidence' is not a number"

    assert "timestamp" in data, "'timestamp' key missing in response"
    assert isinstance(data["timestamp"], str), "'timestamp' is not a string"

    # Additional sanity checks (timestamps non-empty and confidence score in range 0..1)
    assert data["timestamp"], "'timestamp' is empty"
    assert 0.0 <= data["confidence"] <= 1.0, "'confidence' score is out of expected range 0.0 to 1.0"

test_get_qos_anomaly_detection_results()