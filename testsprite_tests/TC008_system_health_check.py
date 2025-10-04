import requests

BASE_URL = "http://localhost:8000"
TIMEOUT = 30
HEADERS = {
    "Accept": "application/json"
}

def test_system_health_check():
    url = f"{BASE_URL}/api/v1/health"
    try:
        response = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        response.raise_for_status()
    except requests.RequestException as e:
        assert False, f"Request to {url} failed: {e}"

    data = response.json()

    # Validate required fields in response
    assert isinstance(data, dict), "Response is not a JSON object"
    assert "status" in data, "Missing 'status' field in response"
    assert data["status"] in ("healthy", "degraded", "unhealthy", ""), "'status' field has unexpected value"
    assert "timestamp" in data, "Missing 'timestamp' field in response"
    assert isinstance(data["timestamp"], str) and data["timestamp"], "'timestamp' must be a non-empty string"
    assert "version" in data, "Missing 'version' field in response"
    assert isinstance(data["version"], str) and data["version"], "'version' must be a non-empty string"
    assert "components" in data, "Missing 'components' field in response"
    assert isinstance(data["components"], dict), "'components' field must be an object"

test_system_health_check()