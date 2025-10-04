import requests

BASE_URL = "http://localhost:8000"
TIMEOUT = 30

def test_get_zero_touch_automation_status():
    url = f"{BASE_URL}/api/v1/telecom/zta-status"
    headers = {
        "Accept": "application/json"
    }
    try:
        response = requests.get(url, headers=headers, timeout=TIMEOUT)
        response.raise_for_status()
    except requests.RequestException as e:
        assert False, f"Request failed: {e}"

    try:
        data = response.json()
    except ValueError:
        assert False, "Response is not valid JSON"

    # Validate presence of expected keys
    assert isinstance(data, dict), "Response JSON is not an object"
    assert "status" in data, "'status' field missing in response"
    assert "active_pipelines" in data, "'active_pipelines' field missing in response"
    assert "deployment_metrics" in data, "'deployment_metrics' field missing in response"

    # Validate data types
    assert isinstance(data["status"], str), "'status' must be a string"
    assert isinstance(data["active_pipelines"], list), "'active_pipelines' must be a list"
    assert isinstance(data["deployment_metrics"], dict), "'deployment_metrics' must be an object"

test_get_zero_touch_automation_status()