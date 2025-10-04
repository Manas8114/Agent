import requests

BASE_URL = "http://localhost:8000"
TIMEOUT = 30

def test_get_real_time_data_for_all_components():
    url = f"{BASE_URL}/api/v1/real-data"
    headers = {"Accept": "application/json"}

    try:
        response = requests.get(url, headers=headers, timeout=TIMEOUT)
        response.raise_for_status()
    except requests.RequestException as e:
        assert False, f"Request to {url} failed: {e}"

    json_data = response.json()

    # Validate the presence of required top-level keys
    for key in ["health", "kpis", "federation", "selfEvolution"]:
        assert key in json_data, f"Missing key '{key}' in response"

    # Validate types of each component
    assert isinstance(json_data["health"], dict), "'health' should be an object"
    assert isinstance(json_data["kpis"], dict), "'kpis' should be an object"
    assert isinstance(json_data["federation"], dict), "'federation' should be an object"
    assert isinstance(json_data["selfEvolution"], dict), "'selfEvolution' should be an object"

    # Basic sanity checks on content presence
    # Health object should have at least one key
    assert len(json_data["health"]) > 0, "'health' object should not be empty"
    # KPIs object should have at least one key
    assert len(json_data["kpis"]) > 0, "'kpis' object should not be empty"
    # Federation object should have at least one key
    assert len(json_data["federation"]) > 0, "'federation' object should not be empty"
    # SelfEvolution object should have at least one key
    assert len(json_data["selfEvolution"]) > 0, "'selfEvolution' object should not be empty"

test_get_real_time_data_for_all_components()