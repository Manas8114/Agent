import requests

BASE_URL = "http://localhost:8000"
TIMEOUT = 30

def test_get_quantum_safe_security_status():
    url = f"{BASE_URL}/api/v1/telecom/quantum-status"
    headers = {
        "Accept": "application/json"
    }
    try:
        response = requests.get(url, headers=headers, timeout=TIMEOUT)
        response.raise_for_status()
    except requests.RequestException as e:
        assert False, f"Request to {url} failed: {e}"

    try:
        data = response.json()
    except ValueError:
        assert False, "Response is not valid JSON"

    # Validate keys presence
    expected_keys = {"security_level", "algorithms", "threat_detection"}
    missing_keys = expected_keys - data.keys()
    assert not missing_keys, f"Response JSON missing keys: {missing_keys}"

    # Validate security_level type and non-empty string
    security_level = data["security_level"]
    assert isinstance(security_level, str) and security_level.strip(), "security_level must be a non-empty string"

    # Validate algorithms is a list (array)
    algorithms = data["algorithms"]
    assert isinstance(algorithms, list), "algorithms must be a list"
    # Optionally, each item in algorithms could be checked to be a string or dict, but schema is generic
    for algo in algorithms:
        assert algo is not None, "Algorithm item must not be None"

    # Validate threat_detection is an object (dict)
    threat_detection = data["threat_detection"]
    assert isinstance(threat_detection, dict), "threat_detection must be an object/dict"

test_get_quantum_safe_security_status()