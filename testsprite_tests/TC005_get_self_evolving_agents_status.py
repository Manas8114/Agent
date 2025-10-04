import requests

def test_get_self_evolving_agents_status():
    base_url = "http://localhost:8000"
    url = f"{base_url}/api/v1/telecom/self-evolution"
    headers = {
        "Accept": "application/json"
    }
    timeout = 30
    try:
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException as e:
        assert False, f"Request failed: {e}"

    try:
        data = response.json()
    except ValueError:
        assert False, "Response is not valid JSON"

    # Validate required fields presence and types
    expected_fields = {
        "agent_id": str,
        "evolution_round": int,
        "architecture_improvement": (int, float),
        "performance_improvement": (int, float),
        "evolution_status": str
    }
    for field, expected_type in expected_fields.items():
        assert field in data, f"Missing field '{field}' in response"
        assert isinstance(data[field], expected_type), f"Field '{field}' expected type {expected_type}, got {type(data[field])}"

    # Additional non-empty validations
    assert data["agent_id"].strip() != "", "agent_id should not be empty"
    assert data["evolution_round"] >= 0, "evolution_round should be non-negative"
    assert -1.0 <= data["architecture_improvement"] <= 1.0, "architecture_improvement should be between -1.0 and 1.0"
    assert -1.0 <= data["performance_improvement"] <= 1.0, "performance_improvement should be between -1.0 and 1.0"
    assert data["evolution_status"].strip() != "", "evolution_status should not be empty"

test_get_self_evolving_agents_status()