import requests

def test_get_global_federation_status():
    base_url = "http://localhost:8000"
    url = f"{base_url}/api/v1/telecom/federation"
    headers = {
        "Accept": "application/json"
    }
    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        assert False, f"HTTP request failed: {e}"

    try:
        data = response.json()
    except ValueError:
        assert False, "Response is not valid JSON"

    expected_fields = [
        "total_nodes",
        "active_nodes",
        "updates_shared",
        "aggregations_total",
        "avg_model_accuracy"
    ]

    for field in expected_fields:
        assert field in data, f"Missing field in response: {field}"

    # Validate types
    assert isinstance(data["total_nodes"], int), "total_nodes should be integer"
    assert isinstance(data["active_nodes"], int), "active_nodes should be integer"
    assert isinstance(data["updates_shared"], int), "updates_shared should be integer"
    assert isinstance(data["aggregations_total"], int), "aggregations_total should be integer"
    assert isinstance(data["avg_model_accuracy"], (float, int)), "avg_model_accuracy should be a number"

    # Validate that numeric values are non-negative (assuming that makes sense for these counts/metrics)
    assert data["total_nodes"] >= 0, "total_nodes should be non-negative"
    assert data["active_nodes"] >= 0, "active_nodes should be non-negative"
    assert data["updates_shared"] >= 0, "updates_shared should be non-negative"
    assert data["aggregations_total"] >= 0, "aggregations_total should be non-negative"
    assert 0.0 <= data["avg_model_accuracy"] <= 1.0, "avg_model_accuracy should be between 0 and 1"

test_get_global_federation_status()