import requests
import uuid
from datetime import datetime, timezone

BASE_URL = "http://localhost:8000"
TIMEOUT = 30
HEADERS = {"Content-Type": "application/json"}


def test_ingest_new_data():
    url = f"{BASE_URL}/api/v1/data/ingest"
    data_type = "telecom_metric"
    data_object = {
        "signal_strength": -70,
        "cell_id": "Cell_12345",
        "user_count": 150,
        "throughput": 120.5
    }
    timestamp = datetime.now(timezone.utc).isoformat()

    payload = {
        "data_type": data_type,
        "data": data_object,
        "timestamp": timestamp
    }

    try:
        response = requests.post(url, headers=HEADERS, json=payload, timeout=TIMEOUT)
        assert response.status_code == 200, f"Expected 200 but got {response.status_code}. Response: {response.text}"
    except requests.RequestException as e:
        assert False, f"Request failed: {e}"


test_ingest_new_data()