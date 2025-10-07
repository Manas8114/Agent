import requests
import uuid
import time

BASE_URL = "http://localhost:8000"
TIMEOUT = 30
HEADERS = {"Content-Type": "application/json"}


def test_create_network_intent_for_ibn():
    url = f"{BASE_URL}/api/v1/telecom/intent"
    # Construct a sample high-level network intent payload consistent with the PRD schema
    payload = {
        "description": "Test intent for QoS optimization with no violations",
        "intent_type": "qos-optimization",
        "constraints": {
            "max_latency_ms": 50,
            "min_throughput_mbps": 100,
            "no_security_violations": True
        },
        "priority": 5
    }

    response = None
    try:
        response = requests.post(url, json=payload, headers=HEADERS, timeout=TIMEOUT)
        response.raise_for_status()
        data = response.json()

        print("Response data:", data)
        
        # Validate the presence of required fields in the response
        assert "intent_id" in data and isinstance(data["intent_id"], str) and data["intent_id"]
        assert "status" in data and isinstance(data["status"], str) and data["status"].lower() in {"created", "enforced", "success"}
        assert "enforcement_logs" in data and isinstance(data["enforcement_logs"], list)

        # Validate that enforcement logs contain entries and no violations exist
        enforcement_logs = data["enforcement_logs"]
        print("Enforcement logs:", enforcement_logs)
        
        # Check logs are non-empty
        assert len(enforcement_logs) > 0
        
        # Check none of the logs contain violation indication (assuming violations are marked)
        violation_keywords = ["violation", "error", "fail", "alert"]
        
        for i, log in enumerate(enforcement_logs):
            log_str = str(log).lower()
            print(f"Log {i}: {log_str}")
            for keyword in violation_keywords:
                if keyword in log_str:
                    print(f"Found violation keyword '{keyword}' in log: {log}")
        
        violations_found = any(
            any(keyword in str(log).lower() for keyword in violation_keywords) for log in enforcement_logs
        )
        print(f"Violations found: {violations_found}")
        assert not violations_found, "Violations detected in enforcement logs"

    except requests.exceptions.RequestException as e:
        assert False, f"HTTP request failed: {e}"
    except ValueError:
        assert False, "Response is not a valid JSON"
    finally:
        # Cleanup: If intent_id is created and an API to delete the intent existed,
        # we would delete the intent here to keep test environment clean.
        # However, PRD does not specify a delete endpoint, so skipping.
        pass


test_create_network_intent_for_ibn()

