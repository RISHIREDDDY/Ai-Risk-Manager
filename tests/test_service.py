"""
test_service.py — Unit and integration tests for FastAPI Service Layer (Phase 4).
"""

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from fastapi.testclient import TestClient
from main import app
from database import get_connection

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    data = response.json()
    assert data["status"] == "healthy"
    assert data["database"] == "connected"
    print("[PASS] /health passed")


def test_root():
    response = client.get("/")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    data = response.json()
    assert data["service"] == "AI Risk Manager API"
    print("[PASS] / root endpoint passed")


def test_get_pending_disputes():
    response = client.get("/api/disputes/pending?status=all")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    data = response.json()
    assert "total" in data
    assert "disputes" in data
    assert data["total"] > 0
    assert len(data["disputes"]) > 0

    first = data["disputes"][0]
    assert "dispute_id" in first
    assert "reason_code" in first
    assert "disputed_amount" in first
    print(f"[PASS] /api/disputes/pending returned {data['total']} total disputes")


def test_get_audit_logs():
    response = client.get("/api/audit-logs")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    data = response.json()
    assert "total" in data
    assert "logs" in data
    print(f"[PASS] /api/audit-logs returned {data['total']} logs")


def test_score_dispute():
    # Fetch a sample dispute
    conn = get_connection()
    row = conn.execute("SELECT dispute_id FROM disputes LIMIT 1").fetchone()
    conn.close()

    assert row is not None, "No disputes in database to test scoring"
    dispute_id = row["dispute_id"]

    response = client.post("/api/disputes/score", json={"dispute_id": dispute_id})
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    data = response.json()
    assert data["dispute_id"] == dispute_id
    assert data["decision"] in ["CONTEST_DISPUTE", "ACCEPT_DISPUTE"]
    assert 0.0 <= data["confidence"] <= 1.0
    assert len(data["reasoning"]) > 0
    assert len(data["tools_used"]) > 0
    print(f"[PASS] /api/disputes/score scored {dispute_id} -> {data['decision']} (conf: {data['confidence']})")


def test_score_invalid_dispute():
    response = client.post("/api/disputes/score", json={"dispute_id": "NON_EXISTENT_ID"})
    assert response.status_code == 404
    print("[PASS] /api/disputes/score correctly returns 404 for invalid dispute ID")


def test_audit_logs_search_and_filter():
    # Test filtering by decision
    response = client.get("/api/audit-logs?decision=CONTEST_DISPUTE")
    assert response.status_code == 200
    data = response.json()
    for item in data["logs"]:
        assert item["decision"] == "CONTEST_DISPUTE"

    # Test search query
    response = client.get("/api/audit-logs?search=evidence")
    assert response.status_code == 200
    print("[PASS] /api/audit-logs filter and search query passed")


def test_evaluation_metrics_endpoint():
    # Ensure test records have mock/sample audit logs or are scored so evaluation runs quickly
    conn = get_connection()
    test_rows = conn.execute("SELECT dispute_id, ground_truth_label FROM disputes WHERE is_test_set = 1").fetchall()
    for r in test_rows:
        # Insert a default log if not existing to enable instant test verification
        conn.execute(
            """
            INSERT OR IGNORE INTO audit_logs (dispute_id, decision, confidence, tools_called, reasoning, evidence_letter, created_at)
            VALUES (?, ?, 0.95, '["get_transaction", "get_delivery_proof", "check_communication_logs"]', 'Automated test evaluation entry', 'Evidence letter', '2026-08-24T22:00:00')
            """,
            (r["dispute_id"], "CONTEST_DISPUTE" if r["ground_truth_label"] == "valid_defense" else "ACCEPT_DISPUTE")
        )
    conn.commit()
    conn.close()

    response = client.get("/api/evaluation/metrics?rescore=false")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    data = response.json()
    assert data["total_test_records"] > 0
    assert "accuracy" in data
    assert "precision" in data
    assert "recall" in data
    assert "f1_score" in data
    assert "confusion_matrix" in data
    assert "financial_impact" in data
    assert len(data["test_cases"]) == data["total_test_records"]
    print(f"[PASS] /api/evaluation/metrics passed: Acc={data['accuracy']}, Precision={data['precision']}, Recall={data['recall']}, Net Financial Benefit=INR {data['financial_impact']['net_financial_benefit']}")


if __name__ == "__main__":
    print("\n--- Running Phase 4 Service Layer Verification Tests ---")
    test_health()
    test_root()
    test_get_pending_disputes()
    test_get_audit_logs()
    test_score_invalid_dispute()
    test_score_dispute()
    test_audit_logs_search_and_filter()
    test_evaluation_metrics_endpoint()
    print("\n=== ALL PHASE 4 SERVICE LAYER TESTS PASSED SUCCESSFULLY! ===\n")
