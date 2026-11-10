"""API tests against the real, committed model artifacts (models/*.joblib,
models/thresholds.json are small and checked into git for exactly this
reason), using a hand-built synthetic transaction rather than the real
dataset."""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app

VALID_TRANSACTION = {
    "TransactionID": 1,
    "TransactionDT": 100000,
    "TransactionAmt": 125.50,
    "ProductCD": "W",
    "card1": 1234,
    "card2": 200.0,
    "card4": "visa",
    "card6": "debit",
    "addr1": 150.0,
    "P_emaildomain": "gmail.com",
    "DeviceType": "mobile",
    "C": {"C1": 1.0, "C2": 0.0},
    "D": {"D1": 14.0},
    "M": {"M1": "T", "M5": "F"},
    "V": {"V1": 1.0},
}


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_model_info(client):
    resp = client.get("/model/info")
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_type"] == "LGBMClassifier"
    assert 0.0 <= body["review_threshold"] <= body["block_threshold"] <= 1.0


def test_score_returns_valid_response(client):
    resp = client.post("/score", json=VALID_TRANSACTION)
    assert resp.status_code == 200
    body = resp.json()

    assert body["TransactionID"] == 1
    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert body["decision"] in {"approve", "review", "block"}
    assert len(body["top_reasons"]) == 3
    for reason in body["top_reasons"]:
        assert reason["direction"] in {"increases", "decreases"}


def test_score_rejects_non_positive_amount(client):
    bad_txn = {**VALID_TRANSACTION, "TransactionAmt": -5.0}
    resp = client.post("/score", json=bad_txn)
    assert resp.status_code == 422


def test_score_batch(client):
    batch = {"transactions": [VALID_TRANSACTION, {**VALID_TRANSACTION, "TransactionID": 2}]}
    resp = client.post("/score/batch", json=batch)
    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 2
    assert [r["TransactionID"] for r in body] == [1, 2]
