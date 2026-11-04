import pytest
from fastapi.testclient import TestClient
from src.api.main import app


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "service" in data
    assert data["docs"] == "/docs"


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "features" in data
    assert len(data["features"]) == 7


def test_predict_valid_payload(client):
    payload = {
        "cereal_yield": 2845.5,
        "fdi_perc_gdp": 2.15,
        "gni_per_cap": 8420.0,
        "en_per_cap": 2150.3,
        "pop_urb_aggl_perc": 34.2,
        "prot_area_perc": 15.6,
        "pop_growth_perc": 1.25,
        "urb_pop_growth_perc": 2.45,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_co2_per_cap" in data
    assert isinstance(data["predicted_co2_per_cap"], (int, float))
    assert data["predicted_co2_per_cap"] >= 0.0
    assert "is_out_of_distribution" in data


def test_predict_invalid_payload_validation_error(client):
    # Invalid: negative cereal_yield and out-of-bounds percentage
    payload = {
        "cereal_yield": -100.0,
        "fdi_perc_gdp": 2.15,
        "gni_per_cap": 8420.0,
        "en_per_cap": 2150.3,
        "pop_urb_aggl_perc": 150.0,  # invalid > 100
        "prot_area_perc": 15.6,
        "pop_growth_perc": 1.25,
        "urb_pop_growth_perc": 2.45,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Unprocessable Entity from Pydantic


def test_batch_predict_endpoint(client):
    payload = {
        "records": [
            {
                "cereal_yield": 2500.0,
                "gni_per_cap": 5000.0,
                "en_per_cap": 1500.0,
                "pop_urb_aggl_perc": 20.0,
                "prot_area_perc": 10.0,
                "pop_growth_perc": 1.0,
                "urb_pop_growth_perc": 2.0,
            },
            {
                "cereal_yield": 4000.0,
                "gni_per_cap": 25000.0,
                "en_per_cap": 4500.0,
                "pop_urb_aggl_perc": 45.0,
                "prot_area_perc": 18.0,
                "pop_growth_perc": 0.5,
                "urb_pop_growth_perc": 1.2,
            },
        ]
    }
    response = client.post("/batch-predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 2
    assert len(data["predictions"]) == 2


def test_scenario_endpoint(client):
    payload = {
        "country": "IND",
        "start_year": 2024,
        "years_ahead": 5,
        "base_features": {
            "cereal_yield": 2845.5,
            "gni_per_cap": 8420.0,
            "en_per_cap": 2150.3,
            "pop_urb_aggl_perc": 34.2,
            "prot_area_perc": 15.6,
            "pop_growth_perc": 1.25,
            "urb_pop_growth_perc": 2.45,
        },
        "growth_rate_overrides": {
            "en_per_cap": -0.02,
        },
    }
    response = client.post("/scenario", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["country"] == "IND"
    assert len(data["timeline"]) == 5
    assert len(data["projected_co2"]) == 5
