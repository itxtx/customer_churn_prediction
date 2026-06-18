import numpy as np
import pandas as pd
import pytest

from src.simulation_utils import (
    calculate_revenue_impact,
    generate_simulated_customer,
    run_simulation,
    validate_distributions,
)


def test_validate_distributions_accepts_supported_specs():
    distributions = {
        "tenure": ("uniform", [1, 72]),
        "Contract": (
            "choice",
            {"options": ["Month-to-month", "Two year"], "probabilities": [0.6, 0.4]},
        ),
    }
    assert validate_distributions(distributions)


def test_validate_distributions_rejects_bad_probabilities():
    distributions = {
        "Contract": (
            "choice",
            {"options": ["Month-to-month", "Two year"], "probabilities": [0.6, 0.6]},
        )
    }
    with pytest.raises(ValueError, match="must sum to 1.0"):
        validate_distributions(distributions)


def test_generate_simulated_customer_is_seedable():
    np.random.seed(42)
    customer = generate_simulated_customer(
        {
            "tenure": ("uniform", [1, 72]),
            "Contract": (
                "choice",
                {"options": ["Month-to-month", "Two year"], "probabilities": [1.0, 0.0]},
            ),
        }
    )
    assert 1 <= customer["tenure"] <= 72
    assert customer["Contract"] == "Month-to-month"


def test_calculate_revenue_impact():
    results = pd.DataFrame(
        {
            "churn_probability": [0.25, 0.75],
            "MonthlyCharges": [100.0, 200.0],
        }
    )
    impact = calculate_revenue_impact(results)
    assert impact["total_customers"] == 2
    assert impact["expected_churners"] == 1.0
    assert impact["expected_lost_revenue"] == 175.0
    assert impact["revenue_retention_rate"] == pytest.approx(125.0 / 300.0)


def test_run_simulation_single_threaded():
    class MockPredictor:
        def prepare_batch_customers(self, data):
            return data.copy(), None

        class model:
            @staticmethod
            def predict_proba(data):
                return np.tile(np.array([[0.7, 0.3]]), (len(data), 1))

    distributions = {
        "tenure": ("uniform", [1, 72]),
        "MonthlyCharges": ("normal", [65, 10]),
    }
    results = run_simulation(MockPredictor(), distributions, num_simulations=3, n_jobs=1)
    assert len(results) == 3
    assert list(results["churn_probability"]) == [0.3, 0.3, 0.3]
    assert set(["customer_id", "risk_level", "MonthlyCharges"]).issubset(results.columns)
