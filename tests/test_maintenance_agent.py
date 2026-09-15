import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from maintenance_agent import build_recommendation_context, get_grounded_recommendation


def test_build_recommendation_context_for_high_risk_tool_wear():
    context = build_recommendation_context(
        failure_probability=0.82,
        risk_tier="Critical Risk",
        failure_mode="Tool Wear Failure (TWF)",
        metrics={
            "tool_wear_min": 225,
            "process_temperature_k": 309.5,
            "air_temperature_k": 298.5,
            "rotational_speed_rpm": 1420,
            "torque_nm": 48.0,
        },
    )

    assert context["primary_failure_mode"] == "Tool Wear Failure (TWF)"
    assert context["risk_tier"] == "Critical Risk"
    assert context["priority"] in {"CRITICAL / EMERGENCY", "HIGH PRIORITY"}
    assert "tool wear" in context["summary"].lower()


def test_get_grounded_recommendation_returns_actionable_guidance():
    recommendation = get_grounded_recommendation(
        failure_probability=0.78,
        risk_tier="Critical Risk",
        failure_mode="Heat Dissipation Failure (HDF)",
        metrics={
            "process_temperature_k": 312.0,
            "air_temperature_k": 298.2,
            "rotational_speed_rpm": 1200,
            "torque_nm": 55.0,
            "tool_wear_min": 90,
        },
    )

    assert recommendation["action"]
    assert recommendation["priority"] == "CRITICAL / EMERGENCY"
    assert recommendation["evidence"]
    assert recommendation["maintenance_steps"]
