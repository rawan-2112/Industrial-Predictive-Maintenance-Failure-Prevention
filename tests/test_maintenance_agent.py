import sys
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from maintenance_agent import (
    build_recommendation_context,
    get_grounded_recommendation,
    get_rag_recommendation,
    retrieve_knowledge,
)

from app import get_preset_params
from openai import OpenAI

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


def test_get_preset_params_returns_expected_machine_profile():
    preset = get_preset_params("Tool Wear (TWF)")

    assert preset["air_temperature_k"] == 298.5
    assert preset["tool_wear_min"] == 225
    assert preset["rotation_speed_rpm"] == 1420
    assert preset["machine_type"] == "L"


def test_retrieval_prioritizes_matching_failure_mode():
    retrieved = retrieve_knowledge("tool wear 225 minutes torque", top_k=3)

    assert retrieved[0]["mode"] == "Tool Wear Failure (TWF)"


def test_rag_falls_back_without_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    recommendation = get_rag_recommendation(
        failure_probability=0.82,
        risk_tier="Critical Risk",
        failure_mode="Tool Wear Failure (TWF)",
        metrics={"tool_wear_min": 225, "torque_nm": 48.0},
    )

    assert recommendation["rag_used"] is False
    assert recommendation["required_spare_parts"] == [
        "Carbide Insert Set (Part #T-880)",
        "Spindle Locking Fixture (Part #S-310)",
    ]

def test_get_rag_recommendation_with_api_key():
    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(
        api_key=api_key,
        base_url=os.getenv("OPENAI_BASE_URL"),
    )

    recommendation = get_rag_recommendation(
        failure_probability=0.82,
        risk_tier="Critical Risk",
        failure_mode="Tool Wear Failure (TWF)",
        metrics={"tool_wear_min": 225, "torque_nm": 48.0},
    )

    assert recommendation["rag_used"] is True
    assert recommendation["required_spare_parts"] == [
        "Carbide Insert Set (Part #T-880)",
        "Spindle Locking Fixture (Part #S-310)",
    ]
