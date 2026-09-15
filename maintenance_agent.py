from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


BASE_DIR = Path(__file__).resolve().parent
KNOWLEDGE_PATH = BASE_DIR / "maintenance_knowledge.json"


def _load_kb() -> Dict[str, Dict[str, Any]]:
    if KNOWLEDGE_PATH.exists():
        try:
            with KNOWLEDGE_PATH.open("r", encoding="utf-8") as handle:
                kb = json.load(handle)
            if isinstance(kb, dict):
                return kb
        except (json.JSONDecodeError, OSError):
            pass
    return {}


MAINTENANCE_KB = _load_kb() or {
    "Tool Wear Failure (TWF)": {
        "summary": "Tool wear is elevated beyond the normal replacement window and is driving a rising failure probability.",
        "priority": "CRITICAL / EMERGENCY",
        "eta": "< 2 hours",
        "recommended_action": "Replace the cutting tool insert immediately, inspect spindle alignment, and schedule a preventive maintenance stop.",
        "checklist": [
            "Inspect tool holder alignment and spindle runout.",
            "Verify cutter wear against the replacement threshold and replace inserts.",
            "Check workpiece clamping integrity and feed consistency.",
            "Document wear progression in the machine log.",
        ],
        "parts": ["Carbide Insert Set (Part #T-880)", "Spindle Locking Fixture (Part #S-310)"],
        "signals": [
            "tool_wear_min >= 200 min",
            "tool_wear_min is trending upward across the last operating cycle",
            "torque is elevated relative to nominal spindle load",
        ],
    },
    "Heat Dissipation Failure (HDF)": {
        "summary": "Thermal imbalance is reducing cooling performance and increasing the likelihood of heat-related failure.",
        "priority": "HIGH PRIORITY",
        "eta": "Within 24 hours",
        "recommended_action": "Flush coolant lines, verify heat exchanger performance, and correct the thermal operating envelope before the next production cycle.",
        "checklist": [
            "Flush and inspect coolant lines for blockage or contamination.",
            "Verify coolant flow rate is at or above the plant minimum.",
            "Check the heat exchanger and radiator condition for fouling.",
            "Confirm temperature deltas remain within the safe operating band.",
        ],
        "parts": ["Coolant Filter Cartridge (Part #C-104)", "Thermal Sensor Calibration Kit (Part #TH-09)"],
        "signals": [
            "process_temperature_k - air_temperature_k is below the healthy threshold",
            "rotational_speed_rpm remains below the normal operating band",
            "temperature drift is accelerating under a sustained load",
        ],
    },
    "Power Failure (PWF)": {
        "summary": "Motor power demand is outside the normal electrical operating envelope, suggesting an electrical or drive issue.",
        "priority": "HIGH PRIORITY",
        "eta": "Within 24 hours",
        "recommended_action": "Inspect the VFD, drive train, and electrical supply before continuing operation at high load.",
        "checklist": [
            "Measure three-phase current balance and check inverter output stability.",
            "Inspect motor insulation and electrical connector condition.",
            "Review power draw against machine load profile and reset out-of-range thresholds.",
            "Check for loose power cabling or fan failures in the drive cabinet.",
        ],
        "parts": ["VFD Inverter Relay Module (Part #E-550)", "Power Cable Set (Part #P-202)"],
        "signals": [
            "power_w is below or above the plant operating range",
            "rotational_speed_rpm and torque indicate abnormal power draw",
            "electrical load is inconsistent with nominal machine conditions",
        ],
    },
    "Overstrain Failure (OSF)": {
        "summary": "The machine is being pushed beyond its mechanical load design envelope, creating elevated stress on spindle and drive components.",
        "priority": "HIGH PRIORITY",
        "eta": "Within 24 hours",
        "recommended_action": "Reduce cutting loads, validate workpiece hardness, and inspect guideways and bearings for mechanical wear.",
        "checklist": [
            "Inspect ball screws, guideways, and spindle bearings for backlash or scoring.",
            "Reduce feed rate and validate workpiece hardness against tool capability.",
            "Check for abnormal vibration and mechanical resonance during load transitions.",
            "Review lubrication distribution and taper lock condition.",
        ],
        "parts": ["Spindle Bearings Set (Part #B-320)", "Guideway Lubrication Kit (Part #G-118)"],
        "signals": [
            "tool_wear_min and torque increase together under sustained load",
            "mechanical overstrain is consistent with elevated spindle and drive stress",
            "machine operating envelope is exceeding the expected work profile",
        ],
    },
    "Normal Operation": {
        "summary": "The current machine state remains within the healthy operating envelope and does not require emergency intervention.",
        "priority": "LOW / ROUTINE",
        "eta": "Routine cycle",
        "recommended_action": "Maintain normal operation, continue routine monitoring, and keep the standard shift log updated.",
        "checklist": [
            "Continue routine monitoring and visual inspection.",
            "Document baseline behavior during the current shift.",
            "Keep standard sensor calibration checks on schedule.",
        ],
        "parts": ["None"],
        "signals": [
            "telemetry remains within normal factory thresholds",
            "failure probability remains below the medium-risk threshold",
        ],
    },
    "Multivariable Machine Strain": {
        "summary": "Multiple operational stressors are compounding risk, even though no single failure signature is dominant.",
        "priority": "MEDIUM",
        "eta": "Next shift",
        "recommended_action": "Perform an inspection within 24 hours and verify calibration, cooling, and tooling health together.",
        "checklist": [
            "Run a full sensor calibration and data quality review.",
            "Inspect tool wear, thermal behavior, and power draw together.",
            "Review recent production logs for load spikes or cooling interruptions.",
            "Schedule a preventive maintenance check before the next high-load cycle.",
        ],
        "parts": ["Preventive Maintenance Kit (Part #PM-01)", "Thermal Sensor Calibration Pack (Part #TC-45)"],
        "signals": [
            "multiple sensor signals are trending away from their healthy baselines",
            "no single root cause dominates, but combined drift is increasing risk",
        ],
    },
}


def _compute_power_watts(metrics: Dict[str, Any]) -> float:
    rpm = float(metrics.get("rotational_speed_rpm", 0.0) or 0.0)
    torque = float(metrics.get("torque_nm", 0.0) or 0.0)
    return rpm * torque * (2 * 3.141592653589793 / 60.0)


def _normalize_failure_mode(failure_mode: Optional[str], metrics: Dict[str, Any], failure_probability: float) -> str:
    if failure_mode and failure_mode in MAINTENANCE_KB:
        return failure_mode

    wear = float(metrics.get("tool_wear_min", 0.0) or 0.0)
    process_temp = float(metrics.get("process_temperature_k", 0.0) or 0.0)
    air_temp = float(metrics.get("air_temperature_k", 0.0) or 0.0)
    rpm = float(metrics.get("rotational_speed_rpm", 0.0) or 0.0)
    torque = float(metrics.get("torque_nm", 0.0) or 0.0)
    temp_diff = process_temp - air_temp
    power_w = _compute_power_watts(metrics)

    if wear >= 200:
        return "Tool Wear Failure (TWF)"
    if temp_diff < 8.6 and rpm < 1380:
        return "Heat Dissipation Failure (HDF)"
    if power_w < 3500 or power_w > 9000:
        return "Power Failure (PWF)"
    if (wear >= 150 and torque >= 55) or (wear >= 180 and torque >= 45):
        return "Overstrain Failure (OSF)"
    if failure_probability >= 0.26:
        return "Multivariable Machine Strain"
    return "Normal Operation"


def _compute_priority(failure_probability: float, mode: str) -> str:
    if failure_probability >= 0.75 or mode in {"Tool Wear Failure (TWF)", "Heat Dissipation Failure (HDF)"}:
        return "CRITICAL / EMERGENCY"
    if failure_probability >= 0.26:
        return "HIGH PRIORITY"
    if failure_probability >= 0.15:
        return "MEDIUM"
    return "LOW / ROUTINE"


def _compute_eta(failure_probability: float, mode: str) -> str:
    if failure_probability >= 0.75 or mode in {"Tool Wear Failure (TWF)"}:
        return "< 2 hours"
    if failure_probability >= 0.26:
        return "Within 24 hours"
    if failure_probability >= 0.15:
        return "Next shift"
    return "Routine cycle"


def _build_evidence(metrics: Dict[str, Any], mode: str) -> List[str]:
    evidence: List[str] = []
    wear = float(metrics.get("tool_wear_min", 0.0) or 0.0)
    process_temp = float(metrics.get("process_temperature_k", 0.0) or 0.0)
    air_temp = float(metrics.get("air_temperature_k", 0.0) or 0.0)
    rpm = float(metrics.get("rotational_speed_rpm", 0.0) or 0.0)
    torque = float(metrics.get("torque_nm", 0.0) or 0.0)
    power_w = _compute_power_watts(metrics)
    temp_diff = process_temp - air_temp

    if wear > 0:
        evidence.append(f"Tool wear = {wear:.1f} minutes; operating near the wear alert boundary.")
    if temp_diff > 0:
        evidence.append(f"Temperature delta = {temp_diff:.2f} K; this is consistent with a thermal stress pattern.")
    if rpm > 0:
        evidence.append(f"Rotor speed = {rpm:.0f} RPM; the machine is within a load regime that amplifies the observed failure signature.")
    if torque > 0:
        evidence.append(f"Torque = {torque:.2f} Nm; loading is elevated and contributes to mechanical stress.")
    if power_w > 0:
        evidence.append(f"Power draw = {power_w:.1f} W; electrical demand is within the monitored risk profile.")

    mode_signals = MAINTENANCE_KB.get(mode, {}).get("signals", [])
    if mode_signals:
        evidence.append(f"Failure mode signature: {mode_signals[0]}.")

    return evidence[:5]


def build_recommendation_context(
    failure_probability: float,
    risk_tier: str,
    failure_mode: Optional[str],
    metrics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    metrics = metrics or {}
    mode = _normalize_failure_mode(failure_mode, metrics, failure_probability)
    knowledge = MAINTENANCE_KB.get(mode, MAINTENANCE_KB["Normal Operation"])

    priority = _compute_priority(failure_probability, mode)
    eta = _compute_eta(failure_probability, mode)
    action = knowledge.get("recommended_action", "Continue routine operation and monitoring.")
    steps = knowledge.get("checklist", [])
    parts = knowledge.get("parts", ["None"])
    evidence = _build_evidence(metrics, mode)

    summary = knowledge.get("summary", "The current telemetry indicates normal operating conditions.")
    if risk_tier and risk_tier.lower() not in {"normal", "low risk"}:
        summary = f"{summary} Risk tier is {risk_tier}, which is consistent with the observed machine condition."

    return {
        "primary_failure_mode": mode,
        "risk_tier": risk_tier or "Low Risk",
        "priority": priority,
        "eta": eta,
        "summary": summary,
        "action": action,
        "maintenance_steps": steps,
        "required_spare_parts": parts,
        "savings_usd": 4500.0 if failure_probability >= 0.26 else 0.0,
        "evidence": evidence,
    }


def get_grounded_recommendation(
    failure_probability: float,
    risk_tier: str,
    failure_mode: Optional[str],
    metrics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    context = build_recommendation_context(failure_probability, risk_tier, failure_mode, metrics)
    return {
        "failure_mode": context["primary_failure_mode"],
        "risk_tier": context["risk_tier"],
        "priority": context["priority"],
        "eta": context["eta"],
        "summary": context["summary"],
        "action": context["action"],
        "maintenance_steps": context["maintenance_steps"],
        "required_spare_parts": context["required_spare_parts"],
        "savings_usd": context["savings_usd"],
        "evidence": context["evidence"],
    }


def retrieve_knowledge(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """Retrieve the most relevant KB entries with lightweight keyword overlap."""
    query_terms = set(re.findall(r"[a-z0-9_]+", query.lower()))
    scored_entries = []
    for mode, entry in MAINTENANCE_KB.items():
        mode_terms = set(re.findall(r"[a-z0-9_]+", mode.lower()))
        text = " ".join([
            mode,
            str(entry.get("summary", "")),
            str(entry.get("recommended_action", "")),
            " ".join(entry.get("checklist", [])),
            " ".join(entry.get("signals", [])),
        ])
        terms = set(re.findall(r"[a-z0-9_]+", text.lower()))
        score = len(query_terms & terms) + (3 * len(query_terms & mode_terms))
        if mode.lower() in query.lower():
            score += 10
        scored_entries.append((score, mode, entry))

    scored_entries.sort(key=lambda item: (-item[0], item[1]))
    return [
        {"mode": mode, "knowledge": entry, "score": score}
        for score, mode, entry in scored_entries[:max(1, top_k)]
    ]


def _allowed_spare_parts() -> set[str]:
    return {
        part
        for entry in MAINTENANCE_KB.values()
        for part in entry.get("parts", [])
        if part != "None"
    }


def _merge_llm_recommendation(base: Dict[str, Any], candidate: Dict[str, Any]) -> Dict[str, Any]:
    """Accept only the structured fields and spare parts supported by the KB."""
    merged = base.copy()
    for field in ("summary", "action", "priority", "eta"):
        value = candidate.get(field)
        if isinstance(value, str) and value.strip():
            merged[field] = value.strip()

    steps = candidate.get("maintenance_steps")
    if isinstance(steps, list) and all(isinstance(step, str) and step.strip() for step in steps):
        merged["maintenance_steps"] = steps[:6]

    parts = candidate.get("required_spare_parts")
    allowed_parts = _allowed_spare_parts()
    if isinstance(parts, list):
        valid_parts = [part for part in parts if isinstance(part, str) and part in allowed_parts]
        if valid_parts or not parts:
            merged["required_spare_parts"] = valid_parts or ["None"]
        else:
            merged["required_spare_parts"] = base["required_spare_parts"]
    return merged


def get_rag_recommendation(
    failure_probability: float,
    risk_tier: str,
    failure_mode: Optional[str],
    metrics: Optional[Dict[str, Any]] = None,
    top_k: int = 3,
) -> Dict[str, Any]:
    """Use retrieved KB context with an optional LLM, then fall back safely."""
    metrics = metrics or {}
    deterministic = build_recommendation_context(
        failure_probability, risk_tier, failure_mode, metrics
    )
    query = " ".join([
        str(failure_mode or ""),
        str(risk_tier),
        f"failure probability {failure_probability:.3f}",
        " ".join(f"{key} {value}" for key, value in metrics.items()),
    ])
    retrieved = retrieve_knowledge(query, top_k=top_k)
    result = {
        **deterministic,
        "retrieved_context": [item["mode"] for item in retrieved],
        "rag_used": False,
    }

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return result

    try:
        from openai import OpenAI

        context = json.dumps(
            [{"mode": item["mode"], **item["knowledge"]} for item in retrieved],
            ensure_ascii=True,
        )
        prompt = (
            "You are a maintenance planner. Use ONLY the retrieved knowledge below. "
            "Return one JSON object with exactly these keys: summary, action, priority, "
            "eta, maintenance_steps, required_spare_parts. Do not invent parts, thresholds, "
            "or procedures.\n\n"
            f"Telemetry and model result:\n{query}\n\n"
            f"Retrieved knowledge:\n{context}"
        )
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": "Return valid JSON only."},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )
        content = response.choices[0].message.content or "{}"
        candidate = json.loads(content)
        if not isinstance(candidate, dict):
            raise ValueError("LLM response was not a JSON object")
        result = {
            **_merge_llm_recommendation(deterministic, candidate),
            "retrieved_context": [item["mode"] for item in retrieved],
            "rag_used": True,
        }
    except Exception:
        # A recommendation must remain available during API, parsing, or dependency failures.
        pass

    return result
