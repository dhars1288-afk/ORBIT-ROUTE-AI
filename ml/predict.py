"""
OrbitRoute AI - Inference & Prediction Engine
Loads the trained Random Forest model and predicts route viability and weighted routing costs.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any

MODEL_PATH = "models/random_forest.pkl"
_loaded_package = None

def get_model():
    """Lazy loader for trained model package."""
    global _loaded_package
    if _loaded_package is None:
        if not os.path.exists(MODEL_PATH):
            from ml.train_model import train_routing_model
            train_routing_model(model_save_path=MODEL_PATH)
        _loaded_package = joblib.load(MODEL_PATH)
    return _loaded_package

def predict_route_viability(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Predicts route classification, optimal probability, and composite routing cost.
    
    Expected keys:
    source_satellite, destination_satellite, distance_km, signal_strength,
    bandwidth_mbps, latency_ms, packet_loss_percent, energy_consumption,
    link_quality, satellite_load, queue_length, relative_velocity,
    connection_stability, number_of_hops
    """
    pkg = get_model()
    model = pkg["model"]
    src_encoder = pkg["src_encoder"]
    dst_encoder = pkg["dst_encoder"]
    feature_names = pkg["feature_names"]

    src = str(params.get("source_satellite", "SAT-01"))
    dst = str(params.get("destination_satellite", "SAT-10"))

    # Encode satellites
    try:
        src_enc = int(src_encoder.transform([src])[0])
    except Exception:
        src_enc = 0

    try:
        dst_enc = int(dst_encoder.transform([dst])[0])
    except Exception:
        dst_enc = 9

    distance_km = float(params.get("distance_km", 2200.0))
    signal_strength = float(params.get("signal_strength", -58.0))
    bandwidth_mbps = float(params.get("bandwidth_mbps", 650.0))
    latency_ms = float(params.get("latency_ms", 65.0))
    packet_loss_percent = float(params.get("packet_loss_percent", 0.8))
    energy_consumption = float(params.get("energy_consumption", 3.2))
    link_quality = float(params.get("link_quality", 82.0))
    satellite_load = float(params.get("satellite_load", 35.0))
    queue_length = int(params.get("queue_length", 15))
    relative_velocity = float(params.get("relative_velocity", 2.4))
    connection_stability = float(params.get("connection_stability", 0.88))
    number_of_hops = int(params.get("number_of_hops", 3))

    features_dict = {
        "source_encoded": src_enc,
        "destination_encoded": dst_enc,
        "distance_km": distance_km,
        "signal_strength": signal_strength,
        "bandwidth_mbps": bandwidth_mbps,
        "latency_ms": latency_ms,
        "packet_loss_percent": packet_loss_percent,
        "energy_consumption": energy_consumption,
        "link_quality": link_quality,
        "satellite_load": satellite_load,
        "queue_length": queue_length,
        "relative_velocity": relative_velocity,
        "connection_stability": connection_stability,
        "number_of_hops": number_of_hops
    }

    X_input = pd.DataFrame([[features_dict[col] for col in feature_names]], columns=feature_names)

    # ML Model prediction
    is_optimal_pred = int(model.predict(X_input)[0])
    proba = model.predict_proba(X_input)[0]
    optimal_proba = float(proba[1]) if len(proba) > 1 else float(is_optimal_pred)

    # Calculate exact weighted routing cost
    # Total Cost = 0.35 * Delay + 0.30 * Loss + 0.20 * Energy + 0.15 * Hops (Normalized)
    norm_delay = min(max((latency_ms - 10.0) / 270.0, 0.0), 1.0)
    norm_loss = min(max((packet_loss_percent - 0.0) / 8.5, 0.0), 1.0)
    norm_energy = min(max((energy_consumption - 1.0) / 9.0, 0.0), 1.0)
    norm_hops = min(max((number_of_hops - 1.0) / 5.0, 0.0), 1.0)

    routing_score = round(
        (0.35 * norm_delay) +
        (0.30 * norm_loss) +
        (0.20 * norm_energy) +
        (0.15 * norm_hops),
        4
    )

    route_quality = round((1.0 - routing_score) * 100.0, 1)

    if is_optimal_pred == 1 and routing_score < 0.35:
        recommendation = "Optimal route verified. Minimal delay, low packet loss, and high link stability guaranteed."
        status_label = "Optimal"
        badge_color = "success"
    elif routing_score < 0.55:
        recommendation = "Acceptable secondary route. Usable under high constellation traffic load."
        status_label = "Sub-optimal"
        badge_color = "warning"
    else:
        recommendation = "High combined routing cost detected. Re-routing or link re-acquisition recommended."
        status_label = "Degraded"
        badge_color = "danger"

    return {
        "source": src,
        "destination": dst,
        "is_optimal": is_optimal_pred == 1,
        "optimal_probability": round(optimal_proba * 100.0, 1),
        "routing_score": routing_score,
        "route_quality": route_quality,
        "predicted_delay": latency_ms,
        "predicted_packet_loss": packet_loss_percent,
        "energy_consumption": energy_consumption,
        "number_of_hops": number_of_hops,
        "status_label": status_label,
        "badge_color": badge_color,
        "recommendation": recommendation,
        "cost_breakdown": {
            "delay_penalty": round(0.35 * norm_delay, 4),
            "loss_penalty": round(0.30 * norm_loss, 4),
            "energy_penalty": round(0.20 * norm_energy, 4),
            "hops_penalty": round(0.15 * norm_hops, 4)
        }
    }
