"""
ORBITROUTE AI - AI-Based Dynamic Inter-Satellite Routing
Flask Web Application & REST API Server
"""

import os
import json
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_file
from ml.data_generator import save_dataset
from ml.train_model import train_routing_model
from ml.predict import predict_route_viability, get_model
from ml.reinforcement_learning import get_rl_agent, train_and_save_rl_agent
from ml.evaluation import get_model_evaluation_metrics, get_routing_benchmark_comparison
from ml.gnn_module import get_gnn_architecture_details
from simulation.satellite_network import get_constellation

app = Flask(__name__)

# Ensure dataset and trained models exist upon server startup
DATASET_PATH = "data/orbitroute_dataset.csv"
RF_MODEL_PATH = "models/random_forest.pkl"
RL_MODEL_PATH = "models/q_learning_model.pkl"
METRICS_PATH = "results/model_metrics.json"

def bootstrap_system():
    print("[System] Initializing OrbitRoute AI Core...")
    if not os.path.exists(DATASET_PATH):
        print("[System] Generating synthetic satellite dataset...")
        save_dataset(DATASET_PATH, num_samples=3500)
    
    if not os.path.exists(RF_MODEL_PATH) or not os.path.exists(METRICS_PATH):
        print("[System] Training Random Forest baseline model...")
        train_routing_model(DATASET_PATH, RF_MODEL_PATH, METRICS_PATH)
        
    if not os.path.exists(RL_MODEL_PATH):
        print("[System] Training Q-Learning Routing Agent...")
        train_and_save_rl_agent(RL_MODEL_PATH)
        
    # Preload models
    get_model()
    get_rl_agent()
    print("[System] All AI engines ready.")

@app.route("/")
def index():
    """Renders the OrbitRoute AI mission control dashboard."""
    return render_template("index.html")

@app.route("/api/status", methods=["GET"])
def get_status():
    """System status and telemetry heartbeat."""
    constellation = get_constellation()
    metrics = get_model_evaluation_metrics()
    return jsonify({
        "status": "ONLINE",
        "system_name": "ORBITROUTE AI",
        "tagline": "AI-Based Dynamic Inter-Satellite Routing",
        "simulation_step": constellation.simulation_step,
        "satellites_online": len(constellation.satellites),
        "model_loaded": os.path.exists(RF_MODEL_PATH),
        "model_accuracy": metrics.get("accuracy", 0.0),
        "rl_model_active": os.path.exists(RL_MODEL_PATH)
    })

@app.route("/api/network", methods=["GET"])
def get_network():
    """Returns dynamic satellite constellation topology and active ISL links."""
    constellation = get_constellation()
    return jsonify(constellation.get_network_topology())

@app.route("/api/simulate", methods=["POST"])
def simulate():
    """
    Advances constellation orbital positions, updates link conditions,
    and recalculates optimal route for requested source/destination.
    """
    constellation = get_constellation()
    sim_res = constellation.advance_simulation()
    
    data = request.get_json(silent=True) or {}
    source = data.get("source", "SAT-01")
    destination = data.get("destination", "SAT-10")
    
    routing_res = constellation.compute_optimal_routes(source, destination)
    
    return jsonify({
        "simulation": sim_res,
        "network": constellation.get_network_topology(),
        "routing": routing_res
    })

@app.route("/api/optimal-route", methods=["POST"])
def optimal_route():
    """
    Computes and compares optimal routes between source and destination
    using Random Forest ML, Dijkstra, and Q-Learning.
    """
    data = request.get_json(silent=True) or {}
    source = data.get("source", "SAT-01")
    destination = data.get("destination", "SAT-10")
    
    if source == destination:
        return jsonify({"error": "Source and Destination satellites must be different."}), 400
        
    constellation = get_constellation()
    result = constellation.compute_optimal_routes(source, destination)
    
    if "error" in result:
        return jsonify(result), 404
        
    return jsonify(result)

@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Predicts route viability and cost metrics from custom link parameters.
    """
    params = request.get_json(silent=True)
    if not params:
        return jsonify({"error": "No input features provided."}), 400

    try:
        prediction = predict_route_viability(params)
        return jsonify(prediction)
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

@app.route("/api/model-performance", methods=["GET"])
def model_performance():
    """
    Returns actual evaluation metrics, confusion matrix, feature importance,
    and algorithm benchmark comparisons.
    """
    metrics = get_model_evaluation_metrics()
    benchmarks = get_routing_benchmark_comparison()
    rl_agent, _ = get_rl_agent()
    
    return jsonify({
        "metrics": metrics,
        "benchmarks": benchmarks,
        "rl_history": rl_agent.training_history
    })

@app.route("/api/dataset", methods=["GET"])
def get_dataset():
    """Returns dataset summary statistics and preview rows."""
    if not os.path.exists(DATASET_PATH):
        save_dataset(DATASET_PATH, num_samples=3500)

    df = pd.read_csv(DATASET_PATH)
    
    # Calculate summary metrics
    total_samples = int(len(df))
    feature_count = int(len(df.columns))
    optimal_count = int(df["optimal_route"].sum())
    suboptimal_count = total_samples - optimal_count

    # Preview 50 records
    preview = df.head(50).to_dict(orient="records")

    return jsonify({
        "total_samples": total_samples,
        "feature_count": feature_count,
        "train_samples": int(total_samples * 0.8),
        "test_samples": int(total_samples * 0.2),
        "optimal_routes": optimal_count,
        "suboptimal_routes": suboptimal_count,
        "columns": list(df.columns),
        "preview": preview
    })

@app.route("/api/dataset/download", methods=["GET"])
def download_dataset():
    """Allows downloading the generated dataset CSV."""
    if not os.path.exists(DATASET_PATH):
        save_dataset(DATASET_PATH, num_samples=3500)
    return send_file(DATASET_PATH, as_attachment=True, download_name="orbitroute_dataset.csv")

@app.route("/api/rl/route", methods=["POST"])
def rl_route():
    """Runs a step-by-step Q-Learning route exploration trace."""
    data = request.get_json(silent=True) or {}
    source = data.get("source", "SAT-01")
    destination = data.get("destination", "SAT-10")

    rl_agent, rl_env = get_rl_agent()
    trace = rl_agent.get_route_trace(source, destination, rl_env)
    return jsonify(trace)

@app.route("/api/gnn-info", methods=["GET"])
def gnn_info():
    """Returns architectural details and mathematical equations for GNN module."""
    return jsonify(get_gnn_architecture_details())

if __name__ == "__main__":
    bootstrap_system()
    app.run(host="0.0.0.0", port=5000, debug=True)
