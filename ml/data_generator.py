"""
OrbitRoute AI - Synthetic Dataset Generator
Generates realistic inter-satellite communication telemetry data for dynamic routing optimization.
"""

import os
import numpy as np
import pandas as pd

def generate_satellite_dataset(num_samples: int = 3500, random_state: int = 42) -> pd.DataFrame:
    """
    Generates realistic inter-satellite link data based on orbital mechanics,
    free-space path loss, queuing delay, and dynamic constellation telemetry.
    """
    np.random.seed(random_state)
    satellites = [f"SAT-{i:02d}" for i in range(1, 11)]
    
    data = []
    
    for _ in range(num_samples):
        # Pick distinct source and destination
        src, dst = np.random.choice(satellites, size=2, replace=False)
        
        # Inter-satellite distance (LEO ISL range typically 800 - 4500 km)
        distance_km = round(np.random.uniform(800.0, 4500.0), 2)
        
        # Relative velocity between satellites (0.2 to 7.8 km/s depending on orbital plane crossing)
        relative_velocity = round(np.random.uniform(0.2, 7.8), 2)
        
        # Number of hops needed across constellation mesh (1 to 5)
        # Closer distances correlate with fewer hops
        if distance_km < 1500:
            hops = np.random.choice([1, 2], p=[0.75, 0.25])
        elif distance_km < 2800:
            hops = np.random.choice([2, 3], p=[0.60, 0.40])
        else:
            hops = np.random.choice([3, 4, 5], p=[0.45, 0.40, 0.15])
            
        # Satellite internal buffer & CPU load (percentage)
        satellite_load = round(np.random.uniform(10.0, 95.0), 1)
        
        # Queue length in router buffer (packets)
        queue_length = int(np.clip(np.random.poisson(lam=satellite_load * 0.5) + np.random.randint(1, 10), 2, 120))
        
        # Signal strength (dBm) - attenuation increases with distance and velocity drift
        # Typical ISL laser / Ka-band RF signal: -40 dBm (strong) to -88 dBm (faint)
        base_signal = -42.0 - (distance_km / 120.0) - (relative_velocity * 1.5) + np.random.normal(0, 2.5)
        signal_strength = round(float(np.clip(base_signal, -90.0, -35.0)), 2)
        
        # Bandwidth (Mbps) - modulated by signal strength and channel noise
        snr_factor = (signal_strength + 90.0) / 55.0  # 0.0 to 1.0
        bandwidth_mbps = round(float(np.clip(100.0 + (snr_factor * 900.0) - (queue_length * 2.0) + np.random.normal(0, 30), 50.0, 1200.0)), 1)
        
        # Latency (ms) = Speed-of-light propagation (~3.33 us/km) * distance * hops + queuing delay + processing
        prop_delay = (distance_km / 300.0) * hops  # propagation delay
        queue_delay = (queue_length * 0.45) * hops
        proc_delay = hops * np.random.uniform(2.0, 6.0)
        latency_ms = round(float(np.clip(prop_delay + queue_delay + proc_delay, 12.0, 280.0)), 2)
        
        # Packet loss percentage (%): increases with low signal, long queues, high velocity
        loss_prob = 0.15 + (1.0 - snr_factor) * 3.5 + (queue_length / 120.0) * 2.5 + (relative_velocity / 7.8) * 1.2 + np.random.normal(0, 0.3)
        packet_loss_percent = round(float(np.clip(loss_prob, 0.05, 8.5)), 2)
        
        # Energy consumption (Joules per Megabit transceived)
        # Power increases with distance squared and transmission amplifier boost
        energy = 1.0 + (distance_km / 4500.0)**1.5 * 5.0 + (hops * 0.8) + (1.0 - snr_factor) * 2.0 + np.random.normal(0, 0.2)
        energy_consumption = round(float(np.clip(energy, 1.1, 9.8)), 2)
        
        # Connection stability (0.0 to 1.0): inverse to relative velocity and signal fluctuation
        stability = 0.95 - (relative_velocity / 7.8) * 0.35 - (packet_loss_percent / 8.5) * 0.25 - (satellite_load / 100.0) * 0.15 + np.random.normal(0, 0.04)
        connection_stability = round(float(np.clip(stability, 0.15, 0.99)), 3)
        
        # Link quality score (0 to 100)
        link_quality = round(float(np.clip((snr_factor * 40.0) + (connection_stability * 40.0) - (packet_loss_percent * 4.0) + 10.0, 5.0, 99.0)), 1)
        
        data.append({
            "source_satellite": src,
            "destination_satellite": dst,
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
            "number_of_hops": hops
        })
        
    df = pd.DataFrame(data)
    
    # Calculate Normalized Components for Routing Score
    # Lower score is better:
    # 0.35 * norm_delay + 0.30 * norm_loss + 0.20 * norm_energy + 0.15 * norm_hops
    norm_delay = (df["latency_ms"] - df["latency_ms"].min()) / (df["latency_ms"].max() - df["latency_ms"].min())
    norm_loss = (df["packet_loss_percent"] - df["packet_loss_percent"].min()) / (df["packet_loss_percent"].max() - df["packet_loss_percent"].min())
    norm_energy = (df["energy_consumption"] - df["energy_consumption"].min()) / (df["energy_consumption"].max() - df["energy_consumption"].min())
    norm_hops = (df["number_of_hops"] - df["number_of_hops"].min()) / (df["number_of_hops"].max() - df["number_of_hops"].min())
    
    routing_score = (
        0.35 * norm_delay +
        0.30 * norm_loss +
        0.20 * norm_energy +
        0.15 * norm_hops
    )
    df["routing_score"] = np.round(routing_score, 4)
    
    # Route Quality Index (0.0 to 100.0, higher is better)
    df["route_quality"] = np.round((1.0 - df["routing_score"]) * 100.0, 2)
    
    # Binary Classification Target: optimal_route (1 = Optimal Path, 0 = Suboptimal Path)
    # Threshold based on top 35% performance & stability
    score_threshold = df["routing_score"].quantile(0.38)
    df["optimal_route"] = np.where(
        (df["routing_score"] <= score_threshold) & 
        (df["connection_stability"] >= 0.55) & 
        (df["packet_loss_percent"] <= 3.0),
        1, 0
    )
    
    return df

def save_dataset(filepath: str = "data/orbitroute_dataset.csv", num_samples: int = 3500) -> str:
    """Generates and writes the dataset to disk."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df = generate_satellite_dataset(num_samples=num_samples)
    df.to_csv(filepath, index=False)
    print(f"[Dataset] Generated {len(df)} samples saved to {filepath}")
    print(f"[Dataset] Target distribution (optimal_route): \n{df['optimal_route'].value_counts(normalize=True)}")
    return filepath

if __name__ == "__main__":
    save_dataset()
