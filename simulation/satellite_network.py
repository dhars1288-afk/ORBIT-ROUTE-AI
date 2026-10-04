"""
OrbitRoute AI - Dynamic Satellite Network Simulation Engine
Simulates low-Earth-orbit (LEO) constellation movement, time-varying inter-satellite laser links (ISL),
dynamic queuing, and ML-driven multi-objective route selection using NetworkX.
"""

import math
import random
import networkx as nx
from typing import Dict, List, Tuple, Any, Optional

class SatelliteConstellation:
    """
    Simulates a dynamic 10-satellite Walker-Delta constellation.
    Satellites orbit in two complementary planes.
    """
    def __init__(self):
        self.simulation_step = 0
        self.satellites = {}
        self.graph = nx.Graph()
        self.previous_optimal_route = []
        self._init_constellation()
        self._build_graph()

    def _init_constellation(self):
        """Initializes 10 satellites with orbital mechanics parameters."""
        # Plane A: SAT-01 to SAT-05 (Inclination 53 deg, phase offset 0)
        # Plane B: SAT-06 to SAT-10 (Inclination 53 deg, phase offset 36 deg)
        for i in range(1, 11):
            sat_id = f"SAT-{i:02d}"
            plane = "A" if i <= 5 else "B"
            base_angle = ((i - 1) % 5) * (2 * math.pi / 5)
            if plane == "B":
                base_angle += math.radians(36)

            self.satellites[sat_id] = {
                "id": sat_id,
                "name": f"Orbital Node {i:02d}",
                "plane": plane,
                "base_angle": base_angle,
                "current_angle": base_angle,
                "altitude_km": 550.0 + (i % 3) * 15.0,
                "velocity_kms": 7.56,
                "cpu_load": round(random.uniform(20.0, 50.0), 1),
                "queue_length": random.randint(5, 30),
                "battery_percent": round(random.uniform(85.0, 99.0), 1),
                "x": 0.0,
                "y": 0.0
            }
        self._update_coordinates()

    def _update_coordinates(self):
        """Maps orbital angles to 2D visualization plane."""
        r_inner = 180.0
        r_outer = 280.0
        center_x = 400.0
        center_y = 300.0

        for sat_id, sat in self.satellites.items():
            radius = r_inner if sat["plane"] == "A" else r_outer
            sat["x"] = round(center_x + radius * math.cos(sat["current_angle"]), 1)
            sat["y"] = round(center_y + radius * math.sin(sat["current_angle"]), 1)

    def _build_graph(self):
        """Initializes NetworkX communication topology."""
        self.graph.clear()
        for sat_id, sat in self.satellites.items():
            self.graph.add_node(sat_id, **sat)

        # Baseline mesh connections (intra-plane and cross-plane ISL)
        base_edges = [
            ("SAT-01", "SAT-02"), ("SAT-02", "SAT-03"), ("SAT-03", "SAT-04"),
            ("SAT-04", "SAT-05"), ("SAT-05", "SAT-01"),
            ("SAT-06", "SAT-07"), ("SAT-07", "SAT-08"), ("SAT-08", "SAT-09"),
            ("SAT-09", "SAT-10"), ("SAT-10", "SAT-06"),
            # Inter-plane cross links
            ("SAT-01", "SAT-06"), ("SAT-02", "SAT-07"), ("SAT-03", "SAT-08"),
            ("SAT-04", "SAT-09"), ("SAT-05", "SAT-10"),
            # Cross diagonals for redundant dynamic paths
            ("SAT-01", "SAT-03"), ("SAT-03", "SAT-06"), ("SAT-02", "SAT-05"),
            ("SAT-04", "SAT-07"), ("SAT-07", "SAT-10"), ("SAT-08", "SAT-10")
        ]

        for u, v in base_edges:
            self._update_edge_metrics(u, v)

    def _calculate_distance(self, u: str, v: str) -> float:
        """Computes Euclidean distance between satellites in km."""
        p1 = self.satellites[u]
        p2 = self.satellites[v]
        dx = p1["x"] - p2["x"]
        dy = p1["y"] - p2["y"]
        # Scale to typical LEO distance (approx 900 to 4200 km)
        pixel_dist = math.hypot(dx, dy)
        dist_km = 800.0 + (pixel_dist * 8.5)
        return round(dist_km, 1)

    def _update_edge_metrics(self, u: str, v: str):
        """Calculates dynamic channel telemetry for link (u, v)."""
        dist_km = self._calculate_distance(u, v)
        load_u = self.satellites[u]["cpu_load"]
        load_v = self.satellites[v]["cpu_load"]
        queue_u = self.satellites[u]["queue_length"]
        queue_v = self.satellites[v]["queue_length"]

        # Propagation delay: light in vacuum ~3.33 us/km
        prop_delay = dist_km / 280.0
        queue_delay = (queue_u + queue_v) * 0.35
        latency_ms = round(prop_delay + queue_delay + random.uniform(1.0, 4.0), 2)

        # Signal attenuation
        rel_vel = round(random.uniform(0.5, 6.5), 2)
        signal_dbm = round(-45.0 - (dist_km / 130.0) - (rel_vel * 1.2) + random.uniform(-2.0, 2.0), 2)
        snr_factor = max(0.1, min(1.0, (signal_dbm + 90.0) / 50.0))

        # Packet loss influenced by signal, queue, and velocity
        loss_pct = round(max(0.1, min(8.0, (1.0 - snr_factor) * 3.0 + ((queue_u + queue_v) / 60.0) * 1.8 + random.uniform(-0.2, 0.4))), 2)

        # Energy consumption (Joules)
        energy_j = round(max(1.1, min(9.5, 1.2 + (dist_km / 4000.0) * 4.5 + (1.0 - snr_factor) * 1.5 + random.uniform(-0.2, 0.3))), 2)

        # Bandwidth
        bandwidth_mbps = round(max(80.0, 1000.0 * snr_factor - (queue_u + queue_v) * 2.0), 1)

        # Stability
        stability = round(max(0.2, min(0.98, 0.95 - (rel_vel / 7.0) * 0.25 - (loss_pct / 8.0) * 0.2)), 3)
        link_quality = round(max(5.0, min(99.0, (snr_factor * 45.0) + (stability * 45.0) - (loss_pct * 3.0))), 1)

        # Link status category
        if latency_ms > 70.0:
            status = "high_latency"
        elif loss_pct > 2.5:
            status = "weak"
        elif load_u > 80.0 or load_v > 80.0:
            status = "congested"
        else:
            status = "optimal"

        self.graph.add_edge(
            u, v,
            distance_km=dist_km,
            latency_ms=latency_ms,
            signal_strength=signal_dbm,
            bandwidth_mbps=bandwidth_mbps,
            packet_loss_percent=loss_pct,
            energy_consumption=energy_j,
            link_quality=link_quality,
            relative_velocity=rel_vel,
            connection_stability=stability,
            status=status
        )

    def advance_simulation(self) -> Dict[str, Any]:
        """
        Advances orbital time step:
        Rotates satellites along orbits, updates link conditions, alters telemetry.
        """
        self.simulation_step += 1
        angle_delta = math.radians(12.0)  # Orbit advance per tick

        for sat_id, sat in self.satellites.items():
            # Plane A orbits counter-clockwise, Plane B slightly faster
            speed_mult = 1.0 if sat["plane"] == "A" else 1.15
            sat["current_angle"] = (sat["current_angle"] + angle_delta * speed_mult) % (2 * math.pi)

            # Random telemetry drift
            sat["cpu_load"] = round(max(10.0, min(95.0, sat["cpu_load"] + random.uniform(-10.0, 12.0))), 1)
            sat["queue_length"] = max(2, min(100, int(sat["queue_length"] + random.randint(-6, 8))))
            sat["battery_percent"] = round(max(75.0, min(100.0, sat["battery_percent"] + random.uniform(-0.5, 0.4))), 1)

        self._update_coordinates()

        # Update edge attributes with new distances and dynamic channel conditions
        for u, v in self.graph.edges():
            self._update_edge_metrics(u, v)

        # Inject an occasional orbital perturbation / link surge for dynamic rerouting demonstration
        if self.simulation_step % 3 == 0:
            surge_edge = random.choice(list(self.graph.edges()))
            u, v = surge_edge
            self.graph[u][v]["latency_ms"] = round(self.graph[u][v]["latency_ms"] + 65.0, 2)
            self.graph[u][v]["packet_loss_percent"] = round(min(8.5, self.graph[u][v]["packet_loss_percent"] + 3.8), 2)
            self.graph[u][v]["status"] = "congested"

        return {
            "step": self.simulation_step,
            "status_message": f"Constellation orbit advanced (Epoch {self.simulation_step:03d}). Link channel states refreshed."
        }

    def get_network_topology(self) -> Dict[str, Any]:
        """Returns JSON-serializable network topology for frontend visualization."""
        nodes = []
        for sat_id, sat in self.satellites.items():
            nodes.append({
                "id": sat_id,
                "name": sat["name"],
                "plane": sat["plane"],
                "x": sat["x"],
                "y": sat["y"],
                "cpu_load": sat["cpu_load"],
                "queue_length": sat["queue_length"],
                "battery_percent": sat["battery_percent"],
                "altitude_km": sat["altitude_km"],
                "velocity_kms": sat["velocity_kms"]
            })

        edges = []
        for u, v, data in self.graph.edges(data=True):
            edges.append({
                "source": u,
                "target": v,
                "distance_km": data.get("distance_km", 1500.0),
                "latency_ms": data.get("latency_ms", 30.0),
                "packet_loss_percent": data.get("packet_loss_percent", 0.5),
                "bandwidth_mbps": data.get("bandwidth_mbps", 500.0),
                "energy_consumption": data.get("energy_consumption", 3.0),
                "link_quality": data.get("link_quality", 80.0),
                "status": data.get("status", "optimal")
            })

        # Calculate network summary stats
        avg_latency = round(float(sum(e["latency_ms"] for e in edges) / max(1, len(edges))), 2)
        avg_loss = round(float(sum(e["packet_loss_percent"] for e in edges) / max(1, len(edges))), 2)
        avg_energy = round(float(sum(e["energy_consumption"] for e in edges) / max(1, len(edges))), 2)
        avg_load = round(float(sum(n["cpu_load"] for n in nodes) / max(1, len(nodes))), 1)

        return {
            "satellites_count": len(nodes),
            "links_count": len(edges),
            "avg_latency_ms": avg_latency,
            "avg_packet_loss_percent": avg_loss,
            "avg_energy_usage_j": avg_energy,
            "avg_cpu_load_percent": avg_load,
            "simulation_step": self.simulation_step,
            "nodes": nodes,
            "edges": edges
        }

    def aggregate_path_metrics(self, path: List[str]) -> Dict[str, Any]:
        """
        Aggregates multi-hop telemetry across all links and intermediate nodes in a candidate path.
        """
        if len(path) < 2:
            return {}

        hops = len(path) - 1
        total_dist = 0.0
        total_latency = 0.0
        total_energy = 0.0
        min_bw = float("inf")
        sum_snr = 0.0
        sum_quality = 0.0
        sum_stability = 0.0
        sum_rel_vel = 0.0

        # Success probability for multi-hop packet loss: P_total = 1 - product(1 - p_i)
        p_success = 1.0

        for i in range(hops):
            u, v = path[i], path[i + 1]
            edge = self.graph.get_edge_data(u, v, default={})
            dist = edge.get("distance_km", 1800.0)
            lat = edge.get("latency_ms", 35.0)
            loss = edge.get("packet_loss_percent", 1.0) / 100.0
            energy = edge.get("energy_consumption", 3.0)
            bw = edge.get("bandwidth_mbps", 500.0)
            snr = edge.get("signal_strength", -55.0)
            quality = edge.get("link_quality", 80.0)
            stab = edge.get("connection_stability", 0.85)
            vel = edge.get("relative_velocity", 2.5)

            total_dist += dist
            total_latency += lat
            total_energy += energy
            min_bw = min(min_bw, bw)
            sum_snr += snr
            sum_quality += quality
            sum_stability += stab
            sum_rel_vel += vel
            p_success *= (1.0 - loss)

        # Intermediate node processing and queueing
        avg_node_load = sum(self.satellites[n]["cpu_load"] for n in path) / len(path)
        avg_queue_len = sum(self.satellites[n]["queue_length"] for n in path) / len(path)

        cum_loss_pct = round((1.0 - p_success) * 100.0, 2)

        return {
            "source_satellite": path[0],
            "destination_satellite": path[-1],
            "distance_km": round(total_dist, 1),
            "signal_strength": round(sum_snr / hops, 2),
            "bandwidth_mbps": round(min_bw if min_bw != float("inf") else 500.0, 1),
            "latency_ms": round(total_latency, 2),
            "packet_loss_percent": cum_loss_pct,
            "energy_consumption": round(total_energy, 2),
            "link_quality": round(sum_quality / hops, 1),
            "satellite_load": round(avg_node_load, 1),
            "queue_length": int(round(avg_queue_len)),
            "relative_velocity": round(sum_rel_vel / hops, 2),
            "connection_stability": round(sum_stability / hops, 3),
            "number_of_hops": hops,
            "path": path,
            "path_string": " → ".join(path)
        }

    def find_all_candidate_routes(self, source: str, destination: str, max_hops: int = 5, max_routes: int = 8) -> List[Dict[str, Any]]:
        """Finds distinct simple paths between source and destination and calculates their parameters."""
        if source not in self.graph or destination not in self.graph:
            return []

        try:
            paths = list(nx.all_simple_paths(self.graph, source=source, target=destination, cutoff=max_hops))
        except Exception:
            paths = []

        if not paths:
            # Fallback to shortest path if cutoff was too tight
            try:
                paths = [nx.shortest_path(self.graph, source, destination)]
            except Exception:
                return []

        # Sort candidate paths by length and take up to max_routes
        paths.sort(key=lambda p: len(p))
        selected_paths = paths[:max_routes]

        candidates = []
        for p in selected_paths:
            cand = self.aggregate_path_metrics(p)
            candidates.append(cand)

        return candidates

    def compute_optimal_routes(self, source: str, destination: str) -> Dict[str, Any]:
        """
        Executes comprehensive comparison across:
        1. Random Forest ML-Driven Route Selection
        2. Traditional Dijkstra Shortest Path (Physical distance only)
        3. Q-Learning Reinforcement Learning Route
        """
        from ml.predict import predict_route_viability
        from ml.reinforcement_learning import get_rl_agent

        candidates = self.find_all_candidate_routes(source, destination)
        if not candidates:
            return {"error": f"No connected paths found between {source} and {destination}"}

        # Evaluate all candidate paths with Random Forest ML Model
        evaluated_candidates = []
        for cand in candidates:
            ml_res = predict_route_viability(cand)
            eval_record = {
                **cand,
                "ml_prediction": ml_res,
                "is_optimal": ml_res["is_optimal"],
                "optimal_probability": ml_res["optimal_probability"],
                "routing_score": ml_res["routing_score"],
                "route_quality": ml_res["route_quality"],
                "recommendation": ml_res["recommendation"],
                "status_label": ml_res["status_label"],
                "badge_color": ml_res["badge_color"],
                "cost_breakdown": ml_res["cost_breakdown"]
            }
            evaluated_candidates.append(eval_record)

        # Select ML Optimal Route: minimum combined routing score (with high probability)
        evaluated_candidates.sort(key=lambda x: (x["routing_score"], -x["optimal_probability"]))
        ml_optimal = evaluated_candidates[0]

        # 2. Traditional Dijkstra (weighted by physical distance only)
        try:
            dijkstra_path = nx.shortest_path(self.graph, source=source, target=destination, weight="distance_km")
            dijkstra_metrics = self.aggregate_path_metrics(dijkstra_path)
            dijkstra_ml = predict_route_viability(dijkstra_metrics)
            dijkstra_result = {
                **dijkstra_metrics,
                "routing_score": dijkstra_ml["routing_score"],
                "route_quality": dijkstra_ml["route_quality"],
                "ml_prediction": dijkstra_ml
            }
        except Exception:
            dijkstra_result = ml_optimal

        # 3. Reinforcement Learning Route
        try:
            rl_agent, rl_env = get_rl_agent()
            rl_trace = rl_agent.get_route_trace(source, destination, rl_env)
            rl_path = rl_trace["path"]
            if rl_trace["success"] and len(rl_path) >= 2:
                rl_metrics = self.aggregate_path_metrics(rl_path)
                rl_ml = predict_route_viability(rl_metrics)
                rl_result = {
                    **rl_metrics,
                    "routing_score": rl_ml["routing_score"],
                    "route_quality": rl_ml["route_quality"],
                    "ml_prediction": rl_ml,
                    "transitions": rl_trace.get("transitions", [])
                }
            else:
                rl_result = ml_optimal
        except Exception as e:
            rl_result = {**ml_optimal, "rl_error": str(e)}

        # Check for route change from previous simulation step
        route_changed = (self.previous_optimal_route != ml_optimal["path"])
        prev_route_str = " → ".join(self.previous_optimal_route) if self.previous_optimal_route else "None"
        self.previous_optimal_route = list(ml_optimal["path"])

        return {
            "source": source,
            "destination": destination,
            "optimal_route": ml_optimal,
            "dijkstra_route": dijkstra_result,
            "rl_route": rl_result,
            "all_candidates": evaluated_candidates,
            "route_changed": route_changed,
            "previous_route": prev_route_str,
            "current_route": ml_optimal["path_string"]
        }

# Global constellation singleton instance
_constellation_instance = None

def get_constellation() -> SatelliteConstellation:
    global _constellation_instance
    if _constellation_instance is None:
        _constellation_instance = SatelliteConstellation()
    return _constellation_instance
