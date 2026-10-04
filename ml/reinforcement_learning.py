"""
OrbitRoute AI - Reinforcement Learning Module
Implements dynamic Q-Learning for multi-objective inter-satellite packet routing.
Demonstrates State-Action-Reward-NextState transitions and policy improvement over episodes.
"""

import os
import random
import joblib
import numpy as np
from typing import Dict, List, Tuple, Any

class SatelliteRoutingEnv:
    """
    Simulated satellite network environment for Q-Learning.
    Nodes: SAT-01 to SAT-10.
    Edges represent dynamic inter-satellite laser/RF communication links.
    """
    def __init__(self, adjacency_matrix: Dict[str, List[str]] = None):
        self.nodes = [f"SAT-{i:02d}" for i in range(1, 11)]
        
        # Default constellation mesh topology (Walker-Delta inspired ring and cross links)
        if adjacency_matrix is not None:
            self.adj = adjacency_matrix
        else:
            self.adj = {
                "SAT-01": ["SAT-02", "SAT-03", "SAT-04"],
                "SAT-02": ["SAT-01", "SAT-03", "SAT-05"],
                "SAT-03": ["SAT-01", "SAT-02", "SAT-06", "SAT-07"],
                "SAT-04": ["SAT-01", "SAT-07", "SAT-08"],
                "SAT-05": ["SAT-02", "SAT-06", "SAT-08", "SAT-09"],
                "SAT-06": ["SAT-03", "SAT-05", "SAT-09", "SAT-10"],
                "SAT-07": ["SAT-03", "SAT-04", "SAT-08", "SAT-10"],
                "SAT-08": ["SAT-04", "SAT-05", "SAT-07", "SAT-10"],
                "SAT-09": ["SAT-05", "SAT-06", "SAT-10"],
                "SAT-10": ["SAT-06", "SAT-07", "SAT-08", "SAT-09"]
            }

    def get_link_metrics(self, u: str, v: str) -> Dict[str, float]:
        """Calculates instantaneous link attributes between two adjacent satellites."""
        u_idx = int(u.split("-")[1])
        v_idx = int(v.split("-")[1])
        diff = abs(u_idx - v_idx)
        
        # Link latency (ms)
        latency = 18.0 + (diff * 12.5) + random.uniform(-3.0, 5.0)
        # Packet loss (%)
        loss = 0.2 + (diff * 0.35) + random.uniform(0.0, 0.5)
        # Energy consumption (Joules)
        energy = 1.2 + (diff * 0.7) + random.uniform(0.0, 0.4)
        # Bandwidth (Mbps)
        bandwidth = max(200.0, 950.0 - (diff * 75.0) + random.uniform(-40.0, 40.0))

        return {
            "latency_ms": round(max(10.0, latency), 2),
            "packet_loss": round(max(0.05, loss), 2),
            "energy_j": round(max(1.0, energy), 2),
            "bandwidth_mbps": round(bandwidth, 1)
        }

    def step(self, current: str, action: str, destination: str, visited: List[str]) -> Tuple[str, float, bool, Dict[str, float]]:
        """
        Takes a step in the environment.
        Returns: (next_state, reward, done, link_metrics)
        """
        if action not in self.adj.get(current, []):
            # Invalid hop penalty
            return current, -60.0, False, {}

        metrics = self.get_link_metrics(current, action)
        
        # Loop prevention penalty
        if action in visited:
            reward = -45.0
            return action, reward, False, metrics

        # Cost penalty based on delay, loss, and energy
        step_penalty = (
            0.35 * (metrics["latency_ms"] * 0.25) +
            0.30 * (metrics["packet_loss"] * 4.0) +
            0.20 * (metrics["energy_j"] * 2.5) +
            5.0  # Hop penalty
        )

        if action == destination:
            # High reward for reaching goal
            reward = 120.0 - step_penalty
            done = True
        else:
            reward = -step_penalty
            done = False

        return action, round(reward, 2), done, metrics


class QLearningRoutingAgent:
    """
    Q-Learning agent for dynamic inter-satellite routing.
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 0.85, epsilon: float = 0.3):
        self.alpha = alpha      # Learning rate
        self.gamma = gamma      # Discount factor
        self.epsilon = epsilon  # Exploration factor
        self.q_table: Dict[str, Dict[str, float]] = {}  # state -> {action: q_value}
        self.training_history: List[Dict[str, Any]] = []

    def _state_key(self, current: str, destination: str) -> str:
        return f"{current}->{destination}"

    def get_q(self, state: str, action: str) -> float:
        return self.q_table.get(state, {}).get(action, 0.0)

    def choose_action(self, current: str, destination: str, neighbors: List[str], explore: bool = True) -> str:
        state = self._state_key(current, destination)
        if not neighbors:
            return destination

        if state not in self.q_table:
            self.q_table[state] = {nbr: 0.0 for nbr in neighbors}

        # Ensure all neighbors exist in state dict
        for nbr in neighbors:
            if nbr not in self.q_table[state]:
                self.q_table[state][nbr] = 0.0

        if explore and random.random() < self.epsilon:
            return random.choice(neighbors)

        # Exploit: pick best Q-value
        best_val = -float("inf")
        best_actions = []
        for nbr in neighbors:
            val = self.q_table[state].get(nbr, 0.0)
            if val > best_val:
                best_val = val
                best_actions = [nbr]
            elif val == best_val:
                best_actions.append(nbr)

        return random.choice(best_actions)

    def update(self, current: str, action: str, reward: float, next_node: str, destination: str, next_neighbors: List[str]):
        state = self._state_key(current, destination)
        next_state = self._state_key(next_node, destination)

        if state not in self.q_table:
            self.q_table[state] = {}
        if action not in self.q_table[state]:
            self.q_table[state][action] = 0.0

        old_val = self.q_table[state][action]

        if next_node == destination:
            max_next = 0.0
        else:
            if next_state not in self.q_table:
                self.q_table[next_state] = {nbr: 0.0 for nbr in next_neighbors}
            max_next = max(self.q_table[next_state].values()) if self.q_table[next_state] else 0.0

        # Bellman equation update
        new_val = old_val + self.alpha * (reward + self.gamma * max_next - old_val)
        self.q_table[state][action] = round(new_val, 3)

    def train(self, env: SatelliteRoutingEnv, episodes: int = 1200):
        print(f"[RL] Training Q-Learning Routing Agent over {episodes} episodes...")
        satellites = env.nodes
        
        episode_rewards = []
        checkpoint_steps = max(1, episodes // 20)

        for ep in range(episodes):
            src, dst = random.sample(satellites, 2)
            current = src
            visited = [src]
            total_reward = 0.0
            done = False
            steps = 0

            # Decay epsilon
            self.epsilon = max(0.05, 0.40 * (1.0 - (ep / episodes)))

            while not done and steps < 12:
                neighbors = env.adj.get(current, [])
                action = self.choose_action(current, dst, neighbors, explore=True)
                next_node, reward, done, _ = env.step(current, action, dst, visited)
                
                next_neighbors = env.adj.get(next_node, [])
                self.update(current, action, reward, next_node, dst, next_neighbors)
                
                visited.append(next_node)
                total_reward += reward
                current = next_node
                steps += 1

            episode_rewards.append(total_reward)

            if (ep + 1) % checkpoint_steps == 0:
                avg_r = np.mean(episode_rewards[-checkpoint_steps:])
                self.training_history.append({
                    "episode": ep + 1,
                    "avg_reward": round(float(avg_r), 2),
                    "epsilon": round(self.epsilon, 3)
                })

        print(f"[RL] Training completed. Final Avg Reward: {self.training_history[-1]['avg_reward']}")
        return self.training_history

    def get_route_trace(self, source: str, destination: str, env: SatelliteRoutingEnv) -> Dict[str, Any]:
        """
        Executes a deterministic step-by-step route selection using learned Q-values.
        Returns detailed transition table: State, Action, Reward, Next State, Q-value.
        """
        current = source
        visited = [current]
        transitions = []
        done = False
        steps = 0
        total_delay = 0.0
        total_loss = 0.0
        total_energy = 0.0

        while not done and steps < 8 and current != destination:
            neighbors = env.adj.get(current, [])
            if not neighbors:
                break

            # Pick greedy action from learned policy
            action = self.choose_action(current, destination, neighbors, explore=False)
            state_str = f"({current}, {destination})"
            q_val = self.get_q(self._state_key(current, destination), action)

            next_node, reward, done, metrics = env.step(current, action, destination, visited)
            
            transitions.append({
                "step": steps + 1,
                "state": state_str,
                "current_node": current,
                "action": action,
                "reward": reward,
                "next_state": f"({next_node}, {destination})",
                "next_node": next_node,
                "q_value": q_val,
                "link_metrics": metrics
            })

            if metrics:
                total_delay += metrics.get("latency_ms", 0.0)
                total_loss += metrics.get("packet_loss", 0.0)
                total_energy += metrics.get("energy_j", 0.0)

            visited.append(next_node)
            current = next_node
            steps += 1

        success = (current == destination)
        return {
            "source": source,
            "destination": destination,
            "path": visited,
            "success": success,
            "hops": len(visited) - 1,
            "total_delay_ms": round(total_delay, 2),
            "total_packet_loss": round(total_loss, 2),
            "total_energy_j": round(total_energy, 2),
            "transitions": transitions
        }

def train_and_save_rl_agent(save_path: str = "models/q_learning_model.pkl") -> QLearningRoutingAgent:
    """Trains and serializes the RL model."""
    env = SatelliteRoutingEnv()
    agent = QLearningRoutingAgent()
    agent.train(env, episodes=1500)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump({
        "q_table": agent.q_table,
        "training_history": agent.training_history,
        "env_adj": env.adj
    }, save_path)
    print(f"[RL] Saved Q-Learning model to {save_path}")
    return agent

_loaded_agent = None

def get_rl_agent(model_path: str = "models/q_learning_model.pkl") -> Tuple[QLearningRoutingAgent, SatelliteRoutingEnv]:
    """Loads existing Q-Learning agent or trains if missing."""
    global _loaded_agent
    env = SatelliteRoutingEnv()
    if _loaded_agent is None:
        if not os.path.exists(model_path):
            _loaded_agent = train_and_save_rl_agent(model_path)
        else:
            data = joblib.load(model_path)
            agent = QLearningRoutingAgent()
            agent.q_table = data.get("q_table", {})
            agent.training_history = data.get("training_history", [])
            _loaded_agent = agent
            if "env_adj" in data:
                env = SatelliteRoutingEnv(adjacency_matrix=data["env_adj"])
    return _loaded_agent, env

if __name__ == "__main__":
    train_and_save_rl_agent()
