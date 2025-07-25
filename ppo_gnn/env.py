import gymnasium as gym
from gymnasium import spaces
import numpy as np
import networkx as nx

def _edge_index_from_graph(G):
    rows = []
    for u, v in G.edges:
        rows.append([G.nodes_idx[u], G.nodes_idx[v]])
        rows.append([G.nodes_idx[v], G.nodes_idx[u]])
    return np.array(rows).T

class HumanitarianVRPFixed(gym.Env):
    metadata = {"render_modes": []}
    def __init__(self, graph, capacity=1000, seed=0,
                 step_max=None, alpha=1.0, beta=100.0,
                 gamma=10.0, delta=50.0):
        super().__init__()
        self.G_raw = graph
        self.capacity, self.alpha, self.beta = capacity, alpha, beta
        self.gamma, self.delta = gamma, delta
        self.rng = np.random.default_rng(seed)

        self.nodes = list(self.G_raw.nodes)
        self.N = len(self.nodes)
        self.G = self.G_raw.copy()
        self.G.nodes_idx = {}
        for i, n in enumerate(self.nodes):
            self.G.nodes[n]["idx"] = i
            self.G.nodes_idx[n] = i

        self.edge_index = _edge_index_from_graph(self.G).astype(np.int64)
        self.action_space = spaces.Discrete(self.N)
        # obs: [N, 4] node features, [N] pos_onehot, [N] mask
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.N * 6,), dtype=np.float32
        )
        self.max_steps = step_max or 2 * self.N
        self.reset()

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.step_count = 0
        self.current = 0
        self.visited = np.zeros(self.N, bool)
        self.demand = np.array([
            max(0.0, self.rng.normal(self.G.nodes[n]["mu"], self.G.nodes[n]["sigma"]))
            for n in self.nodes
        ], dtype=np.float32)
        self.load = self.total_distance = self.total_risk = 0.0
        self.total_viol = 0
        return self._build_obs(), {}

    def step(self, action):
        self.step_count += 1
        u = self.nodes[self.current]
        v = self.nodes[action]

        # 1. Masquage d'action: interdit de rester sur place
        if action == self.current:
            reward = -1000
            dist = risk = 0.0
            done = True
            viol = 1
            # print(f"Action invalide : rester sur place (step terminé, pénalité -1000)")
            return self._build_obs(), reward, done, False, {
                "dist": dist,
                "risk_seg": risk,
                "unmet_seg": self.demand[action],
                "viol": viol
            }

        # 2. Revisite d’un noeud déjà livré (demandes nulles ou presque)
        revisite = self.visited[action]
        revisite_penalty = -100 if revisite else 0

        # 3. Chemin et coût
        if u == v:
            dist = risk = 0.0
        else:
            try:
                dist = nx.shortest_path_length(self.G, u, v, weight="distance")
                path = nx.shortest_path(self.G, u, v, weight="distance")
                risk = sum(self.G.edges[path[i], path[i + 1]]["risk"] for i in range(len(path) - 1))
            except nx.NetworkXNoPath:
                # Action impossible (pas de chemin): pénalité forte
                reward = -1000
                done = True
                viol = 1
                # print(f"Action invalide : pas de chemin (step terminé, pénalité -1000)")
                return self._build_obs(), reward, done, False, {
                    "dist": 0.0,
                    "risk_seg": 0.0,
                    "unmet_seg": self.demand[action],
                    "viol": viol
                }

        self.total_distance += dist
        self.total_risk += risk

        delivered = min(self.capacity - self.load, self.demand[action])
        self.load += delivered
        self.demand[action] -= delivered

        penalty_no_demand = 10.0 if self.demand[action] < 1e-3 else 0.0

        viol = int(self.load > self.capacity + 1e-6)
        self.total_viol += viol
        self.visited[action] = True
        self.current = action

        reward = -(self.alpha * dist +
                   self.beta * risk +
                   self.gamma * self.demand[action] +
                   self.delta * viol +
                   penalty_no_demand) + revisite_penalty

        done = bool((self.demand <= 1e-3).all() or self.step_count >= self.max_steps)

        # print(f"Step {self.step_count}: current={self.current}, action={action}, delivered={delivered:.2f}, "
        #       f"demand_remaining={self.demand[action]:.2f}, load={self.load:.2f}, reward={reward:.2f}")

        return self._build_obs(), reward, done, False, {
            "dist": dist,
            "risk_seg": risk,
            "unmet_seg": self.demand[action],
            "viol": viol
        }

    def _build_obs(self):
        # Node features
        x = np.zeros((self.N, 4), dtype=np.float32)
        x[:, 0] = self.demand
        x[:, 1] = [self.G.nodes[n]["sigma"] for n in self.nodes]
        x[:, 2] = np.array([self.G.degree[n] for n in self.nodes]) / self.N
        x[:, 3] = self.visited.astype(np.float32)
        # Current position one-hot
        pos_onehot = np.zeros(self.N, dtype=np.float32)
        pos_onehot[self.current] = 1
        # Mask des actions valides
        mask = np.ones(self.N, dtype=np.float32)
        # Interdit de rester sur place
        mask[self.current] = 0
        # Interdit les noeuds inaccessibles depuis la position actuelle
        for j in range(self.N):
            if j == self.current:
                mask[j] = 0
            else:
                u = self.nodes[self.current]
                v = self.nodes[j]
                if not nx.has_path(self.G, u, v):
                    mask[j] = 0
        # Retour obs plat (N*6,)
        obs = np.concatenate([x.flatten(), pos_onehot, mask])
        return obs
