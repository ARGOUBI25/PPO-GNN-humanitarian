import time, pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import pathlib
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from ppo_gnn.env import HumanitarianVRPFixed  # Vérifiez que ce module existe
from torch_geometric.nn import GCNConv

# ========== UTILITAIRES OBSERVATION / ACTION ==========

def ensure_tensor(obs, device="cpu"):
    """Force l'obs en torch.Tensor, float32, sur device, gère array ou scalar."""
    if isinstance(obs, torch.Tensor):
        return obs.float().to(device)
    obs = np.asarray(obs)
    return torch.tensor(obs, dtype=torch.float32, device=device)

def adapt_obs(obs, nb_nodes):
    """Reshape obs en (nb_nodes, nb_features) de façon robuste."""
    obs = np.asarray(obs)
    if obs.ndim == 1:
        nb_features = obs.size // nb_nodes
        obs = obs.reshape(nb_nodes, nb_features)
    elif obs.ndim == 2:
        pass  # déjà bon
    elif obs.ndim == 3 and obs.shape[0] == 1:
        obs = obs[0]
    else:
        raise ValueError(f"Observation shape inconnue : {obs.shape}")
    return obs

def safe_action(action):
    """Assure qu'une action est un int, même si array, tuple, scalar."""
    if isinstance(action, (tuple, list, np.ndarray)):
        action = action[0]
    if isinstance(action, np.generic):
        action = np.asscalar(action)
    return int(action)

# ========== TRAINING PPO "REAL" (MLP) ==========

def train_ppo_real(graph_path: str, seed: int, episodes: int = 5_000) -> dict:
    with open(graph_path, "rb") as f:
        G = pickle.load(f)

    print(f"Training PPO with {episodes} episodes...")

    env = DummyVecEnv([lambda: HumanitarianVRPFixed(G, seed=seed)])

    # Extraire l'instance réelle d'env pour lire les coefficients
    real_env = HumanitarianVRPFixed(G, seed=seed)
    print("Training parameters from environment:")
    print(f"  alpha (distance weight): {real_env.alpha}")
    print(f"  beta (risk weight): {real_env.beta}")
    print(f"  gamma (unmet demand weight): {real_env.gamma}")
    print(f"  delta (capacity violation weight): {real_env.delta}")
    print(f"  max steps per episode: {real_env.max_steps}")
    print(f"  vehicle capacity: {real_env.capacity}")

    model = PPO(
        "MlpPolicy", env,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        seed=seed,
        verbose=1,
    )

    # Afficher nombre total de paramètres du modèle
    total_params = sum(p.numel() for p in model.policy.parameters())
    print(f"Model total number of parameters: {total_params}")

    t0 = time.perf_counter()
    model.learn(total_timesteps=episodes)
    elapsed = time.perf_counter() - t0

    metrics = evaluate_policy(model, G, seed, method="ppo")
    metrics["time"] = elapsed

    print(f"PPO training completed in {elapsed:.1f}s")
    return metrics


# ========== EXTRACTEUR GNN ==========

class GCNExtractor(BaseFeaturesExtractor):
    """
    Extracteur de features utilisant des Graph Convolutional Networks.
    Robuste aux obs plates, convertit et reshape automatiquement.
    """
    def __init__(self, observation_space, features_dim=128, edge_index=None):
        super().__init__(observation_space, features_dim)
        assert edge_index is not None, "edge_index cannot be None"
        self.register_buffer('edge_index', edge_index)

        # Dimension d'entrée (si plat)
        in_shape = observation_space.shape
        if len(in_shape) == 2:
            self.nb_nodes, in_feats = in_shape
        else:
            # On force à plat
            self.nb_nodes = in_shape[0] if len(in_shape) == 1 else None
            in_feats = 6  # valeur par défaut, à adapter!
        self.in_feats = in_feats

        self.conv1 = GCNConv(in_feats, 64)
        self.conv2 = GCNConv(64, 64)
        self.conv3 = GCNConv(64, 64)
        self.lin = nn.Linear(64, features_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        print(f"[GCNExtractor] forward x: {type(x)}, shape: {getattr(x, 'shape', 'no shape')}")
        # Conversion auto en tensor
        x = ensure_tensor(x, device=self.edge_index.device)

        # Si batché (DummyVecEnv), x peut être (B, N*F) ou (B, N, F)
        if x.dim() == 2:
            if x.shape[1] == self.nb_nodes * self.in_feats:
                x = x.view(-1, self.nb_nodes, self.in_feats)
        elif x.dim() == 3 and x.shape[0] == 1:
            x = x[0]
        elif x.dim() == 1:
            x = x.view(self.nb_nodes, self.in_feats)

        # On force (batch, N, F)
        if x.dim() == 2:
            x = x.unsqueeze(0)
        batch_size, N, features = x.shape
        out = []
        for i in range(batch_size):
            hi = F.relu(self.conv1(x[i], self.edge_index))
            hi = self.dropout(hi)
            hi = F.relu(self.conv2(hi, self.edge_index))
            hi = self.dropout(hi)
            hi = F.relu(self.conv3(hi, self.edge_index))
            hi = hi.mean(dim=0)
            out.append(hi)
        out = torch.stack(out, dim=0)
        return self.lin(out)

# ========== TRAINING PPO-GNN ==========

def train_ppo_gnn_real(graph_path: str, seed: int, episodes: int = 5_000) -> dict:
    """Entraîne un agent PPO avec GNN sur le VRP humanitaire"""
    with open(graph_path, "rb") as f:
        G = pickle.load(f)

    print(f"Training PPO-GNN with {episodes} episodes...")

    base_env = HumanitarianVRPFixed(G, seed=seed)
    env = DummyVecEnv([lambda: HumanitarianVRPFixed(G, seed=seed)])

    ei = torch.tensor(base_env.edge_index, dtype=torch.long)
    print(f"Graph structure: {ei.shape[1]} edges")

    # Afficher les coefficients lambda (params de la fonction de récompense)
    print("Training parameters from environment:")
    print(f"  alpha (distance weight): {base_env.alpha}")
    print(f"  beta (risk weight): {base_env.beta}")
    print(f"  gamma (unmet demand weight): {base_env.gamma}")
    print(f"  delta (capacity violation weight): {base_env.delta}")
    print(f"  max steps per episode: {base_env.max_steps}")
    print(f"  vehicle capacity: {base_env.capacity}")

    # Configuration de la politique avec extracteur GCN
    features_dim = 128
    policy_kwargs = dict(
        features_extractor_class=GCNExtractor,
        features_extractor_kwargs=dict(
            edge_index=ei, features_dim=features_dim
        )
    )

    model = PPO("MlpPolicy", env,
                policy_kwargs=policy_kwargs,
                learning_rate=3e-4,
                n_steps=512,
                batch_size=64,
                n_epochs=4,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                seed=seed,
                verbose=1)

    # Calculer et afficher le nombre total de paramètres du modèle
    total_params = sum(p.numel() for p in model.policy.parameters())
    print(f"Model total number of parameters: {total_params}")

    t0 = time.perf_counter()
    model.learn(total_timesteps=episodes)
    elapsed = time.perf_counter() - t0

    metrics = evaluate_policy(model, G, seed, method="ppo_gnn")
    metrics["time"] = elapsed

    print(f"PPO-GNN training completed in {elapsed:.1f}s")
    return metrics


# ========== EVALUATION POLITIQUE ==========

def evaluate_policy(model, graph, seed, method="ppo", n_test=5):
    """Évalue la performance du modèle entraîné et sauvegarde les routes parcourues."""
    print(f"Evaluating policy with {n_test} test episodes...")

    env = HumanitarianVRPFixed(graph, seed=seed)
    acc = {"cost": 0, "risk": 0, "unmet": 0, "viol": 0}
    all_routes = []

    for episode in range(n_test):
        reset = env.reset(seed=seed + episode)
        if isinstance(reset, tuple):
            obs = reset[0]
        else:
            obs = reset
        done = False
        cost = risk = unmet = viol = 0
        steps = 0
        route = []

        while not done and steps < getattr(env, 'max_steps', 500):
            print(f"Episode {episode} Step {steps}: obs type={type(obs)}, shape={getattr(obs, 'shape', 'no shape')}")
            try:
                action = model.predict(obs, deterministic=True)
                # Peut être (array, info) ou juste array
                if isinstance(action, tuple):
                    action = action[0]
                action_idx = safe_action(action)
            except Exception as ex:
                print(f"[ERROR] Action extraction failed: {ex}")
                action_idx = 0  # fallback (choix par défaut)

            print(f"Episode {episode} Step {steps}: action={action_idx}")

            try:
                step = env.step(action_idx)
                if len(step) == 5:
                    obs, _, done, _, info = step
                else:
                    obs, _, done, info = step
            except Exception as ex:
                print(f"[ERROR] Step failed: {ex}")
                break

            route.append(action_idx)

            cost += info.get("dist", 0)
            risk += info.get("risk_seg", 0)
            unmet += info.get("unmet_seg", 0)
            viol += info.get("viol", 0)
            steps += 1

        all_routes.append(route)
        acc["cost"] += cost
        acc["risk"] += risk
        acc["unmet"] += unmet
        acc["viol"] += viol

    for k in acc:
        acc[k] /= n_test

    print(f"Evaluation results: cost={acc['cost']:.1f}, risk={acc['risk']:.2f}, "
          f"unmet={acc['unmet']:.2f}, violations={acc['viol']:.1f}")

    routes_dir = pathlib.Path("results/routes")
    routes_dir.mkdir(parents=True, exist_ok=True)
    routes_file = routes_dir / f"routes_{method}_seed{seed}.pkl"
    with open(routes_file, "wb") as f:
        pickle.dump(all_routes, f)

    acc["routes"] = all_routes
    return acc
