#!/usr/bin/env python
"""
End-to-end Afghanistan pipeline – *real implementation template* with scale selection and connectivity validation.
"""

import sys
import pathlib, zipfile, requests, itertools, pickle
import geopandas as gpd, pandas as pd, networkx as nx, shapely.geometry as geom
from geopy.distance import geodesic
import numpy as np, scipy.stats as st
from tqdm import tqdm

from ppo_gnn.train import train_ppo_gnn_real, train_ppo_real
from heuristics.clarke_wright import evaluate_clarke_wright

DATA = pathlib.Path("data")
RAW = DATA / "raw"
PROC = DATA / "proc"
LOGS = pathlib.Path("results/logs")
TABLES = pathlib.Path("results/tables")

GED_URL = "https://ucdp.uu.se/downloads/ged/ged231.csv"
METHODS = ["ppo_gnn", "ppo", "cw"]
SEEDS = [42, 99, 123]
DEMAND_CSV = RAW / "afg_demands15.csv"
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- SCALE = "large" par défaut, modifiable par argument
SCALE = "large"
if len(sys.argv) > 1 and sys.argv[1] in ["small", "medium", "large"]:
    SCALE = sys.argv[1]
else:
    print("Defaulting to 'large' scale (use: python afg_pipeline.py [small|medium|large])")

SCALE_SIZES = {
    "small": (50, 120),
    "medium": (150, 400),
    "large": (500, 2000)
}

N_NODES, N_EDGES = SCALE_SIZES[SCALE]
GRAPH_PKL = PROC / f"afg_graph_{SCALE}.pkl"
MIN_DISTANCE = 0.01  # Espacement minimal entre nœuds


def download_data():
    RAW.mkdir(parents=True, exist_ok=True)
    ged = RAW / "ged_afg.csv"
    if not ged.exists():
        print("⇣ Downloading UCDP GED data...")
        try:
            r = requests.get(GED_URL, timeout=120)
            r.raise_for_status()
            ged.write_bytes(r.content)
            print("✓ GED data downloaded")
        except requests.RequestException as e:
            print(f"Warning: Could not download GED data: {e}")
            # Create dummy data for testing
            dummy_data = pd.DataFrame({
                'year': [2023, 2024],
                'country': ['Afghanistan', 'Afghanistan'],
                'longitude': [69.0, 69.5],
                'latitude': [34.0, 34.5]
            })
            dummy_data.to_csv(ged, index=False)

    shp_file = RAW / "afg_osm_lines_shp" / "afg_roads_lines.shp"
    if not shp_file.exists():
        print("Warning: Shapefile missing. Please ensure roads data is available.")

    if not DEMAND_CSV.exists():
        print("Warning: Demands CSV missing. Creating dummy data for testing.")
        dummy_demands = pd.DataFrame({
            'lon': [69.0, 69.2, 69.4],
            'lat': [34.0, 34.2, 34.4],
            'mu': [10.0, 15.0, 8.0],
            'sigma': [2.0, 3.0, 1.5]
        })
        dummy_demands.to_csv(DEMAND_CSV, index=False)

    print("✓ Data preparation completed")


def build_graph():
    if GRAPH_PKL.exists():
        print(f"✓ Graph for '{SCALE}' scale already built.")
        return

    print("● Building full Afghanistan graph...")

    # Try to find shapefile
    shp_files = list((RAW / "afg_osm_lines_shp").rglob("*.shp")) if (RAW / "afg_osm_lines_shp").exists() else []

    if not shp_files:
        print("Warning: No shapefile found. Creating simple dummy graph.")
        G = nx.Graph()
        coords = []
        last_pt = None
        for i in range(N_NODES):
            # Espace chaque point à au moins MIN_DISTANCE du précédent
            if last_pt is None:
                pt = (69.0, 34.0)
            else:
                pt = (last_pt[0] + MIN_DISTANCE, last_pt[1] + MIN_DISTANCE)
            coords.append(pt)
            last_pt = pt
            G.add_node(pt)
            if i > 0:
                prev_coord = coords[i - 1]
                dist = geodesic((prev_coord[1], prev_coord[0]), (pt[1], pt[0])).km
                G.add_edge(prev_coord, pt, distance=dist, risk=0.1)
    else:
        shp = shp_files[0]
        print(f"Reading shapefile: {shp}")
        roads = gpd.read_file(shp).to_crs(4326)
        if "highway" in roads.columns:
            roads = roads[
                roads["highway"].isin(["primary", "secondary", "tertiary", "unclassified"])
            ]
        if "surface" in roads.columns:
            roads = roads[~roads["surface"].isin(["trail", "unpaved", "dirt", "gravel"])]
        print(f"Processing {len(roads)} road segments...")
        G = nx.Graph()
        for _, r in tqdm(roads.iterrows(), total=len(roads), desc="Building road network"):
            if r.geometry is None or r.geometry.is_empty:
                continue
            try:
                coords = list(r.geometry.coords)
                if len(coords) < 2:
                    continue
                p, q = coords[0], coords[-1]
                d = geodesic((p[1], p[0]), (q[1], q[0])).km
                G.add_edge((p[0], p[1]), (q[0], q[1]), distance=d, risk=0.0)
            except Exception as e:
                print(f"Warning: Skipping invalid geometry: {e}")
                continue

    # Add risk data from events
    events_file = RAW / "ged_afg.csv"
    if events_file.exists():
        try:
            events = pd.read_csv(events_file).query("year>=2023")
            if 'country' in events.columns:
                events = events.query("country=='Afghanistan'")

            if len(events) > 0 and 'longitude' in events.columns and 'latitude' in events.columns:
                ev = gpd.GeoDataFrame(events,
                                      geometry=gpd.points_from_xy(events.longitude, events.latitude),
                                      crs=4326)
                print("Adding risk data to edges...")
                for u, v, d in tqdm(G.edges(data=True), desc="Computing risk"):
                    try:
                        buf = geom.LineString([u, v]).buffer(0.1)
                        risk_count = ev.within(buf).sum() if len(ev) > 0 else 0
                        d["risk"] = risk_count / (buf.area + 1e-9)
                    except Exception:
                        d["risk"] = 0.0
        except Exception as e:
            print(f"Warning: Could not process events data: {e}")

    # Add demand data
    if DEMAND_CSV.exists():
        try:
            dem = pd.read_csv(DEMAND_CSV)
            print("Adding demand data to nodes...")
            assigned = []
            for _, row in dem.iterrows():
                pt = geom.Point(row.lon, row.lat)
                nearest = min(G.nodes, key=lambda n: pt.distance(geom.Point(n)))
                assigned.append(nearest)
                G.nodes[nearest]["mu"] = float(row.mu)
                G.nodes[nearest]["sigma"] = float(row.sigma)
                print(f"Centre {row.get('centre_id', '?')} ({row.lat},{row.lon}) -> nœud {nearest}")

            print(f"Nombre de nœuds affectés : {len(set(assigned))} / {len(list(G.nodes))}")
            print("Répartition des demandes (mu) :")
            count = 0
            for i, n in enumerate(G.nodes):
                mu = G.nodes[n].get("mu", 0)
                if mu > 0:
                    count += 1
                    print(f"Node {i}: {n}, mu={mu}, sigma={G.nodes[n].get('sigma', 0)}")
            print("Nombre total de nœuds avec mu > 0 :", count)
        except Exception as e:
            print(f"Warning: Could not process demand data: {e}")

    # Set default values for nodes without demand data
    for n in G.nodes:
        G.nodes[n].setdefault("mu", 1.0)
        G.nodes[n].setdefault("sigma", 0.3)

    # --- Extraction du sous-graphe connexe avec validation et espacement ---
    G = build_connected_subgraph(G, N_NODES, N_EDGES, min_distance=MIN_DISTANCE)
    print(f"✓ Connected subgraph extracted: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # --- Sauvegarde du graphe ---
    PROC.mkdir(parents=True, exist_ok=True)
    with open(GRAPH_PKL, "wb") as f:
        pickle.dump(G, f)
    print(f"✓ Graph saved to {GRAPH_PKL}")

def build_connected_subgraph(G, num_nodes, num_edges, min_distance=0.01):
    import random

    # 1) Choisir un centre "central"
    degree_centrality = nx.degree_centrality(G)
    if degree_centrality:
        center = max(degree_centrality, key=degree_centrality.get)
    else:
        center = random.choice(list(G.nodes))

    # 2) BFS jusqu'à atteindre au moins num_nodes
    nodes = []
    depth_limit = 10
    while len(nodes) < num_nodes and depth_limit <= 200:
        nodes = list(nx.bfs_tree(G, center, depth_limit=depth_limit))
        depth_limit += 10

    nodes = nodes[:num_nodes]
    subG = G.subgraph(nodes).copy()

    # 3) Ajouter des arêtes pour atteindre le nombre désiré d'arêtes
    while subG.number_of_edges() < num_edges:
        potential_edges = [e for e in G.edges(nodes) if not subG.has_edge(*e)]
        if not potential_edges:
            break
        # Ne pas ajouter d'arête trop courte
        potential_edges = [
            e for e in potential_edges
            if geodesic((e[0][1], e[0][0]), (e[1][1], e[1][0])).km >= min_distance
        ]
        if not potential_edges:
            break
        potential_edges = sorted(potential_edges, key=lambda e: G.edges[e].get("distance", 1e6))
        to_add = potential_edges[:num_edges - subG.number_of_edges()]
        subG.add_edges_from(to_add)

    # 4) Vérifier la connectivité, relier si besoin
    if not nx.is_connected(subG):
        print("Warning: Subgraph not connected, attempting to connect components.")
        components = list(nx.connected_components(subG))
        for i in range(len(components) - 1):
            comp1 = components[i]
            comp2 = components[i + 1]
            possible_edges = [
                (u, v) for u in comp1 for v in comp2
                if G.has_edge(u, v) and geodesic((u[1], u[0]), (v[1], v[0])).km >= min_distance
            ]
            if possible_edges:
                edge_to_add = min(possible_edges, key=lambda e: G.edges[e].get("distance", 1e6))
                subG.add_edge(*edge_to_add, **G.edges[edge_to_add])
        # Extraction du plus grand composant si pas encore connexe
        if not nx.is_connected(subG):
            largest_cc = max(nx.connected_components(subG), key=len)
            subG = subG.subgraph(largest_cc).copy()
            print("Warning: Could not fully connect, using largest component.")

    return subG

def train_ppo_gnn(graph, seed):
    return train_ppo_gnn_real(graph_path=str(GRAPH_PKL), seed=seed)

def train_ppo(graph, seed):
    return train_ppo_real(graph_path=str(GRAPH_PKL), seed=seed)

def clarke_wright(graph, seed):
    return evaluate_clarke_wright(graph_path=str(GRAPH_PKL))

TRAINERS = {"ppo_gnn": train_ppo_gnn,
            "ppo": train_ppo,
            "cw": clarke_wright}

def run_all():
    LOGS.mkdir(parents=True, exist_ok=True)
    ROUTES_DIR = pathlib.Path("results/routes")
    ROUTES_DIR.mkdir(parents=True, exist_ok=True)

    with open(GRAPH_PKL, "rb") as f:
        G = pickle.load(f)

    print(f"Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print("Graph connectivity check:")
    print("Is graph connected?", nx.is_connected(G))
    print("Number of connected components:", nx.number_connected_components(G))
    nodes = list(G.nodes)
    if len(nodes) > 39:
        print("Node at index 39:", nodes[39])
        print("Degree of node 39:", G.degree[nodes[39]])
        print("Neighbors of node 39:", list(G.neighbors(nodes[39])))

    rows = []
    for m, s in itertools.product(METHODS, SEEDS):
        print(f"▶ Running {m} with seed={s}")
        try:
            metrics = TRAINERS[m](G, s)
            print(f"Metrics for {m} seed={s}: {metrics}")
            if "routes" in metrics and metrics["routes"]:
                print(f"Exemple de route {m} seed={s}:", metrics["routes"][0])

            row = dict(method=m, seed=s, **metrics)
            rows.append(row)
            pd.DataFrame([row]).to_csv(LOGS / f"{m}_{s}.csv", index=False)
            print(f"✓ {m} seed={s} completed")

            # --- Sauvegarder les routes dans un fichier pickle dans results/routes ---
            routes = metrics.get("routes")
            if routes is not None:
                routes_path = ROUTES_DIR / f"routes_{m}_{s}.pkl"
                with open(routes_path, "wb") as f_routes:
                    pickle.dump(routes, f_routes)
                print(f"✓ Routes saved to {routes_path}")

        except Exception as e:
            print(f"✗ {m} seed={s} failed: {e}")
            row = dict(method=m, seed=s, cost=np.inf, risk=np.inf,
                       unmet=100, time=0, viol=100)
            rows.append(row)

    return pd.DataFrame(rows)

def aggregate(df):
    TABLES.mkdir(parents=True, exist_ok=True)
    mets = ["cost", "risk", "unmet", "time", "viol"]
    summary = {}

    for m in mets:
        try:
            g = df[df.method == "ppo_gnn"][m]
            p = df[df.method == "ppo"][m]
            c = df[df.method == "cw"][m]

            summary[m] = {
                "PPO-GNN": f"{g.mean():.1f} ± {g.std():.1f}",
                "PPO": f"{p.mean():.1f} ± {p.std():.1f}",
                "CW": f"{c.mean():.1f} ± {c.std():.1f}",
                "p_vs_PPO": st.ttest_rel(g, p).pvalue if len(g) > 1 else np.nan,
                "p_vs_CW": st.ttest_rel(g, c).pvalue if len(g) > 1 else np.nan
            }
        except Exception as e:
            print(f"Warning: Could not compute statistics for {m}: {e}")
            summary[m] = {"PPO-GNN": "N/A", "PPO": "N/A", "CW": "N/A",
                          "p_vs_PPO": np.nan, "p_vs_CW": np.nan}

    pd.DataFrame(summary).T.to_csv(TABLES / f"table3_afg_{SCALE}.csv")
    df.to_csv(TABLES / f"tableS4_afg_{SCALE}.csv", index=False)
    print("✓ Results aggregated and saved")


if __name__ == "__main__":
    for d in (DATA, LOGS, TABLES):
        d.mkdir(parents=True, exist_ok=True)

    try:
        download_data()
        build_graph()
        df = run_all()
        aggregate(df)
        print("✓ Pipeline completed successfully")
    except Exception as e:
        print(f"✗ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()

