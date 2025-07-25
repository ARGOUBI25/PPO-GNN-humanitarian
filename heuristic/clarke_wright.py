import time
import networkx as nx
import numpy as np
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
import pickle
import random

def evaluate_clarke_wright(graph_path: str, vehicle_cap=1_000, depot_idx=0, max_vehicles=None, seed=0) -> dict:
    """Heuristique Clarke-Wright via OR-Tools CVRP Savings, version ultra-robuste et traçable."""

    print("Starting Clarke-Wright heuristic evaluation...")

    # --- Chargement du graphe ---
    with open(graph_path, "rb") as f:
        G = pickle.load(f)

    # --- Extraction d'une composante connexe ---
    if not nx.is_connected(G):
        print("Graph not connected: extracting largest connected component.")
        components = sorted(nx.connected_components(G), key=len, reverse=True)
        main_nodes = components[0]
        G = G.subgraph(main_nodes).copy()
    nodes = list(G.nodes)
    N = len(nodes)
    print(f"Graph after connectivity check: {N} nodes, {G.number_of_edges()} edges, connected: {nx.is_connected(G)}")

    # --- Diagnostic sur les demandes ---
    mu_list = [G.nodes[n].get("mu", 1.0) for n in nodes]
    depot_mu = G.nodes[nodes[depot_idx]].get("mu", 0) if depot_idx < N else None
    print(f"Min/Max demande (mu): {min(mu_list):.2f}/{max(mu_list):.2f}, Somme: {sum(mu_list):.2f}, Dépôt mu: {depot_mu}")

    # --- Si toutes les demandes sont au dépôt, patch temporaire pour forcer du mouvement ---
    if mu_list.count(max(mu_list)) == 1 and max(mu_list) == mu_list[depot_idx] and sum(mu_list) > mu_list[depot_idx]:
        print("Toutes les demandes sont au dépôt ! Redistribution aléatoire des demandes pour forcer du mouvement (debug only).")
        for i, n in enumerate(nodes):
            G.nodes[n]["mu"] = random.randint(1, 15) if i != depot_idx else 0

    # --- Vérifier que le dépôt est présent ---
    if isinstance(depot_idx, int):
        if depot_idx < N:
            depot_node = nodes[depot_idx]
        else:
            print("Depot idx out of bounds, using 0.")
            depot_idx = 0
            depot_node = nodes[depot_idx]
    else:
        if depot_idx not in nodes:
            print("Depot node not in nodes, using the first node as depot.")
            depot_idx = 0
            depot_node = nodes[0]
        else:
            depot_node = depot_idx
            depot_idx = nodes.index(depot_node)

    print(f"Depot index: {depot_idx}, depot node: {depot_node}")

    if max_vehicles is None:
        max_vehicles = min(N, 20)
    print(f"Processing {N} nodes with max {max_vehicles} vehicles...")

    # --- Matrice de distance ---
    HIGH_DIST = 999999
    dist = np.full((N, N), HIGH_DIST, dtype=np.float64)
    print("Building distance matrix...")
    found_paths = 0
    for i, u in enumerate(nodes):
        for j, v in enumerate(nodes):
            if i == j:
                dist[i, j] = 0
            else:
                try:
                    d = nx.shortest_path_length(G, u, v, weight="distance")
                    dist[i, j] = d
                    found_paths += 1
                except nx.NetworkXNoPath:
                    dist[i, j] = HIGH_DIST
    if np.all(dist == HIGH_DIST):
        print("Warning: No valid paths found in graph. Exiting.")
        return {"cost": HIGH_DIST, "risk": HIGH_DIST, "unmet": 100, "time": 0, "viol": 100}
    print(f"Distance matrix built. {found_paths} finite paths, max dist: {dist[dist < HIGH_DIST].max():.1f}")

    print("Sample distance matrix (5x5):")
    print(dist[:5, :5])

    # --- Demandes (tonnes) ---
    demand = []
    total_demand = 0
    for n in nodes:
        try:
            mu = G.nodes[n].get("mu", 1.0)
            q = max(0, int(mu))
            demand.append(q)
            total_demand += q
        except (KeyError, TypeError, ValueError):
            demand.append(1)
            total_demand += 1
    print(f"Total demand: {total_demand}")

    # --- OR-Tools : data model ---
    data = {
        "distance_matrix": dist.astype(int).tolist(),
        "demands": demand,
        "vehicle_capacities": [vehicle_cap] * max_vehicles,
        "num_vehicles": max_vehicles,
        "depot": depot_idx,
    }

    # --- Solver avec timeout ---
    print("Starting OR-Tools solver...")
    t0 = time.perf_counter()

    try:
        manager = pywrapcp.RoutingIndexManager(len(data["distance_matrix"]), data["num_vehicles"], data["depot"])
        routing = pywrapcp.RoutingModel(manager)

        def distance_cb(i, j):
            return data["distance_matrix"][manager.IndexToNode(i)][manager.IndexToNode(j)]
        transit_cb_idx = routing.RegisterTransitCallback(distance_cb)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_cb_idx)

        def demand_cb(i):
            return data["demands"][manager.IndexToNode(i)]
        demand_cb_idx = routing.RegisterUnaryTransitCallback(demand_cb)
        routing.AddDimensionWithVehicleCapacity(
            demand_cb_idx, 0, data["vehicle_capacities"], True, "Capacity")

        search_strategy = routing_enums_pb2.FirstSolutionStrategy.SAVINGS
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = search_strategy
        search_parameters.time_limit.seconds = 30
        search_parameters.solution_limit = 100

        solution = routing.SolveWithParameters(search_parameters)
        elapsed = time.perf_counter() - t0

    except Exception as e:
        print(f"Error in OR-Tools solver: {e}")
        elapsed = time.perf_counter() - t0
        return {"cost": HIGH_DIST, "risk": HIGH_DIST, "unmet": 100, "time": elapsed, "viol": 100}

    if not solution:
        print("No solution found by OR-Tools.")
        return {"cost": HIGH_DIST, "risk": HIGH_DIST, "unmet": 100, "time": elapsed, "viol": 100}

    print("Extracting metrics from solution...")
    tot_dist, tot_risk = 0, 0
    viol = 0
    served_demand = 0
    routes_used = 0
    all_routes = []
    no_movement = True

    for v in range(data["num_vehicles"]):
        idx = routing.Start(v)
        load = 0
        route_nodes = []
        last_real_label = None
        while not routing.IsEnd(idx):
            node = manager.IndexToNode(idx)
            real_label = nodes[node]  # On stocke les labels réels
            route_nodes.append(real_label)
            if last_real_label is not None and real_label != last_real_label:
                no_movement = False
            last_real_label = real_label
            load += data["demands"][node]
            if load > vehicle_cap:
                viol += 1
            next_idx = routing.Next(solution, idx)
            if not routing.IsEnd(next_idx):
                nxt = manager.IndexToNode(next_idx)
                segment_dist = data["distance_matrix"][node][nxt]
                tot_dist += segment_dist
                try:
                    path = nx.shortest_path(G, real_label, nodes[nxt], weight="distance")
                    for i in range(len(path) - 1):
                        edge_data = G.get_edge_data(path[i], path[i + 1], {})
                        tot_risk += edge_data.get("risk", 0)
                except (nx.NetworkXNoPath, IndexError, KeyError):
                    pass
            idx = next_idx
        if len(set(route_nodes)) > 1:
            routes_used += 1
        all_routes.append(route_nodes)
        served_demand += min(load, vehicle_cap)
        print(f"Route {v}: {route_nodes[:10]}... ({len(route_nodes)} points, uniques: {len(set(route_nodes))})")

    unmet = max(0, total_demand - served_demand)
    print(f"Solution found: {routes_used} routes with movement")
    print(f"Total cost: {tot_dist}, Total risk: {tot_risk:.2f}, Violations: {viol}, Unmet demand: {unmet}")

    if no_movement:
        print("⚠️  ATTENTION : Aucune route n'effectue de mouvement réel entre nœuds !")
        print("-> Vérifiez vos demandes, la densité du graphe et la matrice de distance.")

    routes_filename = f"results/routes_cw_seed{seed}.pkl"
    with open(routes_filename, "wb") as f:
        pickle.dump(all_routes, f)

    return {
        "cost": tot_dist,
        "risk": tot_risk,
        "unmet": unmet,
        "time": elapsed,
        "viol": viol,
        "routes": all_routes
    }
