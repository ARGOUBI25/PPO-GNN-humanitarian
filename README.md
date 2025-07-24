# PPO-GNN-humanitarian
Adaptive Vehicle Routing for Humanitarian Aid Delivery in Conflict Zones using PPO with Graph Neural Networks (PPO-GNN). Includes data, code, and benchmarks on Afghanistan road networks with risk-aware routing.
## Overview

We address the challenging problem of routing humanitarian aid vehicles in conflict-affected areas, where road accessibility and security risks vary spatially and temporally. Our approach combines Proximal Policy Optimization (PPO), a state-of-the-art deep reinforcement learning algorithm, with Graph Neural Networks (GNN) to effectively learn routing policies that account for geographic topology and stochastic demands.

The repository includes:

- **Data**: Realistic Afghanistan road network graphs, georeferenced conflict event data, and humanitarian aid demand profiles.
- **Code**: Implementation of custom OpenAI Gym environments modeling the routing problem, PPO and PPO-GNN training scripts, and heuristic baseline algorithms (e.g., Clarke-Wright savings heuristic).
- **Evaluation**: Scripts for running experiments at multiple scales (small, medium, large), performance aggregation, and visualization of routes overlayed on geographic maps.
- **Reproducibility**: Modular code organization, with clear instructions to reproduce all experiments and generate synthetic and real-world inspired datasets.

## Repository Structure
PPO-GNN-humanitarian/
├── data/
│ ├── proc/ # Processed datasets (graph pickles, preprocessed files)
│ └── raw/ # Raw data files (road shapefiles, conflict events, demand CSVs)
├── heuristic/ # Clarke-Wright heuristic implementation
├── ppo_gnn/ # PPO and PPO-GNN environment and training code
├── results/ # Logs, routes, figures, and tables generated after experiments
├── afg_pipeline.py # Main experiment pipeline script
├── plot_routes_on_map.py # Visualization script for route mapping on geographic data
└── README.md # This documentation file

## Getting Started

1. **Data Preparation**: Use the provided raw data and preprocessing scripts to build graph representations.
2. **Training**: Train PPO and PPO-GNN agents on the routing environment with configurable parameters.
3. **Evaluation**: Run heuristic baselines and evaluate learned policies across multiple random seeds.
4. **Visualization**: Visualize optimal routes on Afghanistan maps to analyze spatial and risk-aware routing behavior.

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---
