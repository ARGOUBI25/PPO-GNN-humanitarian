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

│ ├── proc/ # Processed datasets (graph pickles for large, medium, small scales)  

│ └── raw/ # Raw data files: road shapefiles, conflict events, demand CSVs  

│ ├── afg_osm_lines_shp/  

│ ├── afg_demands15.csv  

│ ├── ged_afg.csv  

│ ├── GEDEvent_v23_1.csv  

│ └── hotosm_afg_roads_lines.shp  

├── heuristic/ # Clarke-Wright heuristic implementation (clarke_wright.py)  

├── ppo_gnn/ # PPO and PPO-GNN environment and training code (train.py, env.py)  

├── results/ # Output directory for logs, routes, figures, and summary tables  

│ ├── logs/  

│ ├── routes/  

│ ├── figures/  

│ └── tables/  

├── afg_pipeline.py # Main pipeline script to run experiments end-to-end  

├── plot_routes_on_map.py # Script to visualize routes on Afghanistan geographic maps  

└── README.md # Project documentation and instructions  

## Getting Started

1. **Data Preparation**: Use the provided raw data and preprocessing scripts to build graph representations.
2. **Training**: Train PPO and PPO-GNN agents on the routing environment with configurable parameters.
3. **Evaluation**: Run heuristic baselines and evaluate learned policies across multiple random seeds.
4. **Visualization**: Visualize optimal routes on Afghanistan maps to analyze spatial and risk-aware routing behavior.

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Installation

This project requires Python 3.7+ and the following libraries:

- geopandas
- shapely
- networkx
- pandas
- numpy
- matplotlib
- torch
- torch_geometric
- stable-baselines3
- geopy
- scipy

Install dependencies via conda and pip:

conda install geopandas shapely networkx pandas numpy matplotlib scipy geopy -c conda-forge
pip install torch torch_geometric stable-baselines3

## Usage
Run full experimental pipeline  

python afg_pipeline.py [small|medium|large]  

Runs data preparation, model training (PPO and PPO-GNN), heuristic evaluation, and result aggregation for the selected scale.  


Visualize routes on map  

python plot_routes_on_map.py data/proc/afg_graph_large.pkl data/raw/afg_osm_lines_shp/afg_roads_lines.shp results/routes/routes_ppo_gnn_seed42.pkl  

Visualizes the routes output by the model overlaid on the Afghanistan road network.


