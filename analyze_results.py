import os
import glob
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon

def load_results(results_folder="results"):
    csv_files = glob.glob(os.path.join(results_folder, "*.csv"))
    dfs = []
    for file in csv_files:
        df = pd.read_csv(file)
        dfs.append(df)
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    else:
        raise FileNotFoundError("No CSV result files found in the folder.")

def summarize_metrics(df):
    grouped = df.groupby("method").agg(
        count=("cost", "count"),
        cost_mean=("cost", "mean"),
        cost_std=("cost", "std"),
        risk_mean=("risk", "mean"),
        risk_std=("risk", "std"),
        unmet_mean=("unmet", "mean"),
        unmet_std=("unmet", "std"),
        time_mean=("time", "mean"),
        time_std=("time", "std"),
        viol_mean=("viol", "mean"),
        viol_std=("viol", "std")
    )
    return grouped

def wilcoxon_test(df, metric, method1, method2):
    data1 = df[df["method"] == method1][metric]
    data2 = df[df["method"] == method2][metric]
    stat, p = wilcoxon(data1, data2)
    return stat, p

def main():
    results_folder = "results"
    df = load_results(results_folder)
    summary = summarize_metrics(df)
    print("Summary of metrics by method:")
    print(summary)

    # Exemple de test de Wilcoxon pour 'cost' entre ppo_gnn et ppo
    stat, p = wilcoxon_test(df, "cost", "ppo_gnn", "ppo")
    print(f"\nWilcoxon test for 'cost' PPO-GNN vs PPO: stat={stat}, p-value={p}")

    # Enregistrer le résumé au format CSV pour inclusion dans le repo
    summary.to_csv(os.path.join(results_folder, "summary_metrics.csv"))

if __name__ == "__main__":
    main()
