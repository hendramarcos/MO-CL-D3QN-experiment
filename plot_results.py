"""Plot hasil lima metrik evaluasi penelitian.
Cara: python plot_results.py --csv outputs_single_intersection/evaluation_summary.csv
"""
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="outputs_single_intersection/evaluation_summary.csv")
    p.add_argument("--out-dir", default="outputs_single_intersection/plots")
    args = p.parse_args()
    df = pd.read_csv(args.csv)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    metrics = [
        ("Avg Travel Time (s)", "Avg Travel Time (s)", "01_avg_travel_time.png", True),
        ("Avg Queue Length", "Avg Queue Length", "02_avg_queue_length.png", True),
        ("Avg Delay (s/veh)", "Avg Delay (s/veh)", "03_avg_delay.png", True),
        ("Throughput (veh/h)", "Throughput (veh/h)", "04_throughput.png", False),
        ("Fuel Consumption (L/h)", "Fuel Consumption (L/h)", "05_fuel_consumption.png", True),
    ]
    for col, ylabel, fname, _ in metrics:
        plt.figure(figsize=(11, 5.5))
        plt.bar(df["Model"], df[col])
        plt.xlabel("Model")
        plt.ylabel(ylabel)
        plt.title(f"Perbandingan {ylabel}")
        plt.xticks(rotation=25, ha="right")
        plt.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(out / fname, dpi=300, bbox_inches="tight")
        plt.close()
    score = df.copy()
    for col, _, _, lower in metrics:
        score[f"score_{col}"] = score[col].min() / score[col] if lower else score[col] / score[col].max()
    score_cols = [f"score_{m[0]}" for m in metrics]
    score["Normalized Performance Index"] = score[score_cols].mean(axis=1)
    score[["Model", "Normalized Performance Index"]].to_csv(out / "normalized_performance_index.csv", index=False)
    plt.figure(figsize=(11, 5.5))
    plt.bar(score["Model"], score["Normalized Performance Index"])
    plt.xlabel("Model")
    plt.ylabel("Normalized Performance Index")
    plt.title("Indeks Performa Ternormalisasi Berdasarkan 5 Metrik lima metrik evaluasi penelitian")
    plt.xticks(rotation=25, ha="right")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / "06_normalized_performance_index.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Plot tersimpan di: {out}")


if __name__ == "__main__":
    main()
