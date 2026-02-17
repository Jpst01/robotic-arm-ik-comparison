import argparse
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..")


def load_csv(path):
    import csv as csvmod
    rows = []
    with open(path) as f:
        reader = csvmod.DictReader(f)
        for row in reader:
            rows.append(row)

    def col(name, dtype=float):
        return np.array([dtype(r[name]) for r in rows])

    return {
        "a_time":      col("a_time_ms"),
        "a_pos_err":   col("a_pos_err"),
        "a_angle_err": col("a_angle_err"),
        "a_ok":        col("a_ok", int),
        "ml_time":     col("ml_time_ms"),
        "ml_pos_err":  col("ml_pos_err"),
        "ml_angle_err": col("ml_angle_err"),
        "ml_ok":       col("ml_ok", int),
    }


def plot_bar_summary(d, out):
    a_ok = d["a_ok"].astype(bool)
    ml_ok = d["ml_ok"].astype(bool)

    metrics = {
        "Success Rate (%)": (
            100 * a_ok.mean(),
            100 * ml_ok.mean(),
        ),
        "Avg Solve Time (ms)": (
            d["a_time"].mean(),
            d["ml_time"].mean(),
        ),
        "Avg Pos Error (mm)": (
            d["a_pos_err"][a_ok].mean() * 1000 if a_ok.any() else 0,
            d["ml_pos_err"][ml_ok].mean() * 1000 if ml_ok.any() else 0,
        ),
        "Avg Joint Error (°)": (
            np.degrees(d["a_angle_err"][a_ok].mean()) if a_ok.any() else 0,
            np.degrees(d["ml_angle_err"][ml_ok].mean()) if ml_ok.any() else 0,
        ),
    }

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle("Analytical vs ML IK — Benchmark Summary", fontweight="bold", fontsize=14)

    colors = ["#2196F3", "#FF9800"]
    for ax, (label, (a_val, ml_val)) in zip(axes, metrics.items()):
        bars = ax.bar(["Analytical", "ML"], [a_val, ml_val], color=colors, width=0.5)
        ax.set_title(label, fontsize=11)
        ax.bar_label(bars, fmt="%.3g", fontsize=9)
        ax.set_ylim(0, max(a_val, ml_val) * 1.3 + 1e-9)

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def plot_error_histograms(d, out):
    a_ok = d["a_ok"].astype(bool)
    ml_ok = d["ml_ok"].astype(bool)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle("Error Distribution — Analytical vs ML", fontweight="bold", fontsize=14)

    ax = axes[0]
    if a_ok.any():
        ax.hist(d["a_pos_err"][a_ok] * 1000, bins=50, alpha=0.7,
                label="Analytical", color="#2196F3")
    if ml_ok.any():
        ax.hist(d["ml_pos_err"][ml_ok] * 1000, bins=50, alpha=0.7,
                label="ML", color="#FF9800")
    ax.set_xlabel("Position Error (mm)")
    ax.set_ylabel("Count")
    ax.set_title("Position Error Distribution")
    ax.legend()

    ax = axes[1]
    if a_ok.any():
        ax.hist(np.degrees(d["a_angle_err"][a_ok]), bins=50, alpha=0.7,
                label="Analytical", color="#2196F3")
    if ml_ok.any():
        ax.hist(np.degrees(d["ml_angle_err"][ml_ok]), bins=50, alpha=0.7,
                label="ML", color="#FF9800")
    ax.set_xlabel("Max Joint Angle Error (°)")
    ax.set_ylabel("Count")
    ax.set_title("Joint Angle Error Distribution")
    ax.legend()

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def plot_timing(d, out):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.set_title("Solve Time Distribution", fontweight="bold", fontsize=13)

    data = [d["a_time"], d["ml_time"]]
    bp = ax.boxplot(data, labels=["Analytical", "ML"],
                    patch_artist=True, widths=0.4)
    colors = ["#2196F3", "#FF9800"]
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)

    ax.set_ylabel("Time per solve (ms)")
    ax.set_yscale("log")
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark chart generator")
    parser.add_argument("--csv", default=os.path.join(OUTPUT_DIR, "data", "benchmark_results.csv"),
                        help="Path to benchmark_results.csv")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"ERROR: CSV not found at {args.csv}")
        print("Run the C++ benchmark first:  ./benchmark ../models/ik_model.onnx")
        sys.exit(1)

    d = load_csv(args.csv)
    print(f"Loaded {len(d['a_time'])} samples from {args.csv}")

    assets_dir = os.path.join(OUTPUT_DIR, "assets")
    os.makedirs(assets_dir, exist_ok=True)

    plot_bar_summary(d, os.path.join(assets_dir, "benchmark_summary.png"))
    plot_error_histograms(d, os.path.join(assets_dir, "benchmark_errors.png"))
    plot_timing(d, os.path.join(assets_dir, "benchmark_timing.png"))

    print("\nAll charts saved.")


if __name__ == "__main__":
    main()
