import os
import numpy as np
import pandas as pd

L1 = 0.3
L2 = 0.25
L3 = 0.15

JOINT_MIN = -np.pi
JOINT_MAX = np.pi

NUM_SAMPLES = 50000
NOISE_STD = 0.001

SEED = 42
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def forward_kinematics(t1, t2, t3):
    r = L2 * np.cos(t2) + L3 * np.cos(t2 + t3)
    x = np.cos(t1) * r
    y = np.sin(t1) * r
    z = L1 + L2 * np.sin(t2) + L3 * np.sin(t2 + t3)
    return x, y, z


def generate_dataset(n=NUM_SAMPLES, seed=SEED):
    rng = np.random.default_rng(seed)

    t1 = rng.uniform(-np.pi, np.pi, n)
    t2 = rng.uniform(-np.pi / 2, np.pi / 2, n)
    t3 = rng.uniform(0, np.pi, n)

    x, y, z = forward_kinematics(t1, t2, t3)

    x += rng.normal(0, NOISE_STD, n)
    y += rng.normal(0, NOISE_STD, n)
    z += rng.normal(0, NOISE_STD, n)

    df = pd.DataFrame({
        "x": x, "y": y, "z": z,
        "t1": t1, "t2": t2, "t3": t3,
    })

    return df


def split_dataset(df, train_frac=0.8, val_frac=0.1, seed=SEED):
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    n = len(df)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train = df.iloc[:n_train]
    val = df.iloc[n_train:n_train + n_val]
    test = df.iloc[n_train + n_val:]

    return train, val, test


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Generating {NUM_SAMPLES} samples...")
    df = generate_dataset()

    print(f"Dataset shape: {df.shape}")
    print(f"Position ranges:")
    print(f"  x: [{df['x'].min():.3f}, {df['x'].max():.3f}]")
    print(f"  y: [{df['y'].min():.3f}, {df['y'].max():.3f}]")
    print(f"  z: [{df['z'].min():.3f}, {df['z'].max():.3f}]")

    train, val, test = split_dataset(df)
    print(f"\nSplit: train={len(train)}, val={len(val)}, test={len(test)}")

    train.to_csv(os.path.join(OUTPUT_DIR, "train.csv"), index=False)
    val.to_csv(os.path.join(OUTPUT_DIR, "val.csv"), index=False)
    test.to_csv(os.path.join(OUTPUT_DIR, "test.csv"), index=False)

    print(f"Saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
