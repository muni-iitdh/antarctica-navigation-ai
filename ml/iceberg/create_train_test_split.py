from pathlib import Path
import pandas as pd


# ---------------------------------------------------------
# File locations
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "iceberg"
    / "processed"
    / "iceberg_training.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "iceberg"
    / "processed"
)

TRAIN_FILE = OUTPUT_DIR / "iceberg_train.csv"
TEST_FILE = OUTPUT_DIR / "iceberg_test.csv"


# ---------------------------------------------------------
# Split dates
# ---------------------------------------------------------

TRAIN_START = pd.Timestamp("2025-08-14")
TRAIN_END = pd.Timestamp("2026-05-31")

TEST_START = pd.Timestamp("2026-06-01")
TEST_END = pd.Timestamp("2026-08-20")


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    print("=== Iceberg Train/Test Split ===")
    print()

    print(f"Loading: {INPUT_FILE}")

    df = pd.read_csv(
        INPUT_FILE,
        parse_dates=["previous_date", "Last Update"]
    )

    print(f"Total samples: {len(df)}")

    # -----------------------------------------------------
    # Chronological split
    # -----------------------------------------------------

    train = df[
        (df["Last Update"] >= TRAIN_START)
        & (df["Last Update"] <= TRAIN_END)
    ].copy()

    test = df[
        (df["Last Update"] >= TEST_START)
        & (df["Last Update"] <= TEST_END)
    ].copy()

    # -----------------------------------------------------
    # Sort for reproducibility
    # -----------------------------------------------------

    train = train.sort_values(
        ["Last Update", "Iceberg"]
    ).reset_index(drop=True)

    test = test.sort_values(
        ["Last Update", "Iceberg"]
    ).reset_index(drop=True)

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    train.to_csv(TRAIN_FILE, index=False)
    test.to_csv(TEST_FILE, index=False)

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    print()
    print("=== Training Set ===")
    print(f"Rows: {len(train)}")
    print(f"Date range: {train['Last Update'].min()} → {train['Last Update'].max()}")
    print(f"Icebergs: {train['Iceberg'].nunique()}")

    print()
    print("=== Test Set ===")
    print(f"Rows: {len(test)}")
    print(f"Date range: {test['Last Update'].min()} → {test['Last Update'].max()}")
    print(f"Icebergs: {test['Iceberg'].nunique()}")

    print()
    print("Saved:")
    print(TRAIN_FILE)
    print(TEST_FILE)

    print()
    print("=== Split Complete ===")


if __name__ == "__main__":
    main()