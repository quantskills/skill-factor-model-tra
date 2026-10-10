from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    dates = pd.bdate_range("2020-01-02", periods=54).strftime("%Y%m%d").astype(int)
    rows = []
    for date_index, date in enumerate(dates):
        regime = -1.0 if (date_index // 11) % 2 else 1.0
        for ticker in range(2001, 2017):
            trend, reversal, quality = rng.normal(size=3)
            label = regime * (0.3 * trend - 0.2 * reversal) + 0.15 * quality + rng.normal(scale=0.2)
            rows.append({"date": int(date), "ticker": ticker, "factor_trend": trend,
                         "factor_reversal": reversal, "factor_quality": quality, "label": label})
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(destination, index=False)
    print(destination)


if __name__ == "__main__":
    main()
