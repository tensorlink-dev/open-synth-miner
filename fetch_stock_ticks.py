#!/usr/bin/env python3
"""Fetch 1-min and 5-min OHLCV bars from the Massive API and save as parquet.

Usage
-----
    # Fetch defaults (SPYX, NVDAX, TSLAX, AAPLX, GOOGLX) for 2024:
    python fetch_stock_ticks.py

    # Custom date range and output directory:
    python fetch_stock_ticks.py --from-date 2024-06-01 --to-date 2025-01-01 --out-dir ./data

    # Fetch specific assets only:
    python fetch_stock_ticks.py --assets SPYX NVDAX

Requires the ``MASSIVE_API_KEY`` environment variable to be set.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)

# Default Synth subnet asset names → exchange tickers
DEFAULT_TICKER_MAP: Dict[str, str] = {
    "SPYX": "SPY",
    "NVDAX": "NVDA",
    "TSLAX": "TSLA",
    "AAPLX": "AAPL",
    "GOOGLX": "GOOGL",
}

DEFAULT_ASSETS = list(DEFAULT_TICKER_MAP.keys())
INTERVALS = [
    {"multiplier": 1, "timespan": "minute", "label": "1min"},
    {"multiplier": 5, "timespan": "minute", "label": "5min"},
]


def fetch_bars(
    api_key: str,
    ticker: str,
    multiplier: int,
    timespan: str,
    from_date: str,
    to_date: str,
) -> pd.DataFrame:
    """Fetch aggregate bars from the Massive API and return a DataFrame."""
    from massive import RESTClient

    client = RESTClient(api_key=api_key)
    aggs = list(
        client.list_aggs(
            ticker=ticker,
            multiplier=multiplier,
            timespan=timespan,
            from_=from_date,
            to=to_date,
            adjusted=True,
            sort="asc",
            limit=50000,
        )
    )
    if not aggs:
        return pd.DataFrame()

    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime([a.timestamp for a in aggs], unit="ms", utc=True),
            "open": [a.open for a in aggs],
            "high": [a.high for a in aggs],
            "low": [a.low for a in aggs],
            "close": [a.close for a in aggs],
            "volume": [a.volume for a in aggs],
            "vwap": [a.vwap for a in aggs],
            "transactions": [a.transactions for a in aggs],
        }
    )
    return df.sort_values("timestamp").reset_index(drop=True)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Fetch stock ticks from Massive API")
    parser.add_argument(
        "--assets",
        nargs="+",
        default=DEFAULT_ASSETS,
        help="Asset names to fetch (default: %(default)s)",
    )
    parser.add_argument("--from-date", default="2024-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--to-date", default="2025-01-01", help="End date YYYY-MM-DD")
    parser.add_argument("--out-dir", default="data/stock_ticks", help="Output directory")
    parser.add_argument(
        "--ticker-map",
        nargs="*",
        metavar="ASSET=TICKER",
        help="Override ticker mapping, e.g. SPYX=SPY",
    )
    args = parser.parse_args(argv)

    api_key = os.getenv("MASSIVE_API_KEY")
    if not api_key:
        logger.error("MASSIVE_API_KEY environment variable is not set")
        sys.exit(1)

    ticker_map = dict(DEFAULT_TICKER_MAP)
    if args.ticker_map:
        for pair in args.ticker_map:
            asset, ticker = pair.split("=", 1)
            ticker_map[asset] = ticker

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for asset in args.assets:
        ticker = ticker_map.get(asset, asset)
        for interval in INTERVALS:
            label = interval["label"]
            logger.info(
                "Fetching %s (%s) %s bars  [%s → %s]",
                asset, ticker, label, args.from_date, args.to_date,
            )
            df = fetch_bars(
                api_key=api_key,
                ticker=ticker,
                multiplier=interval["multiplier"],
                timespan=interval["timespan"],
                from_date=args.from_date,
                to_date=args.to_date,
            )
            if df.empty:
                logger.warning("  No data returned for %s %s", asset, label)
                continue

            asset_dir = out_dir / asset
            asset_dir.mkdir(parents=True, exist_ok=True)
            out_path = asset_dir / f"{label}.parquet"
            df.to_parquet(out_path, index=False)
            logger.info(
                "  Saved %d bars → %s  [%s to %s]",
                len(df), out_path,
                df["timestamp"].iloc[0].strftime("%Y-%m-%d %H:%M"),
                df["timestamp"].iloc[-1].strftime("%Y-%m-%d %H:%M"),
            )

    logger.info("Done.")


if __name__ == "__main__":
    main()
