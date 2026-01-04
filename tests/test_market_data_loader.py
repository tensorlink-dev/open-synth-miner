import numpy as np
import pandas as pd
import pytest

from src.data.market_data_loader import MarketDataLoader, MockDataSource, ZScoreEngineer


def test_static_holdout_accepts_fractional_cutoff():
    loader = MarketDataLoader(
        data_source=MockDataSource(length=240, freq="1h", seed=1, base_price=150.0),
        engineer=ZScoreEngineer(),
        assets=["BTC"],
        input_len=12,
        pred_len=4,
        batch_size=16,
    )

    train_loader, val_loader, test_loader = loader.static_holdout(0.2, val_size=0.25, shuffle_train=False)

    horizons = loader.dataset.get_horizon_timestamps()
    starts = loader.dataset.get_start_timestamps()
    cutoff_ts = pd.Series(horizons).quantile(0.8)

    assert len(test_loader.dataset) == int(np.sum(starts >= cutoff_ts))
    assert len(train_loader.dataset) + len(val_loader.dataset) == int(np.sum(horizons < cutoff_ts))


def test_static_holdout_validates_fraction_bounds():
    loader = MarketDataLoader(
        data_source=MockDataSource(length=120, freq="1h", seed=2, base_price=200.0),
        engineer=ZScoreEngineer(),
        assets=["ETH"],
        input_len=8,
        pred_len=2,
        batch_size=8,
    )

    with pytest.raises(ValueError):
        loader.static_holdout(1.2)

