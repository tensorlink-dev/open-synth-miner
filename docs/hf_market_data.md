# Using the `tensorlink-dev/open-synth-training-data` repository

This guide shows how to point the leak-safe `MarketDataLoader` at the public Hugging Face Parquet dataset published at `tensorlink-dev/open-synth-training-data`.

## Dataset expectations
The `HFParquetSource` expects three columns:
- `timestamp`: UTC timestamps (string, pandas-friendly datetime, or integer epoch). They will be converted with `pandas.to_datetime(..., utc=True)`.
- `asset`: Symbol or instrument name. If this column is missing, only a single asset can be loaded.
- `price`: Numeric close/last price.

## Inspect the repository
List available Parquet files (requires `huggingface_hub` and optional authentication for private forks):
```python
from huggingface_hub import HfApi
api = HfApi()
files = api.list_repo_files("tensorlink-dev/open-synth-training-data")
print(files)
```

If your environment needs authentication, run `huggingface-cli login` or set `HF_TOKEN` beforehand.

## Quick start: Z-score features with a static holdout split
```python
import pandas as pd
from src.data import HFParquetSource, MarketDataLoader, ZScoreEngineer

source = HFParquetSource(
    repo_id="tensorlink-dev/open-synth-training-data",
    filename="prices.parquet",  # replace with one of the listed Parquet files
    repo_type="dataset",  # point to the dataset endpoint on the Hub
    asset_column="asset",
    price_column="price",
    timestamp_column="timestamp",
)

loader = MarketDataLoader(
    data_source=source,
    engineer=ZScoreEngineer(short_win=20, long_win=200),
    assets=["BTC"],  # or multiple assets present in the parquet file
    input_len=96,
    pred_len=24,
    batch_size=64,
)

train_dl, val_dl, test_dl = loader.static_holdout(
    cutoff=pd.Timestamp("2023-01-01", tz="UTC"),
    val_size=0.2,
)
```

## Walk-forward or hybrid validation on the same data
```python
import pandas as pd
from src.data import HFParquetSource, MarketDataLoader, WaveletEngineer

source = HFParquetSource(
    repo_id="tensorlink-dev/open-synth-training-data",
    filename="prices.parquet",
    repo_type="dataset",
    asset_column="asset",
    price_column="price",
    timestamp_column="timestamp",
)
engineer = WaveletEngineer(wavelet="db4", level=3)
loader = MarketDataLoader(source, engineer, assets=["BTC"], input_len=64, pred_len=16, batch_size=32)

train_period = pd.Timedelta(days=14)
val_period = pd.Timedelta(days=2)
step = pd.Timedelta(days=2)

for fold, (train_dl, val_dl) in enumerate(loader.walk_forward(train_period, val_period, step)):
    print(f"Fold {fold}: train={len(train_dl)} batches, val={len(val_dl)} batches")
    if fold == 2:
        break

# Hybrid nested split with an immutable holdout
fragments, holdout_loader = loader.hybrid_nested(
    holdout_fraction=0.1,
    train_period=train_period,
    val_period=val_period,
    step_size=step,
)
```

## Safety checklist
- Ensure the cutoff/periods keep training windows strictly before validation/holdout windows—the loader enforces this with timestamp comparisons.
- Avoid mixing timezones; timestamps are normalized to UTC in the loader.
- If you add new Parquet files, preserve the `asset`, `timestamp`, and `price` column names or pass custom names to `HFParquetSource`.
- A `401` during download usually means the repository is private/gated or the wrong Hub endpoint was used. Set `repo_type="dataset"` (default) and authenticate with `huggingface-cli login` or `HF_TOKEN` if needed.
