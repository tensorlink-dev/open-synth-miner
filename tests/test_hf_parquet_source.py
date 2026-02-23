import pandas as pd
import pytest

from osa.data.market_data_loader import HFParquetSource


def test_hf_parquet_source_sets_dataset_repo_type(tmp_path, monkeypatch):
    frame = pd.DataFrame(
        {
            "asset": ["BTC", "BTC", "BTC"],
            "timestamp": pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC"),
            "price": [42000.0, 42100.0, 42200.0],
        }
    )
    parquet_path = tmp_path / "btc.parquet"
    frame.to_parquet(parquet_path)

    recorded = {}

    def fake_download(*, repo_id, filename, revision=None, repo_type=None):
        recorded.update(repo_id=repo_id, filename=filename, revision=revision, repo_type=repo_type)
        return str(parquet_path)

    monkeypatch.setattr("osa.data.market_data_loader.hf_hub_download", fake_download)

    source = HFParquetSource(repo_id="tensorlink-dev/open-synth-training-data", filename="btc.parquet")
    assets = source.load_data(["BTC"])

    assert recorded == {
        "repo_id": "tensorlink-dev/open-synth-training-data",
        "filename": "btc.parquet",
        "revision": None,
        "repo_type": "dataset",
    }
    assert len(assets) == 1
    assert assets[0].name == "BTC"
    assert list(assets[0].prices) == [42000.0, 42100.0, 42200.0]


@pytest.mark.parametrize("custom_type", [None, "dataset", "space"])
def test_hf_parquet_source_allows_custom_repo_type(tmp_path, monkeypatch, custom_type):
    frame = pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=1, freq="h", tz="UTC"), "price": [1.0]})
    parquet_path = tmp_path / "price.parquet"
    frame.to_parquet(parquet_path)

    recorded = {}

    def fake_download(*, repo_id, filename, revision=None, repo_type=None):
        recorded.update(repo_type=repo_type)
        return str(parquet_path)

    monkeypatch.setattr("osa.data.market_data_loader.hf_hub_download", fake_download)

    source = HFParquetSource(repo_id="id", filename="price.parquet", repo_type=custom_type)
    assets = source.load_data(["asset"])

    assert recorded["repo_type"] == custom_type
    assert assets[0].name == "asset"


def test_hf_parquet_source_accepts_close_price_fallback(tmp_path, monkeypatch):
    frame = pd.DataFrame(
        {
            "asset": ["ETH", "ETH", "ETH"],
            "timestamp": pd.date_range("2024-06-01", periods=3, freq="h", tz="UTC"),
            "open": [10.0, 10.5, 11.0],
            "high": [10.5, 11.0, 11.5],
            "low": [9.5, 10.0, 10.5],
            "close": [10.2, 10.8, 11.3],
        }
    )
    parquet_path = tmp_path / "eth.parquet"
    frame.to_parquet(parquet_path)

    monkeypatch.setattr("osa.data.market_data_loader.hf_hub_download", lambda **_: str(parquet_path))

    source = HFParquetSource(repo_id="id", filename="eth.parquet")
    assets = source.load_data(["ETH"])

    assert len(assets) == 1
    assert assets[0].name == "ETH"
    assert list(assets[0].prices) == [10.2, 10.8, 11.3]


def test_hf_parquet_source_supports_covariates(tmp_path, monkeypatch):
    frame = pd.DataFrame(
        {
            "asset": ["BTC", "BTC", "BTC"],
            "timestamp": pd.date_range("2024-02-01", periods=3, freq="h", tz="UTC"),
            "open": [100.0, 101.0, 102.0],
            "high": [101.0, 102.0, 103.0],
            "low": [99.0, 100.0, 101.0],
            "close": [100.5, 101.5, 102.5],
            "volume": [5_000, 6_000, 7_000],
        }
    )
    parquet_path = tmp_path / "btc_ohlcv.parquet"
    frame.to_parquet(parquet_path)

    monkeypatch.setattr("osa.data.market_data_loader.hf_hub_download", lambda **_: str(parquet_path))

    source = HFParquetSource(
        repo_id="id",
        filename="btc_ohlcv.parquet",
        price_column="close",
        covariate_columns=["open", "high", "low", "volume"],
    )
    assets = source.load_data(["BTC"])

    assert len(assets) == 1
    asset = assets[0]
    assert asset.covariate_columns == ["open", "high", "low", "volume"]
    assert asset.covariates.shape == (3, 4)
    assert list(asset.covariates[:, 0]) == [100.0, 101.0, 102.0]
    assert list(asset.prices) == [100.5, 101.5, 102.5]
