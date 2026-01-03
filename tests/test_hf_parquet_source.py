import pandas as pd
import pytest

from src.data.market_data_loader import HFParquetSource


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

    monkeypatch.setattr("src.data.market_data_loader.hf_hub_download", fake_download)

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

    monkeypatch.setattr("src.data.market_data_loader.hf_hub_download", fake_download)

    source = HFParquetSource(repo_id="id", filename="price.parquet", repo_type=custom_type)
    assets = source.load_data(["asset"])

    assert recorded["repo_type"] == custom_type
    assert assets[0].name == "asset"
