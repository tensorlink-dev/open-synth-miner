"""Tests for the regime-aware flexible data loader."""
import numpy as np
import pandas as pd
import pytest

from src.data.regime_loader import (
    HOURLY_FEATURE_NAMES,
    AggregationStep,
    ClusteringStrategy,
    FeatureStep,
    Fold,
    GaussianMixtureStrategy,
    KMeansStrategy,
    LogReturnTarget,
    MultiColumnTarget,
    OHLCVAggregation,
    OHLCVFeatureStep,
    PipelineConfig,
    RawReturnTarget,
    RegimeAwareDataset,
    RegimeBalancedSampler,
    RegimeDriftMonitor,
    RegimeTagger,
    TargetBuilder,
    aggregate_5m_to_1h,
    asset_to_ohlcv_frame,
    engineer_features,
    generate_walk_forward_folds,
    run_pipeline,
)
from src.data.market_data_loader import AssetData, MockDataSource


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_raw_5m(n_bars: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic 5-minute OHLCV candles."""
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2023-01-01", periods=n_bars, freq="5min", tz="UTC")
    log_rets = rng.normal(0, 0.001, n_bars)
    close = 100.0 * np.exp(np.cumsum(log_rets))
    high = close * (1 + np.abs(rng.normal(0, 0.002, n_bars)))
    low = close * (1 - np.abs(rng.normal(0, 0.002, n_bars)))
    open_ = close * (1 + rng.normal(0, 0.0005, n_bars))
    volume = np.abs(rng.normal(1000, 300, n_bars))
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=timestamps,
    )


# ---------------------------------------------------------------------------
# 1. aggregate_5m_to_1h
# ---------------------------------------------------------------------------


class TestAggregate5mTo1h:
    def test_basic_aggregation(self):
        df = _make_raw_5m(720)  # 60 hours
        result = aggregate_5m_to_1h(df)
        # 720 five-minute bars / 12 per hour = 60 hourly bars
        assert len(result) == 60

    def test_ohlcv_correctness(self):
        df = _make_raw_5m(720)
        result = aggregate_5m_to_1h(df)
        # First hour: first 12 bars (indices 0-11)
        first_hour = df.iloc[:12]
        assert result.iloc[0]["open"] == pytest.approx(first_hour["open"].iloc[0])
        assert result.iloc[0]["high"] == pytest.approx(first_hour["high"].max())
        assert result.iloc[0]["low"] == pytest.approx(first_hour["low"].min())
        assert result.iloc[0]["close"] == pytest.approx(first_hour["close"].iloc[-1])
        assert result.iloc[0]["volume"] == pytest.approx(first_hour["volume"].sum())

    def test_has_intra_hour_stats(self):
        df = _make_raw_5m(720)
        result = aggregate_5m_to_1h(df)
        assert "intra_ret_std" in result.columns
        assert "intra_max_drawdown" in result.columns
        assert "intra_directional_consistency" in result.columns

    def test_no_nans(self):
        df = _make_raw_5m(720)
        result = aggregate_5m_to_1h(df)
        assert not result.isnull().any().any()

    def test_custom_resample_rule(self):
        df = _make_raw_5m(720)
        result = aggregate_5m_to_1h(df, resample_rule="2h")
        assert len(result) == 30  # 720 / 24 per 2h = 30


# ---------------------------------------------------------------------------
# 2. engineer_features
# ---------------------------------------------------------------------------


class TestEngineerFeatures:
    def test_output_columns(self):
        df = _make_raw_5m(1440)
        hourly = aggregate_5m_to_1h(df)
        features = engineer_features(hourly)
        assert list(features.columns) == HOURLY_FEATURE_NAMES

    def test_no_nans_or_infs(self):
        df = _make_raw_5m(1440)
        hourly = aggregate_5m_to_1h(df)
        features = engineer_features(hourly)
        assert not features.isnull().any().any()
        assert not np.isinf(features.values).any()

    def test_log_ret_computed(self):
        df = _make_raw_5m(1440)
        hourly = aggregate_5m_to_1h(df)
        features = engineer_features(hourly)
        # log_ret should be mostly small values
        assert features["log_ret"].abs().max() < 1.0

    def test_vol_features_positive(self):
        df = _make_raw_5m(1440)
        hourly = aggregate_5m_to_1h(df)
        features = engineer_features(hourly)
        assert (features["realized_vol"] >= 0).all()
        assert (features["parkinson_vol"] >= 0).all()


# ---------------------------------------------------------------------------
# 3. RegimeTagger
# ---------------------------------------------------------------------------


class TestRegimeTagger:
    def test_fit_predict(self):
        df = _make_raw_5m(2880)
        hourly = aggregate_5m_to_1h(df)
        features = engineer_features(hourly)
        tagger = RegimeTagger(n_regimes=3)
        labels = tagger.fit_predict(features)
        assert labels.shape == (len(features),)
        assert set(labels).issubset({0, 1, 2})

    def test_no_lookahead(self):
        """Tagger fitted on first half should work on second half."""
        df = _make_raw_5m(4000)
        hourly = aggregate_5m_to_1h(df)
        features = engineer_features(hourly)
        mid = len(features) // 2
        train_features = features.iloc[:mid]
        test_features = features.iloc[mid:]

        tagger = RegimeTagger(n_regimes=3)
        tagger.fit(train_features)
        labels = tagger.predict(test_features)
        assert labels.shape == (len(test_features),)

    def test_regime_distribution(self):
        df = _make_raw_5m(2880)
        hourly = aggregate_5m_to_1h(df)
        features = engineer_features(hourly)
        tagger = RegimeTagger(n_regimes=3)
        labels = tagger.fit_predict(features)
        dist = tagger.regime_distribution(labels)
        assert dist.shape == (3,)
        assert abs(dist.sum() - 1.0) < 1e-8

    def test_predict_before_fit_raises(self):
        tagger = RegimeTagger()
        df = _make_raw_5m(720)
        hourly = aggregate_5m_to_1h(df)
        features = engineer_features(hourly)
        with pytest.raises(RuntimeError):
            tagger.predict(features)


# ---------------------------------------------------------------------------
# 4. generate_walk_forward_folds
# ---------------------------------------------------------------------------


class TestWalkForwardFolds:
    def test_basic_fold_generation(self):
        df = _make_raw_5m(5000)
        hourly = aggregate_5m_to_1h(df)
        features = engineer_features(hourly)

        folds = generate_walk_forward_folds(
            features,
            train_size=100,
            val_size=30,
            test_size=30,
            step_size=30,
            gap_size=5,
        )
        assert len(folds) > 0
        assert all(isinstance(f, Fold) for f in folds)

    def test_fold_sizes(self):
        df = _make_raw_5m(5000)
        hourly = aggregate_5m_to_1h(df)
        features = engineer_features(hourly)

        folds = generate_walk_forward_folds(
            features,
            train_size=100,
            val_size=30,
            test_size=30,
            step_size=30,
            gap_size=5,
        )
        for fold in folds:
            assert len(fold.train_features) == 100
            assert len(fold.val_features) == 30
            assert len(fold.test_features) == 30

    def test_regime_labels_shape(self):
        df = _make_raw_5m(5000)
        hourly = aggregate_5m_to_1h(df)
        features = engineer_features(hourly)

        folds = generate_walk_forward_folds(
            features,
            train_size=100,
            val_size=30,
            test_size=30,
            step_size=30,
            gap_size=5,
        )
        for fold in folds:
            assert fold.train_regimes.shape == (100,)
            assert fold.val_regimes.shape == (30,)
            assert fold.test_regimes.shape == (30,)

    def test_gap_separation(self):
        """Validate that val/test don't overlap with train (gap present)."""
        df = _make_raw_5m(5000)
        hourly = aggregate_5m_to_1h(df)
        features = engineer_features(hourly)

        folds = generate_walk_forward_folds(
            features,
            train_size=100,
            val_size=30,
            test_size=30,
            step_size=30,
            gap_size=10,
        )
        for fold in folds:
            train_end_idx = features.index.get_loc(fold.train_features.index[-1])
            val_start_idx = features.index.get_loc(fold.val_features.index[0])
            assert val_start_idx - train_end_idx > 1  # gap exists

    def test_too_large_window_raises(self):
        df = _make_raw_5m(200)  # Very small
        hourly = aggregate_5m_to_1h(df)
        features = engineer_features(hourly)

        with pytest.raises(ValueError, match="exceeds data length"):
            generate_walk_forward_folds(
                features,
                train_size=100,
                val_size=50,
                test_size=50,
                step_size=50,
                gap_size=10,
            )


# ---------------------------------------------------------------------------
# 5. RegimeBalancedSampler
# ---------------------------------------------------------------------------


class TestRegimeBalancedSampler:
    def test_basic_sampling(self):
        labels = np.array([0] * 100 + [1] * 20 + [2] * 80)
        sampler = RegimeBalancedSampler(labels, seq_len=10, n_samples=50)
        indices = list(sampler)
        assert len(indices) == 50
        assert all(0 <= idx <= len(labels) - 10 for idx in indices)

    def test_minority_boost(self):
        """The core behavior: minority regimes get boosted sampling."""
        labels = np.array([0] * 500 + [1] * 100 + [2] * 400)
        sampler = RegimeBalancedSampler(
            labels, seq_len=1, balance_strength=1.0, n_samples=10000
        )
        eff = sampler.effective_distribution()
        # With full balancing, the minority regime (1, ~10% of data)
        # should have roughly equal sampling probability
        assert eff[1] > 0.15  # Much higher than the 10% data share

    def test_balance_strength_zero_is_uniform(self):
        labels = np.array([0] * 500 + [1] * 100 + [2] * 400)
        sampler = RegimeBalancedSampler(
            labels, seq_len=1, balance_strength=0.0
        )
        eff = sampler.effective_distribution()
        # With zero strength, sampling should roughly match data proportions
        assert abs(eff[0] - 0.50) < 0.10
        assert abs(eff[1] - 0.10) < 0.05

    def test_summary(self):
        labels = np.array([0] * 100 + [1] * 20 + [2] * 80)
        sampler = RegimeBalancedSampler(labels, seq_len=5)
        summary = sampler.summary()
        assert "Regime 0" in summary
        assert "Regime 1" in summary
        assert "Regime 2" in summary

    def test_seq_len_exceeds_data_raises(self):
        labels = np.array([0, 1, 2])
        with pytest.raises(ValueError, match="exceeds data length"):
            RegimeBalancedSampler(labels, seq_len=10)


# ---------------------------------------------------------------------------
# 6. RegimeDriftMonitor
# ---------------------------------------------------------------------------


class TestRegimeDriftMonitor:
    def _make_monitor(self):
        df = _make_raw_5m(2880)
        hourly = aggregate_5m_to_1h(df)
        features = engineer_features(hourly)
        tagger = RegimeTagger(n_regimes=3)
        labels = tagger.fit_predict(features)
        train_dist = tagger.regime_distribution(labels)
        monitor = RegimeDriftMonitor(
            tagger=tagger,
            train_distribution=train_dist,
            kl_threshold=0.5,
            window_size=len(features),
        )
        return monitor, features

    def test_no_retrain_on_same_distribution(self):
        monitor, features = self._make_monitor()
        # Feed the full training set so distribution matches
        monitor.update(features)
        assert not monitor.should_retrain()

    def test_status_keys(self):
        monitor, features = self._make_monitor()
        monitor.update(features.iloc[:50])
        status = monitor.status()
        assert "train_distribution" in status
        assert "live_distribution" in status
        assert "symmetric_kl" in status
        assert "should_retrain" in status

    def test_unfitted_tagger_raises(self):
        tagger = RegimeTagger(n_regimes=3)
        with pytest.raises(ValueError, match="fitted"):
            RegimeDriftMonitor(
                tagger=tagger,
                train_distribution=np.array([0.5, 0.3, 0.2]),
            )


# ---------------------------------------------------------------------------
# 7. run_pipeline (integration)
# ---------------------------------------------------------------------------


class TestRunPipeline:
    def test_end_to_end(self):
        df = _make_raw_5m(5000)
        cfg = PipelineConfig(
            train_size=100,
            val_size=30,
            test_size=30,
            step_size=30,
            gap_size=5,
            seq_len=16,
            pred_len=4,
            balance_strength=0.8,
            batch_size=8,
        )
        fold_loaders = run_pipeline(df, cfg)
        assert len(fold_loaders) > 0

        fl = fold_loaders[0]
        for batch in fl.train_loader:
            assert "inputs" in batch
            assert "target" in batch
            assert batch["inputs"].dim() == 3  # (B, F, seq_len)
            assert batch["target"].dim() == 3  # (B, 1, pred_len)
            break

    def test_sampler_reports_boost(self):
        df = _make_raw_5m(5000)
        cfg = PipelineConfig(
            train_size=100,
            val_size=30,
            test_size=30,
            step_size=30,
            gap_size=5,
            seq_len=16,
            pred_len=4,
            balance_strength=0.8,
        )
        fold_loaders = run_pipeline(df, cfg)
        for fl in fold_loaders:
            eff = fl.sampler.effective_distribution()
            # All regimes should have non-zero probability
            assert all(p > 0 for p in eff.values())


# ---------------------------------------------------------------------------
# Convenience: asset_to_ohlcv_frame
# ---------------------------------------------------------------------------


class TestAssetToOhlcvFrame:
    def test_with_covariates(self):
        n = 1000
        rng = np.random.default_rng(42)
        close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.001, n)))
        ts = pd.date_range("2023-01-01", periods=n, freq="5min", tz="UTC")
        asset = AssetData(
            name="TEST",
            timestamps=ts.to_numpy(),
            prices=close,
            covariate_columns=["open", "high", "low", "volume"],
            covariates=np.column_stack([
                close * 0.999,
                close * 1.002,
                close * 0.998,
                np.abs(rng.normal(1000, 300, n)),
            ]),
        )
        df = asset_to_ohlcv_frame(asset)
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]
        assert isinstance(df.index, pd.DatetimeIndex)
        assert len(df) == n

    def test_fallback_without_covariates(self):
        n = 500
        rng = np.random.default_rng(42)
        close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.001, n)))
        ts = pd.date_range("2023-01-01", periods=n, freq="5min", tz="UTC")
        asset = AssetData(
            name="TEST",
            timestamps=ts.to_numpy(),
            prices=close,
        )
        df = asset_to_ohlcv_frame(asset)
        # Should create O=H=L=C=close, volume=1
        assert (df["open"] == df["close"]).all()
        assert (df["volume"] == 1.0).all()


# ---------------------------------------------------------------------------
# 8. ClusteringStrategy ABC
# ---------------------------------------------------------------------------


class TestClusteringStrategyABC:
    def test_kmeans_strategy_fit_predict(self):
        rng = np.random.default_rng(42)
        X = rng.normal(size=(200, 2))
        strategy = KMeansStrategy(n_clusters=3, random_state=42)
        strategy.fit(X)
        labels = strategy.predict(X)
        assert labels.shape == (200,)
        assert set(labels).issubset({0, 1, 2})
        assert strategy.n_clusters == 3

    def test_gmm_strategy_fit_predict(self):
        rng = np.random.default_rng(42)
        X = rng.normal(size=(200, 2))
        strategy = GaussianMixtureStrategy(n_components=4, random_state=42)
        strategy.fit(X)
        labels = strategy.predict(X)
        assert labels.shape == (200,)
        assert set(labels).issubset({0, 1, 2, 3})
        assert strategy.n_clusters == 4

    def test_kmeans_predict_before_fit_raises(self):
        strategy = KMeansStrategy(n_clusters=3)
        with pytest.raises(RuntimeError):
            strategy.predict(np.zeros((10, 2)))

    def test_gmm_predict_before_fit_raises(self):
        strategy = GaussianMixtureStrategy(n_components=3)
        with pytest.raises(RuntimeError):
            strategy.predict(np.zeros((10, 2)))

    def test_custom_strategy_works_with_tagger(self):
        """A custom ClusteringStrategy integrates with RegimeTagger."""

        class FixedStrategy(ClusteringStrategy):
            def __init__(self, n):
                self._n = n
            def fit(self, X):
                return self
            def predict(self, X):
                return np.zeros(len(X), dtype=int)
            @property
            def n_clusters(self):
                return self._n

        df = _make_raw_5m(2880)
        hourly = aggregate_5m_to_1h(df)
        features = engineer_features(hourly)
        tagger = RegimeTagger(n_regimes=2, clustering=FixedStrategy(2))
        labels = tagger.fit_predict(features)
        assert set(labels) == {0}


# ---------------------------------------------------------------------------
# 9. RegimeTagger with GMM clustering
# ---------------------------------------------------------------------------


class TestRegimeTaggerGMM:
    def test_gmm_tagger_fit_predict(self):
        df = _make_raw_5m(2880)
        hourly = aggregate_5m_to_1h(df)
        features = engineer_features(hourly)
        strategy = GaussianMixtureStrategy(n_components=3, random_state=42)
        tagger = RegimeTagger(n_regimes=3, clustering=strategy)
        labels = tagger.fit_predict(features)
        assert labels.shape == (len(features),)
        assert len(set(labels)) >= 1  # At least one cluster used

    def test_gmm_custom_vol_features(self):
        df = _make_raw_5m(2880)
        hourly = aggregate_5m_to_1h(df)
        features = engineer_features(hourly)
        strategy = GaussianMixtureStrategy(n_components=3)
        tagger = RegimeTagger(
            n_regimes=3,
            clustering=strategy,
            vol_features=["realized_vol", "parkinson_vol", "skew"],
        )
        labels = tagger.fit_predict(features)
        assert labels.shape == (len(features),)


# ---------------------------------------------------------------------------
# 10. FeatureStep ABC
# ---------------------------------------------------------------------------


class TestFeatureStepABC:
    def test_ohlcv_feature_step(self):
        df = _make_raw_5m(1440)
        hourly = aggregate_5m_to_1h(df)
        step = OHLCVFeatureStep()
        features = step.transform(hourly)
        assert list(features.columns) == HOURLY_FEATURE_NAMES
        assert step.feature_names == HOURLY_FEATURE_NAMES

    def test_ohlcv_feature_step_custom_windows(self):
        df = _make_raw_5m(1440)
        hourly = aggregate_5m_to_1h(df)
        step = OHLCVFeatureStep(
            realized_vol_window=12,
            parkinson_vol_window=12,
            smooth_window=3,
        )
        features = step.transform(hourly)
        assert not features.isnull().any().any()

    def test_custom_feature_step(self):
        """A custom FeatureStep can produce arbitrary features."""
        from typing import List

        class SimpleFeatureStep(FeatureStep):
            @property
            def feature_names(self) -> List[str]:
                return ["close", "volume"]

            def transform(self, df_bars):
                return df_bars[["close", "volume"]].copy()

        df = _make_raw_5m(720)
        hourly = aggregate_5m_to_1h(df)
        step = SimpleFeatureStep()
        features = step.transform(hourly)
        assert list(features.columns) == ["close", "volume"]
        assert len(features) == len(hourly)


# ---------------------------------------------------------------------------
# 11. AggregationStep ABC
# ---------------------------------------------------------------------------


class TestAggregationStepABC:
    def test_ohlcv_aggregation_default(self):
        df = _make_raw_5m(720)
        agg = OHLCVAggregation()
        result = agg.aggregate(df)
        assert len(result) == 60

    def test_ohlcv_aggregation_custom_rule(self):
        df = _make_raw_5m(720)
        agg = OHLCVAggregation(resample_rule="2h")
        result = agg.aggregate(df)
        assert len(result) == 30

    def test_custom_aggregation_step(self):
        """A custom AggregationStep can implement arbitrary aggregation."""

        class PassthroughAggregation(AggregationStep):
            def aggregate(self, df_raw):
                return df_raw.copy()

        df = _make_raw_5m(100)
        agg = PassthroughAggregation()
        result = agg.aggregate(df)
        assert len(result) == 100


# ---------------------------------------------------------------------------
# 12. TargetBuilder ABC
# ---------------------------------------------------------------------------


class TestTargetBuilderABC:
    def _make_features(self):
        df = _make_raw_5m(1440)
        hourly = aggregate_5m_to_1h(df)
        return engineer_features(hourly)

    def test_log_return_target(self):
        features = self._make_features()
        tb = LogReturnTarget()
        targets = tb.build_targets(features)
        assert targets.shape == (len(features),)
        assert targets.dtype == np.float32
        # First value should be 0 (no previous to diff against)
        assert targets[0] == 0.0

    def test_log_return_extract(self):
        features = self._make_features()
        tb = LogReturnTarget()
        targets = tb.build_targets(features)
        tensor = tb.extract_target(targets, start=10, length=5)
        assert tensor.shape == (1, 5)  # (1, pred_len)

    def test_raw_return_target(self):
        features = self._make_features()
        tb = RawReturnTarget()
        targets = tb.build_targets(features)
        assert targets.shape == (len(features),)
        assert targets.dtype == np.float32

    def test_raw_return_extract(self):
        features = self._make_features()
        tb = RawReturnTarget()
        targets = tb.build_targets(features)
        tensor = tb.extract_target(targets, start=10, length=5)
        assert tensor.shape == (1, 5)

    def test_multi_column_target(self):
        features = self._make_features()
        tb = MultiColumnTarget(columns=["close", "volume"])
        targets = tb.build_targets(features)
        assert targets.shape == (len(features), 2)
        assert targets.dtype == np.float32

    def test_multi_column_extract(self):
        features = self._make_features()
        tb = MultiColumnTarget(columns=["close", "volume"])
        targets = tb.build_targets(features)
        tensor = tb.extract_target(targets, start=10, length=5)
        assert tensor.shape == (2, 5)  # (n_cols, pred_len)

    def test_log_return_missing_column_raises(self):
        features = self._make_features()
        tb = LogReturnTarget(price_column="nonexistent")
        with pytest.raises(ValueError, match="not in features"):
            tb.build_targets(features)

    def test_multi_column_missing_raises(self):
        features = self._make_features()
        tb = MultiColumnTarget(columns=["close", "nonexistent"])
        with pytest.raises(ValueError, match="not in features"):
            tb.build_targets(features)


# ---------------------------------------------------------------------------
# 13. RegimeAwareDataset with custom TargetBuilder
# ---------------------------------------------------------------------------


class TestRegimeAwareDatasetWithTargetBuilder:
    def test_dataset_with_raw_return_target(self):
        df = _make_raw_5m(2880)
        hourly = aggregate_5m_to_1h(df)
        features = engineer_features(hourly)
        tagger = RegimeTagger(n_regimes=3)
        labels = tagger.fit_predict(features)

        ds = RegimeAwareDataset(
            features, labels, seq_len=16, pred_len=4,
            target_builder=RawReturnTarget(),
        )
        item = ds[0]
        assert item["inputs"].shape == (len(HOURLY_FEATURE_NAMES), 16)
        assert item["target"].shape == (1, 4)

    def test_dataset_with_multi_column_target(self):
        df = _make_raw_5m(2880)
        hourly = aggregate_5m_to_1h(df)
        features = engineer_features(hourly)
        tagger = RegimeTagger(n_regimes=3)
        labels = tagger.fit_predict(features)

        ds = RegimeAwareDataset(
            features, labels, seq_len=16, pred_len=4,
            target_builder=MultiColumnTarget(columns=["close", "volume"]),
        )
        item = ds[0]
        assert item["inputs"].shape == (len(HOURLY_FEATURE_NAMES), 16)
        assert item["target"].shape == (2, 4)  # 2 target columns


# ---------------------------------------------------------------------------
# 14. run_pipeline with custom extension points
# ---------------------------------------------------------------------------


class TestRunPipelineCustom:
    def test_pipeline_with_gmm_and_raw_returns(self):
        df = _make_raw_5m(5000)
        cfg = PipelineConfig(
            train_size=100,
            val_size=30,
            test_size=30,
            step_size=30,
            gap_size=5,
            seq_len=16,
            pred_len=4,
            clustering=GaussianMixtureStrategy(n_components=3),
            target_builder=RawReturnTarget(),
            balance_strength=0.8,
            batch_size=8,
        )
        fold_loaders = run_pipeline(df, cfg)
        assert len(fold_loaders) > 0

        fl = fold_loaders[0]
        for batch in fl.train_loader:
            assert batch["inputs"].dim() == 3
            assert batch["target"].dim() == 3
            break

    def test_pipeline_with_custom_vol_features(self):
        df = _make_raw_5m(5000)
        cfg = PipelineConfig(
            train_size=100,
            val_size=30,
            test_size=30,
            step_size=30,
            gap_size=5,
            seq_len=16,
            pred_len=4,
            vol_features=["realized_vol", "parkinson_vol", "skew"],
            balance_strength=0.8,
            batch_size=8,
        )
        fold_loaders = run_pipeline(df, cfg)
        assert len(fold_loaders) > 0

    def test_pipeline_with_custom_aggregation(self):
        df = _make_raw_5m(10000)
        cfg = PipelineConfig(
            aggregation=OHLCVAggregation(resample_rule="2h"),
            train_size=50,
            val_size=15,
            test_size=15,
            step_size=15,
            gap_size=3,
            seq_len=8,
            pred_len=2,
            batch_size=4,
        )
        fold_loaders = run_pipeline(df, cfg)
        assert len(fold_loaders) > 0
