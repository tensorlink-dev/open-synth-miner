"""Tests for the ResearchSession class and quick_experiment helper."""
from __future__ import annotations

import pytest
import torch

from osa.research.agent_api import ResearchSession, quick_experiment


# ── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture
def session():
    return ResearchSession()


@pytest.fixture
def basic_experiment(session):
    """A minimal experiment config for reuse."""
    return session.create_experiment(
        blocks=["TransformerBlock", "LSTMBlock"],
        head="gbm",
        d_model=16,
        feature_dim=4,
        seq_len=16,
        horizon=6,
        n_paths=10,
        batch_size=2,
        lr=0.001,
    )


# ── Constructor ──────────────────────────────────────────────────────────


class TestConstructor:
    def test_no_args(self):
        session = ResearchSession()
        assert session is not None

    def test_starts_empty(self, session):
        s = session.summary()
        assert s["num_experiments"] == 0
        assert s["results"] == []


# ── Discovery methods ────────────────────────────────────────────────────


class TestDiscovery:
    def test_list_blocks_returns_list(self, session):
        blocks = session.list_blocks()
        assert isinstance(blocks, list)
        assert len(blocks) > 0

    def test_list_blocks_has_required_keys(self, session):
        blocks = session.list_blocks()
        for b in blocks:
            assert "name" in b
            assert "cost" in b
            assert "best_for" in b

    def test_list_blocks_includes_core(self, session):
        names = {b["name"] for b in session.list_blocks()}
        assert "transformerblock" in names
        assert "lstmblock" in names

    def test_list_heads_returns_list(self, session):
        heads = session.list_heads()
        assert isinstance(heads, list)
        assert len(heads) > 0

    def test_list_heads_has_required_keys(self, session):
        heads = session.list_heads()
        for h in heads:
            assert "name" in h
            assert "expressiveness" in h
            assert "description" in h

    def test_list_heads_includes_core(self, session):
        names = {h["name"] for h in session.list_heads()}
        assert "gbm" in names
        assert "sde" in names
        assert "horizon" in names

    def test_list_presets_returns_list(self, session):
        presets = session.list_presets()
        assert isinstance(presets, list)
        assert len(presets) > 0

    def test_list_presets_has_required_keys(self, session):
        presets = session.list_presets()
        for p in presets:
            assert "name" in p
            assert "head" in p
            assert "blocks" in p
            assert "tags" in p

    def test_list_presets_json_serializable(self, session):
        import json
        presets = session.list_presets()
        json.dumps(presets)  # Should not raise


# ── Experiment construction ──────────────────────────────────────────────


class TestCreateExperiment:
    def test_returns_dict(self, session):
        exp = session.create_experiment(
            blocks=["TransformerBlock"],
            head="gbm",
            d_model=16,
            feature_dim=4,
            seq_len=16,
            horizon=6,
            n_paths=10,
            batch_size=2,
            lr=0.001,
        )
        assert isinstance(exp, dict)

    def test_config_structure(self, basic_experiment):
        exp = basic_experiment
        assert "model" in exp
        assert "training" in exp
        assert "backbone" in exp["model"]
        assert "head" in exp["model"]
        assert "blocks" in exp["model"]["backbone"]
        assert "d_model" in exp["model"]["backbone"]
        assert "_target_" in exp["model"]["head"]

    def test_training_section(self, basic_experiment):
        t = basic_experiment["training"]
        assert t["horizon"] == 6
        assert t["n_paths"] == 10
        assert t["batch_size"] == 2
        assert t["lr"] == 0.001

    def test_unknown_block_raises(self, session):
        with pytest.raises(KeyError, match="Unknown block"):
            session.create_experiment(
                blocks=["NonExistentBlock"],
                head="gbm",
                d_model=16,
                feature_dim=4,
                seq_len=16,
                horizon=6,
                n_paths=10,
                batch_size=2,
                lr=0.001,
            )

    def test_unknown_head_raises(self, session):
        with pytest.raises(KeyError, match="Unknown head"):
            session.create_experiment(
                blocks=["TransformerBlock"],
                head="nonexistent_head",
                d_model=16,
                feature_dim=4,
                seq_len=16,
                horizon=6,
                n_paths=10,
                batch_size=2,
                lr=0.001,
            )

    def test_block_kwargs_length_mismatch_raises(self, session):
        with pytest.raises(ValueError, match="block_kwargs"):
            session.create_experiment(
                blocks=["TransformerBlock", "LSTMBlock"],
                head="gbm",
                d_model=16,
                feature_dim=4,
                seq_len=16,
                horizon=6,
                n_paths=10,
                batch_size=2,
                lr=0.001,
                block_kwargs=[{"nhead": 2}],  # only 1 but 2 blocks
            )

    def test_head_class_name_accepted(self, session):
        """Head names can be class names like 'GBMHead'."""
        exp = session.create_experiment(
            blocks=["TransformerBlock"],
            head="GBMHead",
            d_model=16,
            feature_dim=4,
            seq_len=16,
            horizon=6,
            n_paths=10,
            batch_size=2,
            lr=0.001,
        )
        assert "GBMHead" in exp["model"]["head"]["_target_"]

    def test_case_insensitive_blocks(self, session):
        """Block names should be case-insensitive."""
        exp = session.create_experiment(
            blocks=["transformerblock"],
            head="gbm",
            d_model=16,
            feature_dim=4,
            seq_len=16,
            horizon=6,
            n_paths=10,
            batch_size=2,
            lr=0.001,
        )
        assert exp["model"]["backbone"]["blocks"] == ["transformerblock"]

    def test_head_kwargs_forwarded(self, session):
        exp = session.create_experiment(
            blocks=["TransformerBlock"],
            head="mixture_density",
            d_model=16,
            feature_dim=4,
            seq_len=16,
            horizon=6,
            n_paths=10,
            batch_size=2,
            lr=0.001,
            head_kwargs={"n_components": 5},
        )
        assert exp["model"]["head"]["n_components"] == 5


# ── Validation ───────────────────────────────────────────────────────────


class TestValidation:
    def test_validate_valid_config(self, session, basic_experiment):
        result = session.validate(basic_experiment)
        assert result["valid"] is True
        assert result["errors"] == []
        assert result["param_count"] > 0

    def test_validate_returns_param_count(self, session, basic_experiment):
        result = session.validate(basic_experiment)
        assert isinstance(result["param_count"], int)
        assert result["param_count"] > 0

    def test_validate_warnings_for_high_lr(self, session):
        exp = session.create_experiment(
            blocks=["TransformerBlock"],
            head="gbm",
            d_model=16,
            feature_dim=4,
            seq_len=16,
            horizon=6,
            n_paths=10,
            batch_size=2,
            lr=0.1,  # high
        )
        result = session.validate(exp)
        assert any("Learning rate" in w for w in result["warnings"])

    def test_validate_warnings_for_few_paths(self, session):
        exp = session.create_experiment(
            blocks=["TransformerBlock"],
            head="gbm",
            d_model=16,
            feature_dim=4,
            seq_len=16,
            horizon=6,
            n_paths=5,  # very few
            batch_size=2,
            lr=0.001,
        )
        result = session.validate(exp)
        assert any("paths" in w.lower() for w in result["warnings"])

    def test_describe_returns_full_info(self, session, basic_experiment):
        desc = session.describe(basic_experiment)
        assert "blocks" in desc
        assert "head" in desc
        assert "param_count" in desc
        assert "training" in desc
        assert "validation" in desc
        assert desc["param_count"] > 0


# ── Execution ────────────────────────────────────────────────────────────


class TestExecution:
    def test_run_returns_ok(self, session, basic_experiment):
        result = session.run(basic_experiment, epochs=1, name="test-run")
        assert result["status"] == "ok"
        assert "metrics" in result
        assert "crps" in result["metrics"]
        assert "sharpness" in result["metrics"]
        assert "log_likelihood" in result["metrics"]
        assert result["param_count"] > 0
        assert result["training_time_s"] >= 0

    def test_run_accumulates_state(self, session, basic_experiment):
        assert session.summary()["num_experiments"] == 0
        session.run(basic_experiment, epochs=1, name="first")
        assert session.summary()["num_experiments"] == 1
        session.run(basic_experiment, epochs=1, name="second")
        assert session.summary()["num_experiments"] == 2

    def test_run_never_raises(self, session):
        """Even broken configs should return error dict, not raise."""
        bad_exp = {
            "model": {
                "backbone": {
                    "blocks": ["TransformerBlock"],
                    "d_model": 16,
                    "feature_dim": 4,
                    "seq_len": 16,
                    "block_kwargs": [],
                },
                "head": {"_target_": "osa.models.heads.NONEXISTENT", "latent_size": 16},
            },
            "training": {"horizon": 6, "n_paths": 10, "batch_size": 2, "lr": 0.001},
        }
        result = session.run(bad_exp, name="broken")
        assert result["status"] == "error"
        assert "error" in result
        assert "traceback" in result

    def test_run_with_data_loader_kwarg(self, session, basic_experiment):
        """run() must accept data_loader= as optional kwarg."""
        result = session.run(basic_experiment, epochs=1, data_loader=None)
        assert result["status"] == "ok"

    def test_run_preset(self, session):
        result = session.run_preset("baseline_gbm", epochs=1)
        assert result["status"] == "ok"
        assert "metrics" in result

    def test_run_preset_with_overrides(self, session):
        result = session.run_preset(
            "baseline_gbm",
            epochs=1,
            overrides={"d_model": 16, "n_paths": 10},
        )
        assert result["status"] == "ok"

    def test_run_preset_unknown_raises(self, session):
        with pytest.raises(KeyError, match="Unknown preset"):
            session.run_preset("nonexistent_preset")

    def test_sweep_all_presets(self, session):
        """Sweep with a subset of presets to keep the test fast."""
        result = session.sweep(preset_names=["baseline_gbm", "sde_transformer"], epochs=1)
        assert "ranking" in result
        assert len(result["ranking"]) >= 2

    def test_sweep_none_runs_all(self, session):
        """sweep(preset_names=None) runs all presets."""
        result = session.sweep(preset_names=None, epochs=1)
        assert "ranking" in result
        from osa.research.agent_api import _PRESETS
        assert len(result["ranking"]) == len(_PRESETS)


# ── Session state ────────────────────────────────────────────────────────


class TestSessionState:
    def test_compare_empty(self, session):
        result = session.compare()
        assert result == {"ranking": []}

    def test_compare_sorted_by_crps(self, session, basic_experiment):
        session.run(basic_experiment, epochs=1, name="a")
        session.run(basic_experiment, epochs=1, name="b")
        result = session.compare()
        ranking = result["ranking"]
        assert len(ranking) == 2
        # Check sorted (ascending CRPS)
        assert ranking[0]["crps"] <= ranking[1]["crps"]

    def test_compare_includes_required_keys(self, session, basic_experiment):
        session.run(basic_experiment, epochs=1, name="test")
        ranking = session.compare()["ranking"]
        entry = ranking[0]
        assert "name" in entry
        assert "crps" in entry
        assert "param_count" in entry
        assert "experiment" in entry

    def test_summary(self, session, basic_experiment):
        session.run(basic_experiment, epochs=1, name="my-run")
        s = session.summary()
        assert s["num_experiments"] == 1
        assert s["results"][0]["name"] == "my-run"
        assert s["results"][0]["status"] == "ok"

    def test_clear(self, session, basic_experiment):
        session.run(basic_experiment, epochs=1)
        assert session.summary()["num_experiments"] == 1
        session.clear()
        assert session.summary()["num_experiments"] == 0
        assert session.compare() == {"ranking": []}


# ── quick_experiment ─────────────────────────────────────────────────────


class TestQuickExperiment:
    def test_returns_result(self):
        result = quick_experiment(
            blocks=["TransformerBlock"],
            head="gbm",
            d_model=16,
            feature_dim=4,
            seq_len=16,
            horizon=6,
            n_paths=10,
            batch_size=2,
            lr=0.001,
            epochs=1,
        )
        assert result["status"] == "ok"
        assert "metrics" in result

    def test_uses_defaults(self):
        result = quick_experiment(
            blocks=["TransformerBlock"],
            head="gbm",
        )
        assert result["status"] == "ok"


# ── Import path ──────────────────────────────────────────────────────────


class TestImportPaths:
    def test_import_from_osa_research_agent_api(self):
        from osa.research.agent_api import ResearchSession as RS
        assert RS is ResearchSession

    def test_import_from_osa_research(self):
        from osa.research import ResearchSession as RS
        assert RS is ResearchSession

    def test_import_quick_experiment_from_osa(self):
        from osa.research.agent_api import quick_experiment as qe
        assert qe is quick_experiment


# ── Different head types ─────────────────────────────────────────────────


class TestHeadVariants:
    """Ensure create_experiment + validate works for every head type."""

    @pytest.mark.parametrize("head", ["gbm", "sde", "simple_horizon", "mixture_density", "vol_term_structure"])
    def test_validate_head(self, session, head):
        exp = session.create_experiment(
            blocks=["TransformerBlock"],
            head=head,
            d_model=16,
            feature_dim=4,
            seq_len=16,
            horizon=6,
            n_paths=10,
            batch_size=2,
            lr=0.001,
        )
        result = session.validate(exp)
        assert result["valid"] is True, f"Head {head} failed validation: {result['errors']}"

    @pytest.mark.parametrize("head", ["gbm", "sde", "mixture_density"])
    def test_run_head(self, session, head):
        exp = session.create_experiment(
            blocks=["TransformerBlock"],
            head=head,
            d_model=16,
            feature_dim=4,
            seq_len=16,
            horizon=6,
            n_paths=10,
            batch_size=2,
            lr=0.001,
        )
        result = session.run(exp, epochs=1, name=f"test-{head}")
        assert result["status"] == "ok", f"Head {head} run failed: {result}"
