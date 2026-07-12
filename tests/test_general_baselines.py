from __future__ import annotations

from datetime import datetime, timedelta
import json
from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory
import subprocess
import sys
import textwrap
import unittest
from unittest.mock import patch

import torch

try:
    from bstalignment.general_baseline_profiles import (
        FORMAL_DATASETS,
        FORMAL_HORIZONS,
        resolve_general_profile,
        time_llm_description,
    )
except (ImportError, ModuleNotFoundError):
    FORMAL_DATASETS = ()
    FORMAL_HORIZONS = ()
    resolve_general_profile = None
    time_llm_description = None

try:
    from bstalignment.baseline_adapters import (
        TimeCMACacheProvenance,
        TimeCMAPromptCache,
        FrozenGPT2PromptEncoder,
        build_general_baseline,
        build_time_llm_prompts,
        build_timecma_prompt,
    )
except (ImportError, ModuleNotFoundError):
    TimeCMACacheProvenance = None
    TimeCMAPromptCache = None
    FrozenGPT2PromptEncoder = None
    build_general_baseline = None
    build_time_llm_prompts = None
    build_timecma_prompt = None

try:
    from bstalignment.train_general_baselines import (
        build_general_optimizer,
        build_general_scheduler,
        build_shared_general_datasets,
        general_metrics,
        general_result_record,
        scaler_checksum,
        step_general_batch_scheduler,
        step_general_epoch_scheduler,
        validation_checkpoint_decision,
    )
except (ImportError, ModuleNotFoundError):
    build_general_optimizer = None
    build_general_scheduler = None
    build_shared_general_datasets = None
    general_metrics = None
    general_result_record = None
    scaler_checksum = None
    step_general_batch_scheduler = None
    step_general_epoch_scheduler = None
    validation_checkpoint_decision = None


SOURCES = {
    "PatchTST": ("https://github.com/yuqinie98/PatchTST", "204c21e", "models.PatchTST", "Model", "none"),
    "iTransformer": ("https://github.com/thuml/iTransformer", "c2426e6", "model.iTransformer", "Model", "none"),
    "TimeCMA": ("https://github.com/ChenxiLiu-HNU/TimeCMA", "223e4ae", "models.TimeCMA", "Dual", "timecma"),
    "TimesNet": ("https://github.com/thuml/Time-Series-Library", "4e938a1", "models.TimesNet", "Model", "none"),
    "DLinear": ("https://github.com/cure-lab/LTSF-Linear", "0c11366", "models.DLinear", "Model", "none"),
    "Time-LLM": ("https://github.com/KimMeen/Time-LLM", "b13e881", "models.TimeLLM", "Model", "time_llm"),
}


class GeneralBaselineProfileTests(unittest.TestCase):
    def require_profiles(self):
        self.assertIsNotNone(resolve_general_profile, "general baseline profile resolver must exist")

    def test_all_formal_model_dataset_horizon_profiles_are_source_identified(self):
        self.require_profiles()
        self.assertEqual(tuple(FORMAL_DATASETS), ("ETTm1", "ETTm2", "ETTh1", "ETTh2", "ECL", "Weather"))
        self.assertEqual(tuple(FORMAL_HORIZONS), (96, 192, 336, 720))
        count = 0
        for model, expected_source in SOURCES.items():
            for dataset in FORMAL_DATASETS:
                for horizon in FORMAL_HORIZONS:
                    with self.subTest(model=model, dataset=dataset, horizon=horizon):
                        profile = resolve_general_profile(model, dataset, horizon)
                        source = profile.source
                        self.assertEqual(
                            (source.url, source.commit, source.module, source.class_name, source.prompt_policy),
                            expected_source,
                        )
                        self.assertEqual(profile.seq_len, 36)
                        self.assertEqual(profile.pred_len, horizon)
                        self.assertEqual(profile.features, "M")
                        self.assertIsNone(profile.label_len)
                        self.assertIn("seq_len", profile.protocol_overrides)
                        self.assertTrue(profile.source_evidence)
                        count += 1
        self.assertEqual(count, 144)

    def test_patchtst_profiles_preserve_dataset_specific_source_schedulers(self):
        self.require_profiles()
        etth = resolve_general_profile("PatchTST", "ETTh1", 96)
        ettm = resolve_general_profile("PatchTST", "ETTm1", 96)
        ecl = resolve_general_profile("PatchTST", "ECL", 96)

        self.assertEqual((etth.training.scheduler, etth.training.max_epochs, etth.training.early_stop_patience), ("type3", 100, 100))
        self.assertEqual((ettm.training.scheduler, ettm.training.scheduler_step, ettm.training.pct_start), ("one_cycle", "batch", 0.4))
        self.assertEqual((ecl.training.scheduler, ecl.training.early_stop_patience, ecl.training.pct_start), ("one_cycle", 10, 0.2))
        self.assertEqual(etth.architecture["patch_len"], 16)
        self.assertEqual(etth.architecture["stride"], 8)
        self.assertEqual(etth.patch_adjustments, ())

    def test_itransformer_profiles_preserve_horizon_and_dataset_architecture(self):
        self.require_profiles()
        self.assertEqual(resolve_general_profile("iTransformer", "ETTh1", 192).architecture["d_model"], 256)
        self.assertEqual(resolve_general_profile("iTransformer", "ETTh1", 336).architecture["d_model"], 512)
        ecl = resolve_general_profile("iTransformer", "ECL", 720)
        self.assertEqual((ecl.architecture["e_layers"], ecl.training.lr, ecl.training.batch_size), (3, 5e-4, 16))

    def test_timesnet_profiles_preserve_source_epoch_exceptions(self):
        self.require_profiles()
        self.assertEqual(resolve_general_profile("TimesNet", "ETTm1", 336).training.max_epochs, 3)
        self.assertEqual(resolve_general_profile("TimesNet", "ETTm2", 192).training.max_epochs, 1)
        self.assertEqual(resolve_general_profile("TimesNet", "Weather", 720).training.max_epochs, 1)
        self.assertEqual(resolve_general_profile("TimesNet", "ETTm1", 96).architecture["d_model"], 64)

    def test_dlinear_profiles_preserve_source_learning_rates(self):
        self.require_profiles()
        self.assertEqual(resolve_general_profile("DLinear", "ETTh2", 96).training.lr, 0.05)
        self.assertEqual(resolve_general_profile("DLinear", "ETTm2", 192).training.lr, 0.001)
        self.assertEqual(resolve_general_profile("DLinear", "ETTm2", 336).training.lr, 0.01)
        self.assertFalse(resolve_general_profile("DLinear", "ECL", 720).architecture["individual"])

    def test_timecma_profiles_preserve_script_epoch_budgets_and_delayed_stopping(self):
        self.require_profiles()
        ett = resolve_general_profile("TimeCMA", "ETTm1", 96)
        weather96 = resolve_general_profile("TimeCMA", "Weather", 96)
        weather192 = resolve_general_profile("TimeCMA", "Weather", 192)

        self.assertEqual((ett.training.max_epochs, ett.training.early_stop_start_epoch), (999, 499))
        self.assertEqual((weather96.training.max_epochs, weather96.training.early_stop_start_epoch), (20, 10))
        self.assertEqual((weather192.training.max_epochs, weather192.training.early_stop_start_epoch), (100, 50))
        self.assertEqual((ett.training.optimizer, ett.training.weight_decay, ett.training.gradient_clip), ("adamw", 1e-3, 5.0))

    def test_time_llm_profiles_preserve_horizon_scheduler_exceptions(self):
        self.require_profiles()
        ettm = resolve_general_profile("Time-LLM", "ETTm1", 96)
        etth_cos = resolve_general_profile("Time-LLM", "ETTh1", 336)
        etth2 = resolve_general_profile("Time-LLM", "ETTh2", 720)
        weather = resolve_general_profile("Time-LLM", "Weather", 720)

        self.assertEqual((ettm.training.lr, ettm.training.scheduler, ettm.training.pct_start), (0.001, "one_cycle", 0.2))
        self.assertEqual((etth_cos.training.scheduler, etth_cos.training.cosine_t_max, etth_cos.training.eta_min), ("cosine", 20, 1e-8))
        self.assertEqual((etth2.training.max_epochs, etth2.architecture["d_model"]), (20, 16))
        self.assertEqual((weather.training.max_epochs, weather.architecture["d_ff"]), (15, 128))
        self.assertEqual(ettm.architecture["llm_model_id"], "huggyllama/llama-7b")
        self.assertEqual(ettm.precision, "bf16")

    def test_profile_resolver_rejects_nonformal_inputs(self):
        self.require_profiles()
        for args in (("GraphReportTS", "ETTm1", 96), ("DLinear", "Traffic", 96), ("DLinear", "ETTm1", 48)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                resolve_general_profile(*args)


def _write_package_file(root: Path, relative: str, content: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    for parent in (path.parent, *path.parents):
        if parent == root.parent:
            break
        if root in parent.parents or parent == root:
            init = parent / "__init__.py"
            if parent != root and not init.exists():
                init.write_text("", encoding="utf-8")
    path.write_text(textwrap.dedent(content), encoding="utf-8")


def _fake_source_tree(root: Path) -> None:
    common_model = """
        import torch
        from torch import nn
        class Model(nn.Module):
            marker = MARKER
            def __init__(self, configs):
                super().__init__()
                self.configs = configs
                self.weight = nn.Parameter(torch.ones(()))
            def forward(self, x, *args, **kwargs):
                return x.new_zeros((x.shape[0], self.configs.pred_len, x.shape[2])) + self.weight * 0
    """
    _write_package_file(root / "patchtst" / "PatchTST_supervised", "models/PatchTST.py", common_model.replace("MARKER", repr("patchtst")))
    _write_package_file(root / "itransformer", "model/iTransformer.py", common_model.replace("MARKER", repr("itransformer")))
    _write_package_file(root / "timesnet", "models/TimesNet.py", common_model.replace("MARKER", repr("timesnet")))
    _write_package_file(root / "dlinear", "models/DLinear.py", common_model.replace("MARKER", repr("dlinear")))
    _write_package_file(
        root / "timecma",
        "models/TimeCMA.py",
        """
        import torch
        from torch import nn
        class Dual(nn.Module):
            marker = "timecma"
            def __init__(self, **kwargs):
                super().__init__()
                self.kwargs = kwargs
                self.weight = nn.Parameter(torch.ones(()))
            def forward(self, input_data, input_data_mark, embeddings):
                assert embeddings.shape == (input_data.shape[0], 768, input_data.shape[2])
                return input_data.new_zeros((input_data.shape[0], self.kwargs["pred_len"], input_data.shape[2])) + self.weight * 0
        """,
    )
    _write_package_file(
        root / "time_llm",
        "models/TimeLLM.py",
        """
        import torch
        from torch import nn
        class Model(nn.Module):
            marker = "time_llm"
            def __init__(self, configs):
                super().__init__()
                self.configs = configs
                self.llm_model = nn.Linear(2, 2)
                self.head = nn.Parameter(torch.ones(()))
            def forward(self, x, x_mark, x_dec, y_mark, mask=None):
                return x.new_zeros((x.shape[0], self.configs.pred_len, x.shape[2])) + self.head * 0
        """,
    )


class GeneralBaselineAdapterTests(unittest.TestCase):
    def require_builder(self):
        self.assertIsNotNone(build_general_baseline, "general baseline builder must exist")

    def test_module_import_is_lightweight(self):
        code = "import sys; import bstalignment.baseline_adapters; print(int('transformers' in sys.modules))"
        output = subprocess.check_output([sys.executable, "-c", code], text=True, cwd=Path(__file__).resolve().parents[1])
        self.assertEqual(output.strip(), "0")

    def test_fake_official_classes_receive_source_configs_and_return_m2m_outputs(self):
        self.require_builder()
        channels = {"ETTm1": 7, "ETTm2": 7, "ETTh1": 7, "ETTh2": 7, "ECL": 321, "Weather": 21}
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _fake_source_tree(root)
            for model in SOURCES:
                for dataset in FORMAL_DATASETS:
                    for horizon in FORMAL_HORIZONS:
                        with self.subTest(model=model, dataset=dataset, horizon=horizon):
                            num_features = channels[dataset]
                            args = SimpleNamespace(external_root=root, pred_len=horizon, verify_source_commit=False)
                            adapter = build_general_baseline(model, {"name": dataset, "num_features": num_features}, args)
                            x = torch.randn(1, 36, num_features)
                            if model == "TimeCMA":
                                output = adapter(x, prompt_embeddings=torch.zeros(1, 768, num_features))
                            else:
                                output = adapter(x)
                            self.assertEqual(tuple(output.shape), (1, horizon, num_features))
                            self.assertEqual(adapter.model.marker, model.lower().replace("-", "_"))
                            if hasattr(adapter.model, "configs"):
                                self.assertEqual(adapter.model.configs.seq_len, 36)
                                self.assertEqual(adapter.model.configs.pred_len, horizon)
                                self.assertEqual(adapter.model.configs.enc_in, num_features)
                                self.assertEqual(adapter.model.configs.c_out, num_features)
                            self.assertEqual(adapter.source_identity.commit, SOURCES[model][1])
                            self.assertEqual(adapter.prompt_policy, SOURCES[model][4])

    def test_conflicting_official_packages_are_isolated_between_builds(self):
        self.require_builder()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _fake_source_tree(root)
            args = SimpleNamespace(external_root=root, pred_len=96, verify_source_commit=False)
            first = build_general_baseline("PatchTST", {"name": "ETTm1", "num_features": 2}, args)
            second = build_general_baseline("DLinear", {"name": "ETTm1", "num_features": 2}, args)
        self.assertEqual(first.model.marker, "patchtst")
        self.assertEqual(second.model.marker, "dlinear")

    def test_time_llm_backbone_is_frozen(self):
        self.require_builder()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _fake_source_tree(root)
            args = SimpleNamespace(external_root=root, pred_len=96, verify_source_commit=False)
            adapter = build_general_baseline("Time-LLM", {"name": "Weather", "num_features": 2}, args)
        self.assertTrue(all(not parameter.requires_grad for parameter in adapter.model.llm_model.parameters()))
        self.assertTrue(adapter.model.head.requires_grad)

    def test_formal_source_commit_validation_fails_closed(self):
        self.require_builder()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _fake_source_tree(root)
            args = SimpleNamespace(external_root=root, pred_len=96, verify_source_commit=True)
            with self.assertRaisesRegex((RuntimeError, FileNotFoundError), "commit|checkout|git"):
                build_general_baseline("DLinear", {"name": "ETTm1", "num_features": 2}, args)

    def test_optional_local_source_commits(self):
        self.require_builder()
        root = Path(__file__).resolve().parents[1] / "external"
        required = tuple(root / directory for directory in ("patchtst", "itransformer", "timecma", "timesnet", "dlinear", "time_llm"))
        if not all(path.exists() for path in required):
            self.skipTest("real official repositories unavailable locally")
        for model in SOURCES:
            with self.subTest(model=model):
                profile = resolve_general_profile(model, "ETTm1", 96)
                repo = root / profile.source.repo_dir
                self.assertEqual(subprocess.check_output(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"], text=True).strip(), profile.source.commit)


class FakeFrozenEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0
        self.anchor = torch.nn.Parameter(torch.ones(()), requires_grad=False)

    def encode_last_token(self, prompt: str) -> torch.Tensor:
        self.calls += 1
        return torch.full((768,), float(len(prompt)))


class GeneralPromptContractTests(unittest.TestCase):
    def require_prompt_contracts(self):
        self.assertIsNotNone(build_timecma_prompt, "TimeCMA prompt builder must exist")
        self.assertIsNotNone(TimeCMAPromptCache, "TimeCMA cache must exist")
        self.assertIsNotNone(build_time_llm_prompts, "Time-LLM prompt builder must exist")

    def test_frozen_gpt2_encoder_returns_official_final_token_without_gradients(self):
        self.require_prompt_contracts()
        self.assertIsNotNone(FrozenGPT2PromptEncoder, "frozen GPT-2 encoder contract must exist")

        class FakeTokenizer:
            def encode(self, prompt, return_tensors):
                self.prompt = prompt
                self.return_tensors = return_tensors
                return torch.tensor([[1, 2, 3]])

        class FakeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(()))
            def forward(self, token_ids):
                hidden = torch.arange(24, dtype=torch.float32).reshape(1, 3, 8) * self.weight
                return SimpleNamespace(last_hidden_state=hidden)

        tokenizer = FakeTokenizer()
        model = FakeModel()
        encoder = FrozenGPT2PromptEncoder(
            tokenizer=tokenizer,
            model=model,
            model_id="gpt2",
            model_revision="m1",
            tokenizer_id="gpt2",
            tokenizer_revision="t1",
            precision="float32",
        )
        embedding = encoder.encode_last_token("observed prompt")
        torch.testing.assert_close(embedding, torch.arange(16, 24, dtype=torch.float32))
        self.assertEqual(tokenizer.return_tensors, "pt")
        self.assertTrue(all(not parameter.requires_grad for parameter in encoder.parameters()))

    @staticmethod
    def minute_markers() -> list[datetime]:
        start = datetime(2020, 1, 1, 0, 0)
        return [start + timedelta(minutes=15 * index) for index in range(36)]

    def test_timecma_prompt_matches_official_timestamp_value_and_trend_template(self):
        self.require_prompt_contracts()
        prompt = build_timecma_prompt("ETTm1", torch.arange(36, dtype=torch.float32) + 0.9, self.minute_markers())
        expected_values = ", ".join(str(index) for index in range(36))
        self.assertEqual(
            prompt,
            f"From 01/01/2020 00:00 to 01/01/2020 08:45, the values were {expected_values} every 15 minutes. The total trend value was 35",
        )

    def test_timecma_cache_reuses_fake_final_token_embedding_and_keys_every_provenance_field(self):
        self.require_prompt_contracts()
        provenance = TimeCMACacheProvenance(
            dataset="ETTm1",
            split="train",
            input_len=36,
            source_commit="223e4ae",
            scaler_checksum="scale-a",
            model_id="gpt2",
            model_revision="model-r1",
            tokenizer_id="gpt2",
            tokenizer_revision="token-r1",
            precision="float32",
        )
        encoder = FakeFrozenEncoder()
        with TemporaryDirectory() as directory:
            cache = TimeCMAPromptCache(Path(directory), provenance)
            kwargs = dict(
                history=torch.arange(36, dtype=torch.float32),
                timestamps=self.minute_markers(),
                absolute_sample_index=100,
                variable_index=2,
                forecast_origin=136,
                encoder=encoder,
            )
            first = cache.get_or_create(**kwargs)
            second = cache.get_or_create(**kwargs)
            tensor_path, metadata_path = cache.entry_paths(100, 2)
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.assertEqual(encoder.calls, 1)
        torch.testing.assert_close(first, second)
        self.assertEqual(tuple(first.shape), (768,))
        self.assertTrue(tensor_path.name.startswith("sample-100-variable-2"))
        self.assertEqual(metadata["provenance"], provenance.as_dict())
        self.assertEqual((metadata["observed_start"], metadata["observed_end"], metadata["forecast_origin"]), (100, 135, 136))

    def test_timecma_cache_rejects_future_values_and_tampered_provenance(self):
        self.require_prompt_contracts()
        provenance = TimeCMACacheProvenance("ETTm1", "val", 36, "223e4ae", "scale-a", "gpt2", "m1", "gpt2", "t1", "float32")
        encoder = FakeFrozenEncoder()
        with TemporaryDirectory() as directory:
            cache = TimeCMAPromptCache(Path(directory), provenance)
            with self.assertRaisesRegex(ValueError, "future|forecast origin"):
                cache.get_or_create(torch.arange(36), self.minute_markers(), 100, 0, 135, encoder)
            cache.get_or_create(torch.arange(36), self.minute_markers(), 100, 0, 136, encoder)
            _, metadata_path = cache.entry_paths(100, 0)
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["provenance"]["split"] = "test"
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "provenance"):
                cache.get_or_create(torch.arange(36), self.minute_markers(), 100, 0, 136, encoder)

    def test_time_llm_prompt_matches_official_per_variable_statistics_and_lags(self):
        self.require_prompt_contracts()
        history = torch.stack(
            [torch.linspace(-2.0, 3.0, 36), torch.tensor([1.0, -1.0] * 18)],
            dim=1,
        ).unsqueeze(0)
        description = "A factual dataset."
        prompts = build_time_llm_prompts(history, description, 192)
        means = history.mean(dim=1, keepdim=True).detach()
        stdev = torch.sqrt(torch.var(history, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        normalized = (history - means) / stdev
        flattened = normalized.permute(0, 2, 1).reshape(2, 36, 1)
        corr = torch.fft.irfft(torch.fft.rfft(flattened.permute(0, 2, 1), dim=-1) * torch.conj(torch.fft.rfft(flattened.permute(0, 2, 1), dim=-1)), dim=-1)
        expected_lags = torch.topk(torch.mean(corr, dim=1), 5, dim=-1).indices
        for variable, prompt in enumerate(prompts):
            values = flattened[variable, :, 0]
            expected = (
                f"<|start_prompt|>Dataset description: {description}"
                "Task description: forecast the next 192 steps given the previous 36 steps information; "
                "Input statistics: "
                f"min value {str(torch.min(values).item())}, "
                f"max value {str(torch.max(values).item())}, "
                f"median value {str(torch.median(values).item())}, "
                f"the trend of input is {'upward' if torch.diff(values).sum() > 0 else 'downward'}, "
                f"top 5 lags are : {str(expected_lags[variable].tolist())}<|<end_prompt>|>"
            )
            self.assertEqual(prompt, expected)

    def test_time_llm_descriptions_match_official_prompt_bank(self):
        self.require_prompt_contracts()
        self.assertIn("Electricity Transformer Temperature", time_llm_description("ETTm2"))
        self.assertIn("2075259 measurements", time_llm_description("ECL"))
        self.assertIn("recorded every 10 minutes", time_llm_description("Weather"))


class GeneralTrainingContractTests(unittest.TestCase):
    def require_training_contracts(self):
        self.assertIsNotNone(build_general_optimizer, "general optimizer builder must exist")
        self.assertIsNotNone(general_result_record, "general result schema must exist")

    def test_optimizer_and_scheduler_follow_resolved_profile(self):
        self.require_training_contracts()
        model = torch.nn.Linear(2, 2)
        timecma = resolve_general_profile("TimeCMA", "Weather", 96)
        optimizer = build_general_optimizer(model, timecma)
        scheduler = build_general_scheduler(optimizer, timecma, steps_per_epoch=2)
        self.assertIsInstance(optimizer, torch.optim.AdamW)
        self.assertEqual(optimizer.param_groups[0]["lr"], 1e-3)
        self.assertEqual(optimizer.param_groups[0]["weight_decay"], 1e-3)
        self.assertIsInstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)
        self.assertEqual(scheduler.T_max, 20)

        patch = resolve_general_profile("PatchTST", "ETTm1", 96)
        patch_optimizer = build_general_optimizer(model, patch)
        patch_scheduler = build_general_scheduler(patch_optimizer, patch, steps_per_epoch=2)
        self.assertIsInstance(patch_scheduler, torch.optim.lr_scheduler.OneCycleLR)

    def test_dataset_builder_reuses_task3_train_scaler_for_validation_and_test(self):
        self.require_training_contracts()
        calls = []

        class FakeDataset:
            def __init__(self, dataset_name, data_root, split, input_len, pred_len, scaler=None, fit_scaler=False, **_):
                self.scaler = scaler if scaler is not None else object()
                calls.append((dataset_name, data_root, split, input_len, pred_len, scaler, fit_scaler, self.scaler))

        with patch("bstalignment.data_general.GeneralForecastGraphDataset", FakeDataset):
            train, val, test = build_shared_general_datasets("ECL", "data-root", 192)
        self.assertEqual([call[2] for call in calls], ["train", "val", "test"])
        self.assertEqual([call[3] for call in calls], [36, 36, 36])
        self.assertEqual([call[4] for call in calls], [192, 192, 192])
        self.assertTrue(calls[0][6])
        self.assertFalse(calls[1][6])
        self.assertFalse(calls[2][6])
        self.assertIs(val.scaler, train.scaler)
        self.assertIs(test.scaler, train.scaler)

    def test_scaler_checksum_is_deterministic_and_sensitive_to_fitted_statistics(self):
        self.require_training_contracts()
        first = SimpleNamespace(mean=torch.tensor([1.0, 2.0]).numpy(), std=torch.tensor([3.0, 4.0]).numpy())
        same = SimpleNamespace(mean=torch.tensor([1.0, 2.0]).numpy(), std=torch.tensor([3.0, 4.0]).numpy())
        changed = SimpleNamespace(mean=torch.tensor([1.0, 2.1]).numpy(), std=torch.tensor([3.0, 4.0]).numpy())
        self.assertEqual(scaler_checksum(first), scaler_checksum(same))
        self.assertNotEqual(scaler_checksum(first), scaler_checksum(changed))

    def test_source_epoch_schedulers_apply_type1_and_type3_formulas(self):
        self.require_training_contracts()
        model = torch.nn.Linear(2, 2)
        type1 = resolve_general_profile("DLinear", "ETTh1", 96)
        optimizer = build_general_optimizer(model, type1)
        step_general_epoch_scheduler(None, optimizer, type1, epoch=3)
        self.assertEqual(optimizer.param_groups[0]["lr"], type1.training.lr * 0.5 ** 2)

        type3 = resolve_general_profile("PatchTST", "ETTh1", 96)
        optimizer = build_general_optimizer(model, type3)
        step_general_epoch_scheduler(None, optimizer, type3, epoch=4)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], type3.training.lr * 0.9)

    def test_batch_scheduler_steps_only_for_batch_profiles(self):
        self.require_training_contracts()
        class Counter:
            calls = 0
            def step(self):
                self.calls += 1
        counter = Counter()
        step_general_batch_scheduler(counter, resolve_general_profile("PatchTST", "ETTm1", 96))
        step_general_batch_scheduler(counter, resolve_general_profile("DLinear", "ETTm1", 96))
        self.assertEqual(counter.calls, 1)

    def test_checkpoint_decisions_use_validation_only_and_delayed_stopping(self):
        self.require_training_contracts()
        profile = resolve_general_profile("TimeCMA", "Weather", 96)
        decision = validation_checkpoint_decision(best_mse=0.5, stale=49, val_mse=0.6, epoch=9, profile=profile)
        self.assertFalse(decision.should_stop)
        self.assertEqual(decision.stale, 0)
        decision = validation_checkpoint_decision(best_mse=0.5, stale=49, val_mse=0.6, epoch=10, profile=profile)
        self.assertTrue(decision.should_stop)
        improved = validation_checkpoint_decision(best_mse=0.5, stale=49, val_mse=0.4, epoch=10, profile=profile)
        self.assertTrue(improved.should_save)
        self.assertEqual(improved.best_mse, 0.4)

    def test_metrics_and_result_schema_cover_all_channels_and_provenance(self):
        self.require_training_contracts()
        pred = torch.tensor([[[1.0, 3.0], [2.0, 5.0]]])
        target = torch.tensor([[[0.0, 1.0], [2.0, 1.0]]])
        metrics = general_metrics(pred, target)
        self.assertEqual(metrics, {"mse": 5.25, "mae": 1.75})
        profile = resolve_general_profile("DLinear", "ETTm1", 96)
        record = general_result_record(
            profile=profile,
            seed=2021,
            metrics=metrics,
            scaler_checksum="scale-a",
            best_epoch=7,
            best_val_mse=0.25,
            prompt_provenance=None,
        )
        self.assertEqual(record["model"], "DLinear")
        self.assertEqual(record["dataset"], "ETTm1")
        self.assertEqual(record["horizon"], 96)
        self.assertEqual(record["metrics_space"], "standardized")
        self.assertEqual(record["source"]["commit"], "0c11366")
        self.assertEqual(record["training"]["optimizer"], "adam")
        self.assertEqual(record["protocol_overrides"]["seq_len"], 36)
        self.assertEqual(record["selection"], {"metric": "validation_mse", "best_epoch": 7, "best_val_mse": 0.25})


if __name__ == "__main__":
    unittest.main()
