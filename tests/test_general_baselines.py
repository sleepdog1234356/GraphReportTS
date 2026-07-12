from __future__ import annotations

from dataclasses import replace
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
    from bstalignment.baseline_adapters import validate_source_checkout
except (ImportError, ModuleNotFoundError):
    validate_source_checkout = None

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

try:
    from bstalignment.train_general_baselines import (
        clip_general_gradients,
        collate_general_baseline_batch,
        forward_general_baseline_batch,
        source_time_markers,
        step_general_optimizer,
    )
except (ImportError, ModuleNotFoundError):
    clip_general_gradients = None
    collate_general_baseline_batch = None
    forward_general_baseline_batch = None
    source_time_markers = None
    step_general_optimizer = None


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
    marked_model = """
        import torch
        from torch import nn
        class Model(nn.Module):
            marker = MARKER
            def __init__(self, configs):
                super().__init__()
                self.configs = configs
                self.weight = nn.Parameter(torch.ones(()))
                self.last_encoder_mark = None
                self.last_decoder_mark = None
            def forward(self, x, x_mark, x_dec, y_mark, *args, **kwargs):
                if x_mark is None:
                    raise AssertionError("official encoder time markers are required")
                self.last_encoder_mark = x_mark.detach().clone()
                self.last_decoder_mark = None if y_mark is None else y_mark.detach().clone()
                return x.new_zeros((x.shape[0], self.configs.pred_len, x.shape[2])) + self.weight * 0
    """
    _write_package_file(root / "itransformer", "model/iTransformer.py", marked_model.replace("MARKER", repr("itransformer")))
    _write_package_file(root / "timesnet", "models/TimesNet.py", marked_model.replace("MARKER", repr("timesnet")))
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
    sys.modules.pop("fake_transformers_for_task6", None)
    _write_package_file(
        root / "time_llm",
        "fake_transformers_for_task6.py",
        """
        from types import SimpleNamespace
        import torch
        from torch import nn

        calls = []

        class LlamaConfig:
            @classmethod
            def from_pretrained(cls, path, *args, **kwargs):
                calls.append(("config", str(path), dict(kwargs)))
                return SimpleNamespace()

        class FakeBackbone(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.ones(2, 2))

        class ModelLoaderBase:
            @classmethod
            def from_pretrained(cls, path, *args, **kwargs):
                calls.append(("model", str(path), dict(kwargs)))
                return FakeBackbone()

        class LlamaModel(ModelLoaderBase):
            pass

        class TokenizerLoaderBase:
            @classmethod
            def from_pretrained(cls, path, *args, **kwargs):
                calls.append(("tokenizer", str(path), dict(kwargs)))
                return SimpleNamespace()

        class LlamaTokenizer(TokenizerLoaderBase):
            pass
        """,
    )
    _write_package_file(
        root / "time_llm",
        "models/TimeLLM.py",
        """
        import torch
        from torch import nn
        from fake_transformers_for_task6 import LlamaConfig, LlamaModel, LlamaTokenizer

        class Model(nn.Module):
            marker = "time_llm"
            def __init__(self, configs):
                super().__init__()
                self.configs = configs
                self.llm_config = LlamaConfig.from_pretrained("huggyllama/llama-7b")
                self.llm_model = LlamaModel.from_pretrained("huggyllama/llama-7b")
                self.tokenizer = LlamaTokenizer.from_pretrained("huggyllama/llama-7b")
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
                            elif model in {"iTransformer", "TimesNet"}:
                                mark_dim = 5 if dataset in {"ETTm1", "ETTm2", "Weather"} else 4
                                output = adapter(
                                    x,
                                    time_mark=torch.zeros(1, 36, mark_dim),
                                    decoder_time_mark=torch.zeros(1, horizon, mark_dim),
                                )
                            else:
                                output = adapter(x)
                            self.assertEqual(tuple(output.shape), (1, horizon, num_features))
                            self.assertEqual(adapter.model.marker, model.lower().replace("-", "_"))
                            if hasattr(adapter.model, "configs"):
                                self.assertEqual(adapter.model.configs.seq_len, 36)
                                self.assertEqual(adapter.model.configs.pred_len, horizon)
                                self.assertEqual(adapter.model.configs.enc_in, num_features)
                                self.assertEqual(adapter.model.configs.c_out, num_features)
                                if model in {"iTransformer", "TimesNet"}:
                                    expected_freq = "t" if dataset in {"ETTm1", "ETTm2", "Weather"} else "h"
                                    self.assertEqual(adapter.model.configs.freq, expected_freq)
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

    def test_time_llm_local_loader_patches_are_scoped_and_runtime_specific(self):
        self.require_builder()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _fake_source_tree(root)
            model_a = root / "weights" / "model-a"
            tokenizer_a = root / "weights" / "tokenizer-a"
            model_b = root / "weights" / "model-b"
            tokenizer_b = root / "weights" / "tokenizer-b"
            for path in (model_a, tokenizer_a, model_b, tokenizer_b):
                path.mkdir(parents=True)

            sys.path.insert(0, str(root / "time_llm"))
            try:
                import fake_transformers_for_task6 as fake_hf
            finally:
                sys.path.pop(0)
            original_descriptors = {
                name: (
                    "from_pretrained" in vars(getattr(fake_hf, name)),
                    vars(getattr(fake_hf, name)).get("from_pretrained"),
                )
                for name in ("LlamaConfig", "LlamaModel", "LlamaTokenizer")
            }

            def build(model_path, tokenizer_path, model_revision, tokenizer_revision):
                return build_general_baseline(
                    "Time-LLM",
                    {"name": "Weather", "num_features": 2},
                    SimpleNamespace(
                        external_root=root,
                        pred_len=96,
                        verify_source_commit=False,
                        local_llm_path=model_path,
                        local_tokenizer_path=tokenizer_path,
                        llm_model_revision=model_revision,
                        tokenizer_revision=tokenizer_revision,
                        precision="bf16",
                    ),
                )

            first = build(model_a, tokenizer_a, "model-rev-a", "token-rev-a")
            for name, (had_own_descriptor, descriptor) in original_descriptors.items():
                self.assertEqual("from_pretrained" in vars(getattr(fake_hf, name)), had_own_descriptor)
                self.assertIs(vars(getattr(fake_hf, name)).get("from_pretrained"), descriptor)
            first_calls = list(fake_hf.calls)
            fake_hf.calls.clear()
            second = build(model_b, tokenizer_b, "model-rev-b", "token-rev-b")
            for name, (had_own_descriptor, descriptor) in original_descriptors.items():
                self.assertEqual("from_pretrained" in vars(getattr(fake_hf, name)), had_own_descriptor)
                self.assertIs(vars(getattr(fake_hf, name)).get("from_pretrained"), descriptor)
            second_calls = list(fake_hf.calls)

        self.assertEqual([call[1] for call in first_calls], [str(model_a.resolve()), str(model_a.resolve()), str(tokenizer_a.resolve())])
        self.assertEqual([call[2]["revision"] for call in first_calls], ["model-rev-a", "model-rev-a", "token-rev-a"])
        self.assertEqual([call[1] for call in second_calls], [str(model_b.resolve()), str(model_b.resolve()), str(tokenizer_b.resolve())])
        self.assertEqual([call[2]["revision"] for call in second_calls], ["model-rev-b", "model-rev-b", "token-rev-b"])
        for adapter, expected_model, expected_tokenizer, model_revision, tokenizer_revision in (
            (first, model_a, tokenizer_a, "model-rev-a", "token-rev-a"),
            (second, model_b, tokenizer_b, "model-rev-b", "token-rev-b"),
        ):
            provenance = adapter.runtime_provenance["time_llm"]
            self.assertEqual(provenance["model_path"], str(expected_model.resolve()))
            self.assertEqual(provenance["tokenizer_path"], str(expected_tokenizer.resolve()))
            self.assertEqual(provenance["model_revision"], model_revision)
            self.assertEqual(provenance["tokenizer_revision"], tokenizer_revision)
            self.assertEqual(provenance["precision"], "bf16")
            self.assertEqual(provenance["backbone_dtype"], "torch.float32")
        sys.modules.pop("fake_transformers_for_task6", None)

    def test_source_validation_records_full_sha_and_rejects_tracked_dirty_checkout(self):
        self.assertIsNotNone(validate_source_checkout, "full source checkout validator must exist")
        with TemporaryDirectory() as directory:
            root = Path(directory)
            repo = root / "dlinear"
            repo.mkdir()
            subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
            subprocess.run(["git", "config", "user.email", "task6@example.com"], cwd=repo, check=True)
            subprocess.run(["git", "config", "user.name", "Task 6"], cwd=repo, check=True)
            tracked = repo / "tracked.py"
            tracked.write_text("clean = True\n", encoding="utf-8")
            subprocess.run(["git", "add", "tracked.py"], cwd=repo, check=True)
            subprocess.run(["git", "commit", "-q", "-m", "fixture"], cwd=repo, check=True)
            full_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
            source = replace(resolve_general_profile("DLinear", "ETTm1", 96).source, commit=full_sha[:7])

            provenance = validate_source_checkout(root, source)
            self.assertEqual(provenance.full_sha, full_sha)
            self.assertEqual(provenance.manifest_revision, full_sha[:7])
            self.assertTrue(provenance.verified)

            (repo / "untracked.txt").write_text("allowed\n", encoding="utf-8")
            self.assertEqual(validate_source_checkout(root, source).full_sha, full_sha)

            tracked.write_text("clean = False\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "tracked.*dirty|dirty.*tracked"):
                validate_source_checkout(root, source)
            subprocess.run(["git", "restore", "tracked.py"], cwd=repo, check=True)
            with self.assertRaisesRegex(RuntimeError, "pinned|revision|commit"):
                validate_source_checkout(root, replace(source, commit="deadbee"))

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
                provenance = validate_source_checkout(root, profile.source)
                self.assertTrue(provenance.full_sha.startswith(profile.source.commit))
                self.assertFalse(provenance.tracked_dirty)


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

    def test_source_time_markers_match_official_hourly_and_minute_cadence_dimensions(self):
        self.assertIsNotNone(source_time_markers, "source-format time marker helper must exist")
        hourly = [datetime(2020, 1, 1) + timedelta(hours=index) for index in range(36)]
        minute = [datetime(2020, 1, 1) + timedelta(minutes=15 * index) for index in range(36)]
        hourly_mark = source_time_markers("ETTh1", hourly)
        minute_mark = source_time_markers("ETTm1", minute)
        weather_mark = source_time_markers("Weather", minute)
        self.assertEqual(tuple(hourly_mark.shape), (36, 4))
        self.assertEqual(tuple(minute_mark.shape), (36, 5))
        self.assertEqual(tuple(weather_mark.shape), (36, 5))
        torch.testing.assert_close(
            hourly_mark[0],
            torch.tensor([-0.5, 2 / 6 - 0.5, -0.5, -0.5], dtype=torch.float32),
        )
        torch.testing.assert_close(
            minute_mark[0],
            torch.tensor([-0.5, -0.5, 2 / 6 - 0.5, -0.5, -0.5], dtype=torch.float32),
        )

    def test_task3_timestamp_collation_reaches_official_marker_forward_arguments(self):
        self.assertIsNotNone(collate_general_baseline_batch, "baseline timestamp collator must exist")
        self.assertIsNotNone(forward_general_baseline_batch, "baseline batch forward helper must exist")
        history_timestamps = tuple(datetime(2020, 1, 1) + timedelta(minutes=15 * index) for index in range(36))
        target_timestamps = tuple(datetime(2020, 1, 1, 9) + timedelta(minutes=15 * index) for index in range(96))
        sample = {
            "series_id": "ETTm1",
            "history_scaled": torch.randn(36, 2),
            "target_scaled": torch.randn(96, 2),
            "timestamp_markers": {"history": history_timestamps, "target": target_timestamps},
            "start_index": 100,
            "columns": ("a", "b"),
        }
        batch = collate_general_baseline_batch([sample])
        self.assertEqual(tuple(batch["x_mark"].shape), (1, 36, 5))
        self.assertEqual(tuple(batch["y_mark"].shape), (1, 96, 5))
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _fake_source_tree(root)
            args = SimpleNamespace(external_root=root, pred_len=96, verify_source_commit=False)
            for name in ("iTransformer", "TimesNet"):
                with self.subTest(name=name):
                    adapter = build_general_baseline(name, {"name": "ETTm1", "num_features": 2}, args)
                    output = forward_general_baseline_batch(adapter, batch)
                    self.assertEqual(tuple(output.shape), (1, 96, 2))
                    torch.testing.assert_close(adapter.model.last_encoder_mark, batch["x_mark"])
                    torch.testing.assert_close(adapter.model.last_decoder_mark, batch["y_mark"])

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

    def test_checkpoint_decisions_accumulate_stale_before_delayed_stop_gate(self):
        self.require_training_contracts()
        profile = resolve_general_profile("TimeCMA", "Weather", 96)
        best_mse, stale = 0.5, 0
        for epoch in range(1, 50):
            decision = validation_checkpoint_decision(
                best_mse=best_mse, stale=stale, val_mse=0.6, epoch=epoch, profile=profile
            )
            best_mse, stale = decision.best_mse, decision.stale
            self.assertEqual(stale, epoch)
            self.assertFalse(decision.should_stop)
        decision = validation_checkpoint_decision(
            best_mse=best_mse, stale=stale, val_mse=0.6, epoch=50, profile=profile
        )
        self.assertEqual(decision.stale, 50)
        self.assertTrue(decision.should_stop)
        improved = validation_checkpoint_decision(best_mse=0.5, stale=49, val_mse=0.4, epoch=50, profile=profile)
        self.assertTrue(improved.should_save)
        self.assertEqual(improved.best_mse, 0.4)
        self.assertEqual(improved.stale, 0)

    def test_optimizer_step_applies_gradient_clip_then_optimizer_then_batch_scheduler(self):
        self.assertIsNotNone(clip_general_gradients, "general gradient clipping helper must exist")
        self.assertIsNotNone(step_general_optimizer, "general optimizer step helper must exist")
        base = resolve_general_profile("PatchTST", "ETTm1", 96)
        profile = replace(base, training=replace(base.training, gradient_clip=0.5))
        model = torch.nn.Linear(2, 1, bias=False)
        model(torch.tensor([[100.0, -100.0]])).sum().backward()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        events = []
        original_clip = torch.nn.utils.clip_grad_norm_
        original_step = optimizer.step

        def record_clip(parameters, max_norm):
            events.append(("clip", max_norm))
            return original_clip(parameters, max_norm)

        def record_step(*args, **kwargs):
            events.append(("optimizer", None))
            return original_step(*args, **kwargs)

        class Scheduler:
            def step(self):
                events.append(("scheduler", None))

        optimizer.step = record_step
        with patch("torch.nn.utils.clip_grad_norm_", side_effect=record_clip):
            total_norm = step_general_optimizer(model, optimizer, profile, Scheduler())
        self.assertGreater(float(total_norm), 0.5)
        self.assertEqual(events, [("clip", 0.5), ("optimizer", None), ("scheduler", None)])

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
            runtime_provenance={
                "source_checkout": {
                    "manifest_revision": "0c11366",
                    "full_sha": "b" * 40,
                    "verified": True,
                    "tracked_dirty": False,
                },
                "time_llm": None,
            },
        )
        self.assertEqual(record["model"], "DLinear")
        self.assertEqual(record["dataset"], "ETTm1")
        self.assertEqual(record["horizon"], 96)
        self.assertEqual(record["metrics_space"], "standardized")
        self.assertEqual(record["source"]["commit"], "0c11366")
        self.assertEqual(record["training"]["optimizer"], "adam")
        self.assertEqual(record["protocol_overrides"]["seq_len"], 36)
        self.assertEqual(record["selection"], {"metric": "validation_mse", "best_epoch": 7, "best_val_mse": 0.25})

    def test_time_llm_result_requires_actual_runtime_paths_revisions_and_precision(self):
        self.require_training_contracts()
        profile = resolve_general_profile("Time-LLM", "ETTm1", 96)
        base_kwargs = dict(
            profile=profile,
            seed=2021,
            metrics={"mse": 0.1, "mae": 0.2},
            scaler_checksum="scale-a",
            best_epoch=7,
            best_val_mse=0.25,
            prompt_provenance=None,
        )
        placeholder = {
            "source_checkout": {"manifest_revision": "b13e881", "full_sha": "a" * 40, "verified": True},
            "time_llm": {
                "model_path": "required-at-runtime",
                "tokenizer_path": "required-at-runtime",
                "model_revision": "required-at-runtime",
                "tokenizer_revision": "required-at-runtime",
                "precision": "bf16",
                "backbone_dtype": "torch.float32",
            },
        }
        with self.assertRaisesRegex(ValueError, "runtime provenance|placeholder|model_path"):
            general_result_record(**base_kwargs, runtime_provenance=placeholder)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            model_path = (root / "model").resolve()
            tokenizer_path = (root / "tokenizer").resolve()
            model_path.mkdir()
            tokenizer_path.mkdir()
            runtime = {
                "source_checkout": {"manifest_revision": "b13e881", "full_sha": "a" * 40, "verified": True},
                "time_llm": {
                    "model_path": str(model_path),
                    "tokenizer_path": str(tokenizer_path),
                    "model_revision": "model-rev-1",
                    "tokenizer_revision": "token-rev-1",
                    "precision": "bf16",
                    "backbone_dtype": "torch.float32",
                },
            }
            record = general_result_record(**base_kwargs, runtime_provenance=runtime)
        self.assertEqual(record["runtime_provenance"], runtime)

    def test_trainer_imports_core_numerics_but_not_external_models_transformers_or_cuda(self):
        code = textwrap.dedent(
            """
            import json, sys
            import bstalignment.train_general_baselines
            import torch
            forbidden = [
                name for name in sys.modules
                if name == 'transformers' or name.startswith('transformers.')
                or name == 'models' or name.startswith('models.')
                or name == 'model' or name.startswith('model.')
                or name == 'layers' or name.startswith('layers.')
                or name == 'data_provider' or name.startswith('data_provider.')
                or name == 'utils' or name.startswith('utils.')
            ]
            print(json.dumps({
                'numpy': 'numpy' in sys.modules,
                'torch': 'torch' in sys.modules,
                'forbidden': forbidden,
                'cuda_initialized': torch.cuda.is_initialized(),
            }))
            """
        )
        output = subprocess.check_output(
            [sys.executable, "-c", code], text=True, cwd=Path(__file__).resolve().parents[1]
        )
        state = json.loads(output)
        self.assertTrue(state["numpy"])
        self.assertTrue(state["torch"])
        self.assertEqual(state["forbidden"], [])
        self.assertFalse(state["cuda_initialized"])


if __name__ == "__main__":
    unittest.main()
