from __future__ import annotations

import math
import unittest
from dataclasses import replace

import torch

from bstalignment.v2.edges import (
    RELATIONS,
    SparseEdgeBuilder,
    _difference_correlation,
)
from bstalignment.v2.contracts import GraphReportTSv2Config
from bstalignment.v2.model import BatteryGraphReportTSv2, GraphReportTSv2
from bstalignment.v2.patching import AdaptivePatchifier


def _reference_edges(builder, patches, history, observed):
    """The original list-based implementation, retained only as a test oracle."""

    batch, nodes = patches.node_mask.shape
    variable_meta = patches.variable_index.tolist()
    scale_meta = patches.scale_index.tolist()
    patch_meta = patches.patch_index.tolist()
    start_meta = patches.start.tolist()
    end_meta = patches.end.tolist()
    lookup = {(scale_meta[n], patch_meta[n], variable_meta[n]): n for n in range(nodes)}
    variable_nodes = {}
    for node in range(nodes):
        variable_nodes.setdefault((scale_meta[node], variable_meta[node]), []).append(node)
    all_scores = builder._variable_scores(patches, history, observed)

    result = {name: [] for name in RELATIONS}
    for batch_index in range(batch):
        valid_nodes = patches.node_mask[batch_index].tolist()
        valid_variables = torch.nonzero(patches.variable_mask[batch_index], as_tuple=False).flatten()
        neighbors = {int(index): [] for index in valid_variables.tolist()}
        if valid_variables.numel() > 1:
            score = all_scores[batch_index].index_select(0, valid_variables).index_select(1, valid_variables)
            score = score.masked_fill(
                torch.eye(valid_variables.numel(), dtype=torch.bool, device=history.device), -torch.inf
            )
            requested = max(8, math.ceil(2.0 * math.log2(valid_variables.numel())))
            count = min(16, requested, valid_variables.numel() - 1)
            columns = list(score.detach().topk(count, dim=-1).indices)
            for row, destination in enumerate(valid_variables.tolist()):
                neighbors[destination] = [
                    (int(valid_variables[column]), score[row, column]) for column in columns[row].tolist()
                ]

        offset = batch_index * nodes
        for local in torch.nonzero(patches.node_mask[batch_index], as_tuple=False).flatten().tolist():
            scale = scale_meta[local]
            patch_index = patch_meta[local]
            variable = variable_meta[local]
            destination = offset + local
            for delta in (-2, -1, 1, 2):
                source_local = lookup.get((scale, patch_index + delta, variable))
                if source_local is not None and valid_nodes[source_local]:
                    result["temporal"].append(
                        (offset + source_local, destination, 0.35 if abs(delta) == 1 else 0.15)
                    )
            for other_scale in range(len(patches.widths)):
                if other_scale == scale:
                    continue
                for source_local in variable_nodes.get((other_scale, variable), []):
                    if (
                        start_meta[source_local] < end_meta[local]
                        and end_meta[source_local] > start_meta[local]
                        and valid_nodes[source_local]
                    ):
                        result["cross_scale"].append((offset + source_local, destination, 0.25))
            for source_variable, score in neighbors.get(variable, []):
                source_local = lookup.get((scale, patch_index, source_variable))
                if source_local is not None and valid_nodes[source_local]:
                    result["variable"].append((offset + source_local, destination, score))
    return result


def _canonical(edges):
    if edges.prior.numel() == 0:
        return edges.edge_index.new_empty((0, 2)), edges.prior
    key = edges.edge_index[0] * (int(edges.edge_index.max()) + 1) + edges.edge_index[1]
    order = torch.argsort(key)
    return edges.edge_index[:, order].transpose(0, 1), edges.prior[order]


class SparseEdgeBuilderVectorizationTests(unittest.TestCase):
    def _case(self, variables):
        torch.manual_seed(41 + variables)
        batch, length = 2, 9
        history = torch.randn(batch, length, variables, dtype=torch.double)
        observed = torch.rand(batch, length, variables) > 0.15
        variable_mask = torch.ones(batch, variables, dtype=torch.bool)
        variable_mask[1, -1] = False
        history[1, :, -1] = 0.0
        observed[1, :, -1] = False
        reliability = observed.to(torch.double)
        patchifier = AdaptivePatchifier(d_model=8, dropout=0.0).double().eval()
        patches = patchifier(
            history,
            observed,
            reliability,
            torch.zeros(batch, variables, dtype=torch.long),
            variable_mask,
        )
        builder = SparseEdgeBuilder(
            relation_dim=5,
            d_model=8,
        ).double()
        reference = _reference_edges(builder, patches, history, observed)
        actual = builder(patches, history, observed)

        for name in RELATIONS:
            with self.subTest(relation=name):
                expected_items = reference[name]
                expected_index = torch.tensor(
                    [[item[0], item[1]] for item in expected_items], dtype=torch.long
                )
                expected_prior = torch.stack(
                    [item[2] if torch.is_tensor(item[2]) else history.new_tensor(item[2]) for item in expected_items]
                ) if expected_items else history.new_empty(0)
                actual_index, actual_prior = _canonical(actual.relations[name])
                if expected_items:
                    key = expected_index[:, 0] * (int(expected_index.max()) + 1) + expected_index[:, 1]
                    order = torch.argsort(key)
                    expected_index = expected_index[order]
                    expected_prior = expected_prior[order]
                self.assertTrue(torch.equal(actual_index.cpu(), expected_index))
                torch.testing.assert_close(actual_prior, expected_prior, rtol=1e-12, atol=1e-12)
        return builder, patches, history, observed, reference, actual

    def test_adaptive_small_graph_matches_list_reference(self):
        self._case(variables=4)

    def test_adaptive_larger_graph_matches_list_reference(self):
        self._case(variables=21)

    def test_variable_prior_gradients_match_list_reference(self):
        builder, patches, history, observed, reference, actual = self._case(variables=5)
        reference_loss = torch.stack(
            [item[2] for item in reference["variable"]]
        ).sum()
        actual_loss = actual.relations["variable"].prior.sum()
        reference_grad = torch.autograd.grad(
            reference_loss,
            (
                builder.relation_mix_logits,
                builder.pool_score.weight,
                builder.dynamic_q.weight,
                builder.dynamic_k.weight,
                builder.static_embedding.weight,
            ),
            retain_graph=True,
        )
        actual_grad = torch.autograd.grad(
            actual_loss,
            (
                builder.relation_mix_logits,
                builder.pool_score.weight,
                builder.dynamic_q.weight,
                builder.dynamic_k.weight,
                builder.static_embedding.weight,
            ),
        )
        for actual_value, expected_value in zip(actual_grad, reference_grad):
            torch.testing.assert_close(actual_value, expected_value, rtol=1e-11, atol=1e-11)


class DynamicPatchScoringTests(unittest.TestCase):
    @staticmethod
    def _case(variables: int = 4, *, masked_last_variable: bool = False):
        torch.manual_seed(73 + variables)
        batch, length, d_model = 2, 9, 8
        history = torch.randn(batch, length, variables, dtype=torch.double)
        observed = torch.ones_like(history, dtype=torch.bool)
        variable_mask = torch.ones(batch, variables, dtype=torch.bool)
        if masked_last_variable:
            variable_mask[1, -1] = False
            observed[1, :, -1] = False
            history[1, :, -1] = 0.0
        patchifier = AdaptivePatchifier(d_model=d_model, dropout=0.0).double().eval()
        patches = patchifier(
            history,
            observed,
            observed.to(torch.double),
            torch.zeros(batch, variables, dtype=torch.long),
            variable_mask,
        )
        builder = SparseEdgeBuilder(relation_dim=4, d_model=d_model).double()
        return builder, patches, history, observed

    def test_identical_history_with_different_patch_tokens_changes_dynamic_scores(self):
        builder, patches, history, observed = self._case()
        with torch.no_grad():
            builder.pool_score.weight.zero_()
            builder.dynamic_q.weight.zero_()
            builder.dynamic_k.weight.zero_()
            builder.dynamic_q.weight[:, :4] = torch.eye(4, dtype=torch.double)
            builder.dynamic_k.weight[:, 4:] = torch.eye(4, dtype=torch.double)

        nodes = torch.zeros_like(patches.nodes)
        nodes[:, patches.variable_index == 0, 0] = 1.0
        nodes[:, patches.variable_index == 1, 4] = 1.0
        original = replace(patches, nodes=nodes)
        altered_nodes = nodes.clone()
        # Change direction rather than magnitude: cosine scoring is correctly
        # invariant to positive rescaling of the same patch representation.
        altered_nodes[:, patches.variable_index == 0, 1] = 1.0
        altered = replace(patches, nodes=altered_nodes)

        original_score = builder._variable_scores(original, history, observed)
        altered_score = builder._variable_scores(altered, history, observed)
        self.assertFalse(torch.allclose(original_score, altered_score))
        self.assertGreater(
            float((altered_score[:, 0, 1] - original_score[:, 0, 1]).detach().abs().min()),
            0.0,
        )

    def test_dynamic_query_key_and_pool_parameters_are_independent_and_receive_gradients(self):
        builder, patches, history, observed = self._case()
        self.assertIsNot(builder.dynamic_q, builder.dynamic_k)
        self.assertNotEqual(builder.dynamic_q.weight.data_ptr(), builder.dynamic_k.weight.data_ptr())
        loss = builder._variable_scores(patches, history, observed).square().sum()
        loss.backward()
        for parameter in (
            builder.pool_score.weight,
            builder.dynamic_q.weight,
            builder.dynamic_k.weight,
        ):
            self.assertIsNotNone(parameter.grad)
            self.assertGreater(float(parameter.grad.abs().sum()), 0.0)

    def test_attention_pool_supports_bfloat16_autocast_with_float_patch_tokens(self):
        """Regression: scatter accumulators must follow the score dtype under AMP."""

        builder, patches, history, observed = self._case()
        builder = builder.float()
        patches = replace(patches, nodes=patches.nodes.float())
        history = history.float()
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            scores = builder._variable_scores(patches, history, observed)
        self.assertEqual(scores.shape, (history.size(0), history.size(-1), history.size(-1)))
        self.assertTrue(torch.isfinite(scores).all().item())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for BF16 AMP regression")
    def test_attention_pool_supports_cuda_bfloat16_autocast(self):
        builder, patches, history, observed = self._case()
        builder = builder.float().cuda()
        tensor_fields = (
            "nodes",
            "node_mask",
            "variable_index",
            "scale_index",
            "patch_index",
            "start",
            "end",
            "center",
            "variable_mask",
        )
        cuda_fields = {name: getattr(patches, name).cuda() for name in tensor_fields}
        cuda_fields["nodes"] = cuda_fields["nodes"].float()
        patches = replace(patches, **cuda_fields)
        history = history.float().cuda()
        observed = observed.cuda()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            scores = builder._variable_scores(patches, history, observed)
        self.assertTrue(torch.isfinite(scores).all().item())

    def test_directed_dynamic_cosine_is_bounded_and_scale_invariant(self):
        builder, patches, _, _ = self._case()
        summary = builder._pool_patch_tokens(patches)
        original = builder._dynamic_scores(summary)
        self.assertLessEqual(float(original.detach().abs().max()), 1.0 + 1e-12)
        self.assertFalse(torch.allclose(original, original.transpose(-2, -1)))
        with torch.no_grad():
            builder.dynamic_q.weight.mul_(7.0)
            builder.dynamic_k.weight.mul_(0.25)
        scaled = builder._dynamic_scores(summary)
        torch.testing.assert_close(scaled, original, rtol=1e-11, atol=1e-11)

    def test_selected_priors_keep_all_scoring_paths_trainable(self):
        builder, patches, history, observed = self._case(variables=21)
        prior = builder(patches, history, observed).relations["variable"].prior
        self.assertGreater(prior.numel(), 0)
        loss = prior.square().mean()
        parameters = (
            builder.relation_mix_logits,
            builder.pool_score.weight,
            builder.dynamic_q.weight,
            builder.dynamic_k.weight,
            builder.static_embedding.weight,
        )
        gradients = torch.autograd.grad(loss, parameters)
        for gradient in gradients:
            self.assertGreater(float(gradient.abs().sum()), 0.0)

    def test_masked_patch_nodes_cannot_change_variable_scores(self):
        builder, patches, history, observed = self._case(masked_last_variable=True)
        self.assertTrue((~patches.node_mask).any())
        altered_nodes = patches.nodes.clone()
        altered_nodes[~patches.node_mask] = torch.randn_like(altered_nodes[~patches.node_mask]) * 1e6
        altered = replace(patches, nodes=altered_nodes)
        expected = builder._variable_scores(patches, history, observed)
        actual = builder._variable_scores(altered, history, observed)
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    def test_difference_correlation_preserves_sign(self):
        increasing = torch.arange(8, dtype=torch.double).square()
        values = torch.stack((increasing, -increasing), dim=-1).unsqueeze(0)
        correlation = _difference_correlation(values, torch.ones_like(values, dtype=torch.bool))
        self.assertLess(float(correlation[0, 0, 1]), -0.99)


class HistoryConditionedPatchEmbeddingTests(unittest.TestCase):
    @staticmethod
    def _patches(variant: str, history: torch.Tensor):
        batch, length, variables = history.shape
        patchifier = AdaptivePatchifier(
            d_model=8,
            dropout=0.0,
            history_len=length,
            embedding_variant=variant,
        ).eval()
        observed = torch.ones_like(history, dtype=torch.bool)
        return patchifier(
            history,
            observed,
            observed.float(),
            torch.zeros(batch, variables, dtype=torch.long),
            torch.ones(batch, variables, dtype=torch.bool),
        )

    def test_series_context_makes_early_patch_nodes_depend_on_the_full_history(self):
        torch.manual_seed(71)
        original = torch.randn(1, 36, 2)
        altered = original.clone()
        altered[:, -1, :] += 5.0

        torch.manual_seed(73)
        local_original = self._patches("patch", original)
        torch.manual_seed(73)
        local_altered = self._patches("patch", altered)
        early = local_original.patch_index.eq(0)
        torch.testing.assert_close(local_original.nodes[:, early], local_altered.nodes[:, early])

        torch.manual_seed(79)
        contextual_original = self._patches("series_context", original)
        torch.manual_seed(79)
        contextual_altered = self._patches("series_context", altered)
        self.assertFalse(torch.allclose(contextual_original.nodes[:, early], contextual_altered.nodes[:, early]))

    def test_difference_variant_preserves_the_approved_node_budget(self):
        history = torch.randn(2, 36, 7)
        contextual = self._patches("series_context_diff", history)
        local = self._patches("patch", history)
        self.assertEqual(contextual.nodes.shape, local.nodes.shape)
        self.assertEqual(contextual.real_node_count, local.real_node_count)

    def test_decomposition_variant_keeps_the_local_graph_shape(self):
        history = torch.randn(2, 36, 7)
        decomposed = self._patches("series_context_decomp", history)
        local = self._patches("patch", history)
        self.assertEqual(decomposed.nodes.shape, local.nodes.shape)
        self.assertEqual(decomposed.widths, local.widths)

    def test_global_node_adds_exactly_one_node_per_variable_and_stays_within_ecl_budget(self):
        for variables in (7, 321):
            with self.subTest(variables=variables):
                history = torch.randn(1, 36, variables)
                local = self._patches("patch", history)
                global_graph = self._patches("global_node", history)
                self.assertEqual(global_graph.nodes.size(1), local.nodes.size(1) + variables)
                self.assertEqual(global_graph.real_node_count, local.real_node_count + variables)
                self.assertLessEqual(global_graph.real_node_count, 6000)
                self.assertEqual(global_graph.widths[-1], 36)
                self.assertEqual(global_graph.patches_per_scale[-1], 1)

    def test_global_raw_residual_changes_only_the_global_nodes_at_initialization(self):
        torch.manual_seed(83)
        history = torch.randn(1, 36, 7)
        torch.manual_seed(89)
        shape_only = self._patches("global_node", history)
        torch.manual_seed(89)
        raw_residual = self._patches("global_node_raw", history)
        global_scale = len(raw_residual.widths) - 1
        local = raw_residual.scale_index.ne(global_scale)
        torch.testing.assert_close(raw_residual.nodes[:, local], shape_only.nodes[:, local])
        self.assertFalse(torch.allclose(raw_residual.nodes[:, ~local], shape_only.nodes[:, ~local]))

    def test_history_conditioning_rejects_a_mismatched_window(self):
        patchifier = AdaptivePatchifier(
            d_model=8,
            dropout=0.0,
            history_len=36,
            embedding_variant="series_context",
        )
        values = torch.randn(1, 35, 2)
        observed = torch.ones_like(values, dtype=torch.bool)
        with self.assertRaisesRegex(ValueError, "expected history length 36"):
            patchifier(
                values,
                observed,
                observed.float(),
                torch.zeros(1, 2, dtype=torch.long),
                torch.ones(1, 2, dtype=torch.bool),
            )


class AdaptiveVariableNeighborTests(unittest.TestCase):
    @staticmethod
    def _graph(
        variables: int,
        valid_counts: tuple[int, ...] | None = None,
        *,
        max_neighbors: int = 16,
    ):
        torch.manual_seed(101 + variables)
        valid_counts = valid_counts or (variables,)
        batch, length, d_model = len(valid_counts), 9, 8
        history = torch.randn(batch, length, variables)
        variable_ids = torch.arange(variables).unsqueeze(0)
        variable_mask = variable_ids < torch.tensor(valid_counts).unsqueeze(1)
        observed = variable_mask.unsqueeze(1).expand(-1, length, -1).clone()
        history = history * observed
        patchifier = AdaptivePatchifier(d_model=d_model, dropout=0.0).eval()
        patches = patchifier(
            history,
            observed,
            observed.float(),
            torch.zeros(batch, variables, dtype=torch.long),
            variable_mask,
        )
        builder = SparseEdgeBuilder(d_model=d_model, max_neighbors=max_neighbors).eval()
        edges = builder(patches, history, observed).relations["variable"].edge_index
        return patches, edges

    @staticmethod
    def _incoming_degree(patches, edge_index: torch.Tensor) -> torch.Tensor:
        batch, nodes = patches.node_mask.shape
        return torch.bincount(edge_index[1], minlength=batch * nodes).reshape(batch, nodes)

    def test_approved_neighbor_schedule_and_no_self_loops(self):
        for variables, expected_neighbors in ((7, 6), (21, 9), (58, 12), (321, 16)):
            with self.subTest(variables=variables):
                patches, edge_index = self._graph(variables)
                degree = self._incoming_degree(patches, edge_index)
                self.assertTrue(
                    torch.equal(
                        degree[patches.node_mask],
                        torch.full_like(degree[patches.node_mask], expected_neighbors),
                    )
                )
                self.assertLessEqual(int(degree.max()), 16)
                self.assertFalse(torch.eq(edge_index[0], edge_index[1]).any().item())

    def test_each_sample_uses_its_own_valid_feature_count(self):
        patches, edge_index = self._graph(58, valid_counts=(58, 21, 1))
        degree = self._incoming_degree(patches, edge_index)
        for sample, expected_neighbors in enumerate((12, 9, 0)):
            with self.subTest(sample=sample):
                valid_degree = degree[sample, patches.node_mask[sample]]
                self.assertTrue(
                    torch.equal(valid_degree, torch.full_like(valid_degree, expected_neighbors))
                )
                self.assertTrue(torch.eq(degree[sample, ~patches.node_mask[sample]], 0).all().item())
        flat_mask = patches.node_mask.reshape(-1)
        self.assertTrue(flat_mask.index_select(0, edge_index[0]).all().item())
        self.assertTrue(flat_mask.index_select(0, edge_index[1]).all().item())
        self.assertFalse(torch.eq(edge_index[0], edge_index[1]).any().item())

    def test_configured_neighbor_cap_is_respected_below_the_formal_default(self):
        patches, edge_index = self._graph(58, max_neighbors=8)
        degree = self._incoming_degree(patches, edge_index)
        self.assertTrue(
            torch.equal(
                degree[patches.node_mask],
                torch.full_like(degree[patches.node_mask], 8),
            )
        )

    def test_neighbor_cap_rejects_values_outside_the_contract(self):
        with self.assertRaisesRegex(ValueError, "8..16"):
            SparseEdgeBuilder(max_neighbors=7)

    def test_general_and_battery_models_share_the_same_edge_builder(self):
        general = GraphReportTSv2(
            GraphReportTSv2Config(domain="general", input_len=36, pred_len=24, use_text=False)
        )
        battery = BatteryGraphReportTSv2(
            GraphReportTSv2Config(domain="battery", input_len=32, pred_len=20, use_text=False)
        )
        self.assertIs(type(general.edge_builder), SparseEdgeBuilder)
        self.assertIs(type(battery.shared.edge_builder), SparseEdgeBuilder)


if __name__ == "__main__":
    unittest.main()
