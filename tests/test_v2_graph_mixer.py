from __future__ import annotations

import math
import unittest

import torch

from bstalignment.v2.edges import RelationEdges
from bstalignment.v2.graph_mixer import RelationAttention, segment_softmax


class RelationAttentionProjectionTests(unittest.TestCase):
    def test_node_first_projection_matches_edge_first_reference(self):
        torch.manual_seed(17)
        attention = RelationAttention(d_model=8, heads=2).double()
        nodes = torch.randn(6, 8, dtype=torch.double)
        source = torch.tensor([0, 1, 2, 4, 5, 1], dtype=torch.long)
        destination = torch.tensor([1, 2, 3, 3, 3, 5], dtype=torch.long)
        edges = RelationEdges(
            edge_index=torch.stack((source, destination)),
            prior=torch.randn(source.numel(), dtype=torch.double),
        )

        actual = attention(nodes, edges)

        query = attention.q(nodes.index_select(0, destination)).view(-1, 2, 4)
        key = attention.k(nodes.index_select(0, source)).view(-1, 2, 4)
        value = attention.v(nodes.index_select(0, source)).view(-1, 2, 4)
        score = (query * key).sum(-1) / math.sqrt(4)
        weight = segment_softmax(score + edges.prior.unsqueeze(-1), destination, nodes.size(0))
        message = (weight.unsqueeze(-1) * value).reshape(-1, 8)
        projected = attention.out(message)
        expected = projected.new_zeros(nodes.shape)
        expected.index_add_(0, destination, projected)

        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
