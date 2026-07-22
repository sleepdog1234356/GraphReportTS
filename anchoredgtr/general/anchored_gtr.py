"""Stable paper-facing AnchoredGTR model identity."""

from __future__ import annotations

from anchoredgtr.core.contracts import GTRConfig
from anchoredgtr.core.model import GraphTextResidualCore


ANCHORED_GTR_MODEL_NAME = "AnchoredGTR"


class AnchoredGTR(GraphTextResidualCore):
    """Paper-facing general graph-text residual model.

    Ridge fitting/freezing is a train-split preparation step owned by the
    trainer, so the class validates the structural half of the contract: a
    general-domain model with the approved decomposition-aware graph encoder.
    """

    def __init__(self, config: GTRConfig) -> None:
        if config.domain != "general":
            raise ValueError("AnchoredGTR requires domain='general'")
        if config.graph_embedding_variant != "series_context_decomp":
            raise ValueError("AnchoredGTR requires graph_embedding_variant='series_context_decomp'")
        super().__init__(config)


__all__ = [
    "ANCHORED_GTR_MODEL_NAME",
    "AnchoredGTR",
]
