"""AnchoredGTR model identity and compatibility helpers.

The shared implementation remains in :mod:`bstalignment.v2`; this module is
the stable paper-facing boundary for the general model only.
"""

from __future__ import annotations

from bstalignment.v2.contracts import GraphReportTSv2Config
from bstalignment.v2.model import GraphReportTSv2


ANCHORED_GTR_MODEL_NAME = "AnchoredGTR"
LEGACY_GENERAL_MODEL_PREFIX = "GraphReportTS-v2-DRF"


def canonical_general_model_name(name: str) -> str:
    """Map the former general-model prefix while preserving variant suffixes."""

    value = str(name)
    if value.startswith(LEGACY_GENERAL_MODEL_PREFIX):
        return ANCHORED_GTR_MODEL_NAME + value[len(LEGACY_GENERAL_MODEL_PREFIX) :]
    return value


class AnchoredGTR(GraphReportTSv2):
    """Paper-facing general graph-text residual model.

    Ridge fitting/freezing is a train-split preparation step owned by the
    trainer, so the class validates the structural half of the contract: a
    general-domain model with the approved decomposition-aware graph encoder.
    """

    def __init__(self, config: GraphReportTSv2Config) -> None:
        if config.domain != "general":
            raise ValueError("AnchoredGTR requires domain='general'")
        if config.graph_embedding_variant != "series_context_decomp":
            raise ValueError("AnchoredGTR requires graph_embedding_variant='series_context_decomp'")
        super().__init__(config)


__all__ = [
    "ANCHORED_GTR_MODEL_NAME",
    "LEGACY_GENERAL_MODEL_PREFIX",
    "AnchoredGTR",
    "canonical_general_model_name",
]
