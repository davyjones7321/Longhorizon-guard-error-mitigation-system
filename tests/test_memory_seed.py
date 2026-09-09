"""Unit tests for Phase 4: Memory Seed Ingestion."""

import os
import tempfile
import pytest

from longhorizon_guard.memory.associative_engine import AssociativeMemoryEngine
from longhorizon_guard.memory.causal_graph import CausalErrorGraph
from longhorizon_guard.memory.seed_loader import (
    bootstrap_memory_graph,
    seed_canonical_anti_patterns,
    seed_error_cascades,
    seed_from_pattern_library,
)


class TestMemorySeedLoader:
    """Validate seeding of patterns, anti-patterns, and cascades into CausalErrorGraph."""

    def test_seed_from_real_pattern_library(self):
        graph = CausalErrorGraph(storage_path=":memory:")
        count = seed_from_pattern_library(graph)
        assert count > 0

        summary = graph.summary()
        assert summary["total_nodes"] > 10
        assert summary["total_edges"] > 5
        assert "error_signature" in summary["nodes_by_type"]

    def test_seed_anti_patterns_and_prerequisites(self):
        graph = CausalErrorGraph(storage_path=":memory:")
        count = seed_canonical_anti_patterns(graph)
        assert count > 0

        # Verify prerequisite checking
        valid, missing = graph.check_subgoal_prerequisites([], "deploy_service")
        assert valid is False
        assert len(missing) >= 1

        valid, missing = graph.check_subgoal_prerequisites(["build_project", "run_tests"], "deploy_service")
        assert valid is True
        assert len(missing) == 0

    def test_seed_error_cascades(self):
        graph = CausalErrorGraph(storage_path=":memory:")
        count = seed_error_cascades(graph)
        assert count > 0
        summary = graph.summary()
        assert summary["edges_by_type"].get("propagates_to", 0) >= 3

    def test_bootstrap_memory_graph_and_associative_query(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
            temp_path = tf.name

        try:
            graph = bootstrap_memory_graph(storage_path=temp_path)
            assert graph.graph.number_of_nodes() > 20

            # Run associative retrieval on a seeded anti-pattern
            engine = AssociativeMemoryEngine(graph)
            paths = graph.find_recovery_paths("bash")
            assert len(paths) >= 1
            assert any("trash" in p["recovery_suggestion"] or "rebase" in p["recovery_suggestion"] for p in paths)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
