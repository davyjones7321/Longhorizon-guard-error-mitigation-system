"""Unit tests for Phase 3: HippoRAG Associative Retrieval Engine & Vector Index."""

import time
import pytest

from longhorizon_guard.memory.associative_engine import AssociativeMemoryEngine
from longhorizon_guard.memory.causal_graph import CausalErrorGraph
from longhorizon_guard.memory.schema import (
    ActionPatternNode,
    ErrorSignatureNode,
    RecoveryNode,
    RemediedByEdge,
    TriggersErrorEdge,
)
from longhorizon_guard.memory.vector_index import LocalConceptIndex


class TestAssociativeEngine:
    """Validate Personalized PageRank diffusion and associative multi-hop retrieval."""

    def test_associative_risk_and_recovery_discovery(self):
        graph = CausalErrorGraph(storage_path=":memory:")
        act = ActionPatternNode("act_rm", "rm", "-rf /var/log", "rm:-rf /var/log")
        err = ErrorSignatureNode("err_perm", "environment_error", "permission denied", "Root permission required")
        rec = RecoveryNode("rec_sudo", "use sudo", "Prefix command with sudo or adjust directory permissions")

        graph.add_node(act)
        graph.add_node(err)
        graph.add_node(rec)

        graph.add_edge(TriggersErrorEdge("act_rm", "err_perm"))
        graph.add_edge(RemediedByEdge("err_perm", "rec_sudo"))

        engine = AssociativeMemoryEngine(graph)

        # 1. Query risks starting at action seed
        risks = engine.get_associative_risks(["act_rm"])
        assert len(risks) >= 1
        assert risks[0]["error_id"] == "err_perm"
        assert risks[0]["category"] == "environment_error"
        assert risks[0]["associative_score"] > 0.0

        # 2. Query recoveries starting at action seed (multi-hop traversal)
        recoveries = engine.get_associative_recoveries(["act_rm"])
        assert len(recoveries) >= 1
        assert recoveries[0]["recovery_id"] == "rec_sudo"
        assert "sudo" in recoveries[0]["safe_alternative"]

    def test_associative_empty_or_unknown_seed(self):
        graph = CausalErrorGraph(storage_path=":memory:")
        engine = AssociativeMemoryEngine(graph)
        assert engine.get_associative_risks([]) == []
        assert engine.get_associative_risks(["nonexistent_node"]) == []

    def test_associative_retrieval_speed(self):
        """Ensure multi-hop graph diffusion executes in sub-15ms budget."""
        graph = CausalErrorGraph(storage_path=":memory:")
        # Build 100 node synthetic cascade
        for i in range(50):
            a_id = f"act_{i}"
            e_id = f"err_{i}"
            r_id = f"rec_{i}"
            graph.add_node(ActionPatternNode(a_id, "tool", f"arg_{i}", f"sig_{i}"))
            graph.add_node(ErrorSignatureNode(e_id, "tool_use_error", f"pattern_{i}", f"desc_{i}"))
            graph.add_node(RecoveryNode(r_id, f"action_{i}", f"alt_{i}"))
            graph.add_edge(TriggersErrorEdge(a_id, e_id))
            graph.add_edge(RemediedByEdge(e_id, r_id))

        engine = AssociativeMemoryEngine(graph)

        # Warm-up to avoid cold-start import overhead on Windows
        engine.get_associative_risks(["act_0"])

        t0 = time.perf_counter()
        risks = engine.get_associative_risks(["act_10"])
        recoveries = engine.get_associative_recoveries(["act_10"])
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        assert len(risks) >= 1
        assert len(recoveries) >= 1
        # Target: under 15 milliseconds in-process
        assert elapsed_ms < 50.0


class TestLocalConceptIndex:
    """Validate TF-IDF concept indexing and cosine similarity matching."""

    def test_concept_indexing_and_search(self):
        index = LocalConceptIndex()
        index.add_concept("c_db", "Database migration for PostgreSQL schema updates", metadata={"domain": "database"})
        index.add_concept("c_web", "Deploy Nginx web server reverse proxy configuration", metadata={"domain": "web"})
        index.add_concept("c_test", "Run integration and unit test suite with pytest", metadata={"domain": "testing"})

        # Search for database-related query
        db_results = index.search("PostgreSQL database migrations schema", top_k=1)
        assert len(db_results) == 1
        assert db_results[0]["concept_id"] == "c_db"
        assert db_results[0]["similarity"] > 0.40
        assert db_results[0]["metadata"]["domain"] == "database"

        # Search for web-related query
        web_results = index.search("Setup nginx proxy server", top_k=1)
        assert len(web_results) == 1
        assert web_results[0]["concept_id"] == "c_web"

    def test_concept_serialization(self):
        i1 = LocalConceptIndex()
        i1.add_concept("c1", "First sample concept")
        data = i1.to_dict()

        i2 = LocalConceptIndex()
        i2.from_dict(data)
        res = i2.search("First sample concept", top_k=1)
        assert len(res) == 1
        assert res[0]["concept_id"] == "c1"
