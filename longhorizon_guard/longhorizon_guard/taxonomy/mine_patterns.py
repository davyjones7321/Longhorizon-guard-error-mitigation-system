#!/usr/bin/env python3
"""
Phase 4 Failure Pattern Mining Pipeline for longhorizon_guard.

Mines recurring failure patterns from LLM-as-judge tagged trajectories
(planning_error and reflection_error) using TF-IDF text vectorization and
deterministic Leader-Follower clustering.

Usage:
    python -m longhorizon_guard.taxonomy.mine_patterns \
        --judged-data findings/agenterrorbench_judged.json \
        --converted-data findings/agenterrorbench_converted.json \
        --output findings/pattern_library.json \
        --min-cluster-size 3 \
        --similarity-threshold 0.35 \
        --dry-run
"""

import argparse
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("longhorizon_guard"))

from longhorizon_guard.pattern_library.schema import PatternEntry, TrajectorySnippet
from longhorizon_guard.pattern_library.store import save_patterns
from longhorizon_guard.storage.reader import load_dataset


CORRECTIVE_KEYWORDS: Set[str] = {
    "should", "instead", "must", "recommend", "require", "correct", "fix",
    "needs", "needed", "verify", "check", "ensure", "update", "validate"
}

STOPWORDS: Set[str] = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "to", "of", "and", "in",
    "that", "with", "for", "on", "at", "by", "from", "it", "this", "these",
    "those", "not", "or", "but", "as", "if", "they", "their", "them", "agent",
    "step", "steps", "which", "than", "because", "so", "can",
    "could", "would", "may", "might", "very", "also"
} - CORRECTIVE_KEYWORDS


def preprocess_text(text: str) -> List[str]:
    """Tokenize, lowercase, strip punctuation and filter stopwords."""
    if not text:
        return []
    words = re.findall(r"[a-z0-9_]+", text.lower())
    return [w for w in words if len(w) > 1 and w not in STOPWORDS]


def compute_tfidf_vectors(texts: List[str]) -> Tuple[List[Dict[str, float]], List[str]]:
    """Compute TF-IDF vectors for a list of document texts.

    Returns:
        (vectors, vocabulary) where each vector is a dict mapping word -> tfidf_score.
    """
    tokenized_docs = [preprocess_text(t) for t in texts]
    num_docs = len(texts)

    # Document frequency
    df: Dict[str, int] = defaultdict(int)
    for doc in tokenized_docs:
        unique_words = set(doc)
        for w in unique_words:
            df[w] += 1

    vocab = sorted(list(df.keys()))
    if not vocab or num_docs == 0:
        return [{} for _ in texts], []

    # IDF calculation with smoothing
    idf: Dict[str, float] = {}
    for w, count in df.items():
        idf[w] = math.log((num_docs + 1.0) / (count + 1.0)) + 1.0

    vectors: List[Dict[str, float]] = []
    for doc in tokenized_docs:
        if not doc:
            vectors.append({})
            continue
        tf_counts = Counter(doc)
        doc_len = len(doc)
        vec: Dict[str, float] = {}
        norm_sq = 0.0
        for w, count in tf_counts.items():
            tf = count / doc_len
            val = tf * idf[w]
            vec[w] = val
            norm_sq += val * val
        norm = math.sqrt(norm_sq)
        if norm > 0:
            for w in vec:
                vec[w] /= norm
        vectors.append(vec)

    return vectors, vocab


def cosine_similarity(vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
    """Compute cosine similarity between two normalized TF-IDF term dicts."""
    if not vec1 or not vec2:
        return 0.0
    common_keys = set(vec1.keys()).intersection(vec2.keys())
    if not common_keys:
        return 0.0
    return sum(vec1[k] * vec2[k] for k in common_keys)


def compute_centroid(cluster_vectors: List[Dict[str, float]]) -> Dict[str, float]:
    """Compute normalized centroid vector for a list of sparse TF-IDF vectors."""
    if not cluster_vectors:
        return {}
    sum_vec: Dict[str, float] = defaultdict(float)
    for vec in cluster_vectors:
        for k, v in vec.items():
            sum_vec[k] += v
    num_vecs = len(cluster_vectors)
    norm_sq = 0.0
    centroid: Dict[str, float] = {}
    for k, v in sum_vec.items():
        avg_val = v / num_vecs
        centroid[k] = avg_val
        norm_sq += avg_val * avg_val
    norm = math.sqrt(norm_sq)
    if norm > 0:
        for k in centroid:
            centroid[k] /= norm
    return centroid


def leader_follower_clustering(
    records: List[Dict[str, Any]],
    vectors: List[Dict[str, float]],
    threshold: float
) -> List[List[int]]:
    """Perform deterministic Leader-Follower clustering on records ordered by run_id.

    Returns:
        List of clusters, where each cluster is a list of integer indices into records/vectors.
    """
    if not records:
        return []

    clusters: List[List[int]] = []
    centroids: List[Dict[str, float]] = []

    for idx, (rec, vec) in enumerate(zip(records, vectors)):
        if not clusters:
            clusters.append([idx])
            centroids.append(dict(vec))
            continue

        best_cluster_idx = -1
        best_sim = -1.0
        for c_idx, centroid in enumerate(centroids):
            sim = cosine_similarity(vec, centroid)
            if sim > best_sim:
                best_sim = sim
                best_cluster_idx = c_idx

        if best_sim >= threshold and best_cluster_idx >= 0:
            clusters[best_cluster_idx].append(idx)
            # Recompute centroid as mean of all member vectors
            c_member_vecs = [vectors[i] for i in clusters[best_cluster_idx]]
            centroids[best_cluster_idx] = compute_centroid(c_member_vecs)
        else:
            clusters.append([idx])
            centroids.append(dict(vec))

    return clusters


def find_medoid_and_pairwise(
    cluster_indices: List[int],
    vectors: List[Dict[str, float]]
) -> Tuple[int, Dict[Tuple[int, int], float], Dict[int, float]]:
    """Compute pairwise similarities and select medoid index for a cluster.

    Returns:
        (medoid_index, pairwise_sim_dict, avg_sim_dict)
    """
    n = len(cluster_indices)
    pairwise: Dict[Tuple[int, int], float] = {}
    avg_sims: Dict[int, float] = {}

    if n == 1:
        idx = cluster_indices[0]
        pairwise[(idx, idx)] = 1.0
        avg_sims[idx] = 1.0
        return idx, pairwise, avg_sims

    for i in range(n):
        idx1 = cluster_indices[i]
        sum_sim = 0.0
        for j in range(n):
            idx2 = cluster_indices[j]
            if i == j:
                sim = 1.0
            else:
                sim = cosine_similarity(vectors[idx1], vectors[idx2])
            pairwise[(idx1, idx2)] = sim
            if i != j:
                sum_sim += sim
        avg_sims[idx1] = sum_sim / (n - 1)

    medoid_idx = max(cluster_indices, key=lambda idx: avg_sims[idx])
    return medoid_idx, pairwise, avg_sims


def extract_example_snippet(run_record: Dict[str, Any], root_cause_step_index: Optional[int]) -> List[TrajectorySnippet]:
    """Extract +/- 2 steps around root_cause_step_index as TrajectorySnippet objects."""
    traj = run_record.get("trajectory", {}) or {}
    steps = traj.get("steps", [])
    if not steps:
        return []

    if root_cause_step_index is None:
        target_step = 0
    else:
        target_step = root_cause_step_index

    # Find position of step with matching step_index
    step_indices = [s.get("step_index") for s in steps]
    if target_step in step_indices:
        pos = step_indices.index(target_step)
    else:
        pos = 0

    start_pos = max(0, pos - 2)
    end_pos = min(len(steps), pos + 3)

    snippet_steps = steps[start_pos:end_pos]
    snippets: List[TrajectorySnippet] = []
    for s in snippet_steps:
        raw_args = s.get("action_args", {})
        clean_args = raw_args if isinstance(raw_args, dict) else {"args": str(raw_args)}
        snippets.append(
            TrajectorySnippet(
                step_index=s.get("step_index", 0),
                reasoning=(s.get("reasoning") or "")[:150],
                action_name=(s.get("action_name") or "none")[:50],
                action_args=clean_args,
                tool_response=(s.get("tool_response") or "")[:150] if s.get("tool_response") else None,
            )
        )
    return snippets


SPLIT_KEYWORD_RE = re.compile(
    r"\b(instead of|instead|should|must|recommend|require|correct|fix|needs to|needed|verify|check|ensure|update|validate)\b",
    re.IGNORECASE
)


def split_sentence_on_corrective_keyword(sentence: str) -> Tuple[str, str, bool]:
    """Split a sentence on a mid-sentence corrective keyword.

    Returns:
        (clause_before, clause_after, has_split)
    """
    sentence_clean = sentence.strip()
    match = SPLIT_KEYWORD_RE.search(sentence_clean)
    if not match:
        return sentence_clean, "", False

    start, end = match.span()
    before = sentence_clean[:start].strip(" ,;:-")
    after = sentence_clean[start:].strip(" ,;:-")

    # If keyword is near start (before clause is < 5 chars), treat full sentence as after
    if len(before) < 5:
        return "", sentence_clean, True

    return before, after, True


def extract_corrective_text_and_trigger(
    cluster_recs: List[Dict[str, Any]],
    medoid_rec: Dict[str, Any]
) -> Tuple[str, str, bool]:
    """Extract trigger_description and safe_alternative for a cluster.

    Rules:
    1. Prefer extracting safe_alternative from a non-medoid cluster member if available.
    2. If a corrective keyword is mid-sentence, split and use the clause AFTER it for safe_alternative,
       and clause BEFORE it (from medoid) for trigger_description.
    3. If corrective keyword is in a separate sentence, use that separate sentence for safe_alternative.
    4. Fall back to splitting medoid justification if no other member has corrective language.

    Returns:
        (trigger_description, safe_alternative_text, has_corrective_language)
    """
    medoid_id = medoid_rec["run_id"]
    medoid_just = medoid_rec["justification"] or ""

    # Clean lead-ins from medoid justification
    medoid_clean = re.sub(
        r"^(Step \d+:|Root cause:|The agent|Agent|Because)\s*", "", medoid_just, flags=re.IGNORECASE
    ).strip()

    # Find corrective candidates in non-medoid records first, then medoid
    non_medoid_recs = [r for r in cluster_recs if r["run_id"] != medoid_id]
    all_ordered_recs = non_medoid_recs + [medoid_rec]

    chosen_safe_alt = ""
    has_corrective = False

    for rec in all_ordered_recs:
        just = rec["justification"] or ""
        if not just:
            continue

        sentences = [s.strip() for s in re.split(r"[.!?\n]+", just) if s.strip()]
        for s in sentences:
            words = set(preprocess_text(s))
            if words.intersection(CORRECTIVE_KEYWORDS):
                before, after, has_split = split_sentence_on_corrective_keyword(s)
                if has_split:
                    chosen_safe_alt = after if after else s
                else:
                    chosen_safe_alt = s
                has_corrective = True
                break
        if has_corrective:
            break

    # Determine trigger description from medoid sentence
    medoid_sentences = [s.strip() for s in re.split(r"[.!?\n]+", medoid_clean) if s.strip()]
    if medoid_sentences:
        first_s = medoid_sentences[0]
        words = set(preprocess_text(first_s))
        if words.intersection(CORRECTIVE_KEYWORDS):
            before, after, has_split = split_sentence_on_corrective_keyword(first_s)
            if has_split and before:
                trigger_desc = before
            else:
                trigger_desc = first_s
        else:
            trigger_desc = first_s
    else:
        trigger_desc = medoid_clean

    trigger_desc = re.sub(
        r"^(Step \d+:|Root cause:|The agent|Agent|Because)\s*", "", trigger_desc, flags=re.IGNORECASE
    ).strip()
    trigger_desc = trigger_desc.rstrip(".,;:")
    chosen_safe_alt = chosen_safe_alt.strip().rstrip(".,;:")

    if not trigger_desc:
        trigger_desc = "Agent executed invalid plan or failed to update state based on environment feedback."

    return trigger_desc, chosen_safe_alt, has_corrective


def mine_patterns_pipeline(
    judged_data_path: str,
    converted_data_path: str,
    min_cluster_size: int = 3,
    similarity_threshold: float = 0.35,
    dry_run: bool = True,
    output_path: str = "findings/pattern_library.json"
) -> List[PatternEntry]:
    """Mine failure patterns from judged trajectory logs."""
    print(f"Loading converted dataset from {converted_data_path}...")
    converted_runs = load_dataset(converted_data_path, verbose=False)
    run_map = {r["metadata"]["run_id"]: r for r in converted_runs}

    print(f"Loading judged dataset from {judged_data_path}...")
    if not os.path.exists(judged_data_path):
        raise FileNotFoundError(f"Judged output file not found: {judged_data_path}")

    with open(judged_data_path, "r", encoding="utf-8") as f:
        judged_raw = json.load(f)

    judgments = judged_raw.get("judgments", {})
    print(f"Total judged records in output file: {len(judgments)}")

    # Step 1 & 2: Join and filter to ALL categories where pred == gt
    candidate_records: List[Dict[str, Any]] = []
    for run_id, jdata in judgments.items():
        pred_cat = jdata.get("root_cause_error_type")
        if not pred_cat:
            continue
        
        c_run = run_map.get(run_id)
        if not c_run:
            continue

        gt_cat = c_run["metadata"].get("root_cause_error_type")
        if pred_cat != gt_cat:
            continue  # Only mine from correct predictions

        justification = jdata.get("justification") or jdata.get("root_cause_justification") or ""
        step_idx = jdata.get("root_cause_step_index")

        candidate_records.append({
            "run_id": run_id,
            "pred_cat": pred_cat,
            "gt_cat": gt_cat,
            "justification": justification,
            "step_idx": step_idx,
            "converted_run": c_run,
        })

    print(f"Filtered candidate records (pred == gt, all categories): {len(candidate_records)}")

    # Step 4: Sort deterministically by run_id
    candidate_records.sort(key=lambda r: r["run_id"])

    # Group candidate records by category
    by_category: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for rec in candidate_records:
        by_category[rec["pred_cat"]].append(rec)

    mined_patterns: List[PatternEntry] = []

    for cat in sorted(by_category.keys()):
        recs = by_category.get(cat, [])
        if not recs:
            print(f"\nNo candidate records for category '{cat}'")
            continue

        print(f"\n--- Mining category '{cat}' ({len(recs)} records) ---")
        justifications = [r["justification"] for r in recs]

        # Step 5: TF-IDF vectorization
        vectors, vocab = compute_tfidf_vectors(justifications)

        # Step 6: Leader-follower clustering
        clusters = leader_follower_clustering(recs, vectors, threshold=similarity_threshold)
        print(f"Discovered {len(clusters)} raw clusters in '{cat}'")

        category_pattern_counter = 1

        for c_idx, cluster_indices in enumerate(clusters, 1):
            cluster_recs = [recs[i] for i in cluster_indices]
            cluster_size = len(cluster_recs)

            if cluster_size < min_cluster_size:
                if dry_run:
                    print(f"  [Cluster {c_idx}] Size={cluster_size} < min_size={min_cluster_size} (Skipped from pattern library)")
                continue

            # Step 7: Compute medoid and pairwise similarity matrix
            medoid_idx, pairwise, avg_sims = find_medoid_and_pairwise(cluster_indices, vectors)
            medoid_rec = recs[medoid_idx]
            medoid_run_id = medoid_rec["run_id"]

            trigger_desc, safe_alt, has_corrective = extract_corrective_text_and_trigger(cluster_recs, medoid_rec)
            example_snippets = extract_example_snippet(medoid_rec["converted_run"], medoid_rec["step_idx"])
            source_rids = sorted([r["run_id"] for r in cluster_recs])

            pattern_id = f"pattern_{cat}_{category_pattern_counter:03d}"
            category_pattern_counter += 1

            pattern_entry = PatternEntry(
                pattern_id=pattern_id,
                category=cat,
                trigger_description=trigger_desc,
                example_snippet=example_snippets,
                safe_alternative=safe_alt,
                source_run_ids=source_rids,
                confidence=1.0,
            )
            mined_patterns.append(pattern_entry)

            # Dry-run printing
            if dry_run:
                print(f"\n  === PROPOSED PATTERN: {pattern_id} (Cluster {c_idx}, Size={cluster_size}) ===")
                print(f"  Category            : {cat}")
                print(f"  Medoid Run ID       : {medoid_run_id}")
                print(f"  Member Run IDs ({cluster_size}): {source_rids}")
                print("  Pairwise Cosine Similarities:")
                for i in cluster_indices:
                    row_str = "    "
                    for j in cluster_indices:
                        sim_val = pairwise.get((i, j), 0.0)
                        row_str += f"[{recs[i]['run_id'][:12]} vs {recs[j]['run_id'][:12]}: {sim_val:.3f}] "
                    print(row_str)
                print(f"  Trigger Description : {trigger_desc}")
                if has_corrective:
                    print(f"  Safe Alternative    : {safe_alt}")
                else:
                    print(f"  Safe Alternative    : [EMPTY]")
                    print(f"  [NOTE: Needs manual authoring - no corrective/prescriptive language detected in cluster justifications]")
                print(f"  Example Snippet     : {len(example_snippets)} steps around step {medoid_rec['step_idx']}")
                for snip in example_snippets:
                    print(f"    Step {snip.step_index}: action={snip.action_name} | reasoning={snip.reasoning[:50]}...")

    print(f"\n{'='*60}")
    print(f"PATTERN MINING SUMMARY ({'DRY-RUN' if dry_run else 'PERSISTED'})")
    print(f"{'='*60}")
    print(f"Total Pattern Entries Synthesized: {len(mined_patterns)}")

    # Step 8: Save if dry-run is False
    if not dry_run:
        save_patterns(mined_patterns, output_path)
        print(f"Successfully saved {len(mined_patterns)} patterns to {output_path}")

    return mined_patterns


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4 Failure Pattern Mining Pipeline")
    parser.add_argument("--judged-data", default="findings/agenterrorbench_judged.json", help="Path to judged JSON dataset")
    parser.add_argument("--converted-data", default="findings/agenterrorbench_converted.json", help="Path to converted JSON dataset")
    parser.add_argument("--output", default="findings/pattern_library.json", help="Output pattern library path")
    parser.add_argument("--min-cluster-size", type=int, default=3, help="Minimum cluster size required to form a pattern")
    parser.add_argument("--similarity-threshold", type=float, default=0.35, help="Cosine similarity threshold for clustering")
    
    # BooleanOptionalAction supports --dry-run and --no-dry-run seamlessly in Python 3.9+
    parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=True, help="Dry run mode (default: True)")

    args = parser.parse_args()

    mine_patterns_pipeline(
        judged_data_path=args.judged_data,
        converted_data_path=args.converted_data,
        min_cluster_size=args.min_cluster_size,
        similarity_threshold=args.similarity_threshold,
        dry_run=args.dry_run,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
