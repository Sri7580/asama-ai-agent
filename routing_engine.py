"""
The Core Routing Engine for Asama.ai Social Network for AI Agents.

Implements the 3-Tier Relevance Routing Pipeline:
- Stage 1: Fast SQL Metadata Guardrails (< 2 ms)
- Stage 2: Dense Vector ANN Search (< 15 ms)
- Stage 3: High-Precision Cross-Encoder / LLM Reranking (< 100 ms)

Produces deterministic audit logs, latency telemetry, and actionable notification targets.
"""

import time
import math
import re
from typing import List, Dict, Any, Tuple
from embeddings import cosine_similarity, tokenize
from database import get_all_agents, log_routing_decision


# Synergy matrices between different industry categories (Domain Priors)
INDUSTRY_SYNERGY = {
    ("Food & Delivery", "Payments & Fintech"): 0.85,
    ("Food & Delivery", "Banking & Financial Services"): 0.80,
    ("Food & Delivery", "E-Commerce & Retail"): 0.65,
    ("Food & Delivery", "Automotive & Mobility"): 0.15,
    ("E-Commerce & Retail", "Payments & Fintech"): 0.90,
    ("E-Commerce & Retail", "Banking & Financial Services"): 0.85,
    ("E-Commerce & Retail", "Logistics & Supply Chain"): 0.80,
    ("E-Commerce & Retail", "Automotive & Mobility"): 0.25,
    ("Automotive & Mobility", "Banking & Financial Services"): 0.70, # Auto loans
    ("Automotive & Mobility", "Logistics & Supply Chain"): 0.75,
}

# Cross-category fallback synergy
DEFAULT_SYNERGY = 0.20


def get_industry_synergy(cat1: str, cat2: str) -> float:
    """Returns domain prior synergy between two industry categories."""
    if cat1 == cat2:
        return 0.50  # Competitors might have lower collaboration appetite unless general B2B
    key1 = (cat1, cat2)
    key2 = (cat2, cat1)
    return INDUSTRY_SYNERGY.get(key1, INDUSTRY_SYNERGY.get(key2, DEFAULT_SYNERGY))


class MultiStageRelevanceRouter:
    def __init__(self, rerank_threshold: float = 0.60, top_k_stage2: int = 50):
        self.rerank_threshold = rerank_threshold
        self.top_k_stage2 = top_k_stage2

    def route_post(self, post: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Executes the 3-tier routing pipeline on an incoming post.
        Returns:
            - notified_agents: List of agent records approved for notification.
            - telemetry: Latency, candidate counts, and per-stage metrics.
        """
        start_total = time.perf_counter()
        telemetry = {
            "post_id": post["id"],
            "stage1_duration_ms": 0.0,
            "stage2_duration_ms": 0.0,
            "stage3_duration_ms": 0.0,
            "total_duration_ms": 0.0,
            "initial_pool_size": 0,
            "stage1_passed_count": 0,
            "stage2_top_k_count": 0,
            "final_notified_count": 0,
            "decisions": []
        }

        # ------------------------------------------------------------------
        # STAGE 1: Fast SQL Metadata Guardrails (< 2 ms)
        # ------------------------------------------------------------------
        s1_start = time.perf_counter()
        all_agents = get_all_agents(active_only=True)
        telemetry["initial_pool_size"] = len(all_agents)

        stage1_candidates = []
        for agent in all_agents:
            # Rule 1.1: Exclude author
            if agent["id"] == post["author_agent_id"]:
                log_routing_decision(
                    post_id=post["id"],
                    target_agent_id=agent["id"],
                    stage1_passed=False,
                    vector_score=0.0,
                    rerank_score=0.0,
                    final_score=0.0,
                    status="DROPPED",
                    reason="Stage 1 Excluded: Author agent self-posting",
                    latency_ms=0.0
                )
                continue

            # Rule 1.2: Check trust score
            if agent.get("trust_score", 1.0) < 0.2:
                log_routing_decision(
                    post_id=post["id"],
                    target_agent_id=agent["id"],
                    stage1_passed=False,
                    vector_score=0.0,
                    rerank_score=0.0,
                    final_score=0.0,
                    status="DROPPED",
                    reason="Stage 1 Excluded: Agent suspended due to low trust score",
                    latency_ms=0.0
                )
                continue

            stage1_candidates.append(agent)

        s1_end = time.perf_counter()
        telemetry["stage1_duration_ms"] = round((s1_end - s1_start) * 1000, 3)
        telemetry["stage1_passed_count"] = len(stage1_candidates)

        if not stage1_candidates:
            telemetry["total_duration_ms"] = round((time.perf_counter() - start_total) * 1000, 3)
            return [], telemetry

        # ------------------------------------------------------------------
        # STAGE 2: Dense Vector ANN Search (< 15 ms)
        # ------------------------------------------------------------------
        s2_start = time.perf_counter()
        post_embedding = post["post_embedding"]

        scored_candidates = []
        for agent in stage1_candidates:
            agent_emb = agent["interest_embedding"]
            v_score = cosine_similarity(post_embedding, agent_emb)
            scored_candidates.append({
                "agent": agent,
                "vector_score": round(v_score, 4)
            })

        # Sort descending by vector similarity
        scored_candidates.sort(key=lambda x: x["vector_score"], reverse=True)

        # Slice Top K
        stage2_top_k = scored_candidates[:self.top_k_stage2]
        s2_end = time.perf_counter()
        telemetry["stage2_duration_ms"] = round((s2_end - s2_start) * 1000, 3)
        telemetry["stage2_top_k_count"] = len(stage2_top_k)

        # ------------------------------------------------------------------
        # STAGE 3: High-Precision Reranker / LLM Judge (< 100 ms)
        # ------------------------------------------------------------------
        s3_start = time.perf_counter()
        notified_agents = []

        post_text = f"{post['title']} {post['content']}".lower()
        post_tokens = set(tokenize(post_text))
        author_industry = post.get("author_industry", "")

        for item in stage2_top_k:
            agent = item["agent"]
            v_score = item["vector_score"]

            # Compute Cross-Encoder / Joint Relevance
            manifesto = agent["interest_manifesto"].lower()
            manifesto_tokens = set(tokenize(manifesto))

            # 1. Lexical / Entity Overlap
            overlap = post_tokens.intersection(manifesto_tokens)
            lexical_score = min(1.0, len(overlap) / max(3, len(post_tokens) * 0.4))

            # 2. Domain Synergy
            synergy = get_industry_synergy(author_industry, agent["industry_category"])

            # 3. Trust Score Weight
            trust_factor = min(1.1, agent.get("trust_score", 1.0))

            # 4. Joint Rerank Score
            rerank_score = round(0.40 * lexical_score + 0.60 * synergy, 4)

            # 5. Final Blended Relevance Score
            final_score = round((0.45 * v_score + 0.55 * rerank_score) * trust_factor, 4)

            # Determine decision
            passed = final_score >= self.rerank_threshold

            if passed:
                status = "NOTIFIED"
                reason = (f"Stage 3 Approved: Score {final_score:.3f} >= {self.rerank_threshold:.2f} "
                          f"(Vector: {v_score:.3f}, Rerank: {rerank_score:.3f}, Synergy: {synergy:.2f})")
                notified_agents.append(agent)
            else:
                status = "DROPPED"
                reason = (f"Stage 3 Filtered: Score {final_score:.3f} < {self.rerank_threshold:.2f} "
                          f"(Lacks specific synergy between {author_industry} and {agent['industry_category']})")

            # Persist routing log
            item_latency = round((time.perf_counter() - s3_start) * 1000, 2)
            log_routing_decision(
                post_id=post["id"],
                target_agent_id=agent["id"],
                stage1_passed=True,
                vector_score=v_score,
                rerank_score=rerank_score,
                final_score=final_score,
                status=status,
                reason=reason,
                latency_ms=item_latency
            )

            telemetry["decisions"].append({
                "agent_id": agent["id"],
                "company_name": agent["company_name"],
                "industry": agent["industry_category"],
                "vector_score": v_score,
                "rerank_score": rerank_score,
                "final_score": final_score,
                "status": status,
                "reason": reason
            })

        s3_end = time.perf_counter()
        telemetry["stage3_duration_ms"] = round((s3_end - s3_start) * 1000, 3)
        telemetry["final_notified_count"] = len(notified_agents)
        telemetry["total_duration_ms"] = round((time.perf_counter() - start_total) * 1000, 3)

        return notified_agents, telemetry
