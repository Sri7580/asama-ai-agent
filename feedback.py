"""
Closed-Loop Feedback & Dynamic Profile Learning Module.

Adapts agent interest vectors based on engagement:
- Positive Reinforcement: When an agent responds to a notified post with a comment,
  shift its interest embedding toward the post vector (+ learning_rate).
- Negative Feedback / Penalty: If an agent repeatedly ignores notifications,
  shift its interest embedding away from unengaged post vectors (- penalty_rate).
"""

import math
from typing import List, Dict, Any
from database import get_agent_by_id, update_agent_embedding, update_routing_status


LEARNING_RATE_ENGAGED = 0.08    # Positive gradient step
PENALTY_RATE_IGNORED = 0.03     # Negative gradient step


def normalize_vector(v: List[float]) -> List[float]:
    """L2 normalizes a vector."""
    norm = math.sqrt(sum(x * x for x in v))
    if norm > 1e-9:
        return [round(x / norm, 5) for x in v]
    return [0.0] * len(v)


def record_agent_engagement(post_id: str, agent_id: str, post_embedding: List[float]):
    """
    Called when an agent posts a comment on a notified post.
    Reinforces affinity between the agent's profile and the post content.
    """
    update_routing_status(post_id=post_id, target_agent_id=agent_id, new_status="ENGAGED")

    agent = get_agent_by_id(agent_id)
    if not agent:
        return

    curr_emb = agent["interest_embedding"]
    if len(curr_emb) != len(post_embedding):
        return

    # Positive gradient shift: W_new = W_old + lr * P_post
    new_emb = [
        c + LEARNING_RATE_ENGAGED * p
        for c, p in zip(curr_emb, post_embedding)
    ]
    normalized_new_emb = normalize_vector(new_emb)

    # Slight positive trust score increment
    update_agent_embedding(agent_id, normalized_new_emb, trust_score_delta=0.02)


def record_agent_ignored(post_id: str, agent_id: str, post_embedding: List[float]):
    """
    Called when a notification times out with no engagement from the target agent.
    Applies negative shift to avoid future spam.
    """
    update_routing_status(post_id=post_id, target_agent_id=agent_id, new_status="IGNORED")

    agent = get_agent_by_id(agent_id)
    if not agent:
        return

    curr_emb = agent["interest_embedding"]
    if len(curr_emb) != len(post_embedding):
        return

    # Negative gradient shift: W_new = W_old - lr * P_post
    new_emb = [
        c - PENALTY_RATE_IGNORED * p
        for c, p in zip(curr_emb, post_embedding)
    ]
    normalized_new_emb = normalize_vector(new_emb)

    update_agent_embedding(agent_id, normalized_new_emb, trust_score_delta=0.0)
