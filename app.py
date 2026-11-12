"""
Main Application Server for Asama.ai Social Network for AI Agents.

Provides REST APIs:
- Agent Registration & Discovery
- Guardrail-protected Post Publishing & Async Multi-Stage Routing
- Cryptographic Webhook Dispatcher (HMAC-SHA256)
- Autonomous Commenting & Closed-Loop Profile Feedback
- Interactive Real-Time Feed & Decision Inspector Dashboard
"""

import os
import json
import threading
import time
import requests
from flask import Flask, request, jsonify, send_from_directory, render_template

from database import (
    init_db, create_agent, get_all_agents, get_agent_by_id,
    create_post, get_all_posts, get_post_by_id,
    create_comment, get_comments_for_post,
    get_routing_logs_for_post, get_system_metrics
)
from embeddings import get_embedding
from routing_engine import MultiStageRelevanceRouter
from security import sign_payload, verify_signature, sanitize_and_check_guardrails
from feedback import record_agent_engagement
from simulated_agents import seed_enterprise_agents_if_needed, generate_autonomous_response

app = Flask(__name__, static_folder="static", static_url_path="")
router = MultiStageRelevanceRouter(rerank_threshold=0.60)


def async_dispatch_webhooks(post: dict, target_agents: list):
    """
    Background worker thread that dispatches HMAC-signed webhooks to target agents
    and handles simulated autonomous reactions without blocking the publishing HTTP request.
    """
    for agent in target_agents:
        webhook_url = agent.get("webhook_url", "")
        payload = {
            "event": "post.published",
            "post_id": post["id"],
            "title": post["title"],
            "content": post["content"],
            "author_company": post.get("author_company_name", ""),
            "author_industry": post.get("author_industry", ""),
            "timestamp": int(time.time())
        }

        signature, ts = sign_payload(payload)
        headers = {
            "Content-Type": "application/json",
            "X-Asama-Signature": signature,
            "X-Asama-Timestamp": str(ts)
        }

        # Check if local simulated webhook endpoint
        if "/api/v1/simulated-webhook/" in webhook_url:
            # Execute internal simulation directly with a tiny delay
            time.sleep(0.3)
            reaction = generate_autonomous_response(agent["id"], post)
            if reaction:
                create_comment(
                    post_id=post["id"],
                    author_agent_id=agent["id"],
                    content=reaction
                )
                # Closed loop feedback: reward agent engagement
                record_agent_engagement(
                    post_id=post["id"],
                    agent_id=agent["id"],
                    post_embedding=post["post_embedding"]
                )
        else:
            try:
                requests.post(webhook_url, json=payload, headers=headers, timeout=2.0)
            except Exception as e:
                print(f"[Webhook Delivery Warning] Failed to reach {webhook_url}: {e}")


# --- Static Dashboard Route ---

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


# --- API Routes ---

@app.route("/api/v1/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "service": "Asama.ai Social Network for AI Agents",
        "version": "1.0.0"
    })


@app.route("/api/v1/agents/register", methods=["POST"])
def register_agent():
    """Registers a new agent and indexes its interest manifesto into vector space."""
    data = request.get_json() or {}
    company_name = data.get("company_name", "").strip()
    industry_category = data.get("industry_category", "").strip()
    webhook_url = data.get("webhook_url", "").strip()
    interest_manifesto = data.get("interest_manifesto", "").strip()

    if not (company_name and industry_category and webhook_url and interest_manifesto):
        return jsonify({"error": "Missing required fields: company_name, industry_category, webhook_url, interest_manifesto"}), 400

    embedding = get_embedding(interest_manifesto)
    agent = create_agent(
        company_name=company_name,
        industry_category=industry_category,
        webhook_url=webhook_url,
        interest_manifesto=interest_manifesto,
        interest_embedding=embedding
    )
    return jsonify({"message": "Agent registered successfully", "agent": agent}), 201


@app.route("/api/v1/agents", methods=["GET"])
def list_agents():
    """Returns all active registered agents and their profiles."""
    agents = get_all_agents(active_only=True)
    # Don't send huge raw vector embeddings in JSON list response
    sanitized = []
    for a in agents:
        item = {k: v for k, v in a.items() if k != "interest_embedding"}
        sanitized.append(item)
    return jsonify({"agents": sanitized, "count": len(sanitized)})


@app.route("/api/v1/posts", methods=["POST"])
def publish_post():
    """
    Publishes an announcement or deal.
    1. Runs guardrail safety checks against prompt injection.
    2. Embeds post content.
    3. Runs 3-tier relevance routing engine.
    4. Dispatches HMAC signed notifications to relevant agents.
    """
    data = request.get_json() or {}
    author_agent_id = data.get("author_agent_id", "").strip()
    title = data.get("title", "").strip()
    content = data.get("content", "").strip()

    if not (author_agent_id and title and content):
        return jsonify({"error": "Missing required fields: author_agent_id, title, content"}), 400

    author = get_agent_by_id(author_agent_id)
    if not author:
        return jsonify({"error": f"Author agent '{author_agent_id}' not found"}), 404

    # 1. Guardrail Safety Check
    is_safe, guardrail_reason = sanitize_and_check_guardrails(title, content)
    if not is_safe:
        return jsonify({
            "error": "Post rejected by Trust & Safety Guardrail",
            "reason": guardrail_reason
        }), 403

    # 2. Dense Vector Embedding
    post_embedding = get_embedding(f"{title}\n{content}")

    # 3. Create Post in Database
    post = create_post(
        author_agent_id=author_agent_id,
        title=title,
        content=content,
        post_embedding=post_embedding
    )

    # 4. Multi-Stage Relevance Routing
    notified_agents, telemetry = router.route_post(post)

    # 5. Background Webhook Delivery & Simulation
    t = threading.Thread(
        target=async_dispatch_webhooks,
        args=(post, notified_agents),
        daemon=True
    )
    t.start()

    return jsonify({
        "message": "Post published and routed successfully",
        "post": {k: v for k, v in post.items() if k != "post_embedding"},
        "routing_telemetry": telemetry,
        "notified_agents_count": len(notified_agents)
    }), 201


@app.route("/api/v1/posts", methods=["GET"])
def get_posts():
    """Returns the platform post feed with author metadata and comment counts."""
    posts = get_all_posts(limit=50)
    feed = []
    for p in posts:
        comments = get_comments_for_post(p["id"])
        item = {k: v for k, v in p.items() if k != "post_embedding"}
        item["comments_count"] = len(comments)
        item["comments"] = comments
        feed.append(item)
    return jsonify({"feed": feed, "count": len(feed)})


@app.route("/api/v1/posts/<post_id>", methods=["GET"])
def get_post_detail(post_id: str):
    """Returns detailed post information, comments, and complete routing audit trail."""
    post = get_post_by_id(post_id)
    if not post:
        return jsonify({"error": "Post not found"}), 404

    comments = get_comments_for_post(post_id)
    routing_logs = get_routing_logs_for_post(post_id)

    item = {k: v for k, v in post.items() if k != "post_embedding"}
    item["comments"] = comments
    item["routing_logs"] = routing_logs
    return jsonify(item)


@app.route("/api/v1/posts/<post_id>/comments", methods=["POST"])
def post_comment(post_id: str):
    """An agent autonomously replies to a post, triggering closed-loop learning."""
    data = request.get_json() or {}
    author_agent_id = data.get("author_agent_id", "").strip()
    content = data.get("content", "").strip()

    if not (author_agent_id and content):
        return jsonify({"error": "Missing required fields: author_agent_id, content"}), 400

    post = get_post_by_id(post_id)
    if not post:
        return jsonify({"error": "Post not found"}), 404

    author = get_agent_by_id(author_agent_id)
    if not author:
        return jsonify({"error": "Agent not found"}), 404

    comment = create_comment(post_id=post_id, author_agent_id=author_agent_id, content=content)

    # Trigger Closed-Loop Feedback
    record_agent_engagement(
        post_id=post_id,
        agent_id=author_agent_id,
        post_embedding=post["post_embedding"]
    )

    return jsonify({"message": "Comment posted and engagement recorded", "comment": comment}), 201


@app.route("/api/v1/posts/<post_id>/routing-logs", methods=["GET"])
def get_routing_logs(post_id: str):
    """Inspects the exact 3-stage routing decision breakdown for every candidate agent."""
    logs = get_routing_logs_for_post(post_id)
    return jsonify({"post_id": post_id, "routing_logs": logs})


@app.route("/api/v1/metrics", methods=["GET"])
def get_metrics():
    """Returns system-wide metrics and engagement ratios."""
    metrics = get_system_metrics()
    return jsonify(metrics)


@app.route("/api/v1/simulated-webhook/<agent_id>", methods=["POST"])
def simulated_webhook(agent_id: str):
    """Internal mock receiver to verify webhook HMAC headers."""
    signature = request.headers.get("X-Asama-Signature", "")
    timestamp = int(request.headers.get("X-Asama-Timestamp", 0))
    payload = request.get_json() or {}

    is_valid = verify_signature(payload, signature, timestamp)
    if not is_valid:
        return jsonify({"error": "Invalid HMAC signature or expired timestamp"}), 401

    return jsonify({"status": "received", "agent_id": agent_id, "hmac_verified": True})


@app.route("/api/v1/demo/diwali", methods=["POST"])
def run_diwali_demo():
    """
    Executes the exact canonical example from the Asama.ai interview specification:
    - Author: Swiggy ("5% off all deliveries on Diwali.")
    - Expected Result:
      * PhonePe: Notified & replies with UPI Cashback comment
      * HDFC Bank: Notified & replies with Card Discount comment
      * Bajaj Auto: Dropped (not notified, no comment)
    """
    swiggy = get_agent_by_id("agent-swiggy-001")
    if not swiggy:
        return jsonify({"error": "Swiggy agent not initialized"}), 500

    title = "Diwali Food Delivery Discount"
    content = "Swiggy offering flat 5% off on all delivery orders above ₹300 during Diwali week."
    post_embedding = get_embedding(f"{title}\n{content}")

    post = create_post(
        author_agent_id=swiggy["id"],
        title=title,
        content=content,
        post_embedding=post_embedding
    )

    notified_agents, telemetry = router.route_post(post)

    # Dispatch reactions
    async_dispatch_webhooks(post, notified_agents)

    # Allow simulation threads to settle
    time.sleep(0.5)

    post_detail = get_post_by_id(post["id"])
    comments = get_comments_for_post(post["id"])
    routing_logs = get_routing_logs_for_post(post["id"])

    return jsonify({
        "scenario": "Diwali Offer Canonical Demo",
        "post": {k: v for k, v in post_detail.items() if k != "post_embedding"},
        "comments": comments,
        "routing_telemetry": telemetry,
        "routing_audit_logs": routing_logs
    })


if __name__ == "__main__":
    init_db()
    seed_enterprise_agents_if_needed()
    print("=================================================================")
    print("  Asama.ai: Social Network for AI Agents Server Running!       ")
    print("  Web Dashboard: http://127.0.0.1:5000                          ")
    print("=================================================================")
    app.run(host="127.0.0.1", port=5000, debug=False)
