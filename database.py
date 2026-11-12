"""
Database layer for AI Agent Social Network.
Manages SQLite storage for Agents, Posts, Routing Logs, Comments, and Subscriptions.
Provides clean transactional methods and thread-safe connections.
"""

import sqlite3
import json
import os
import uuid
from typing import List, Dict, Any, Optional

DB_PATH = os.path.join(os.path.dirname(__file__), "asama_network.db")


def get_connection():
    """Returns a SQLite connection with dict-like row factory and WAL mode for high concurrency."""
    conn = sqlite3.connect(DB_PATH, timeout=10.0, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    """Initializes the database schema."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.executescript("""
    -- 1. Agents Table
    CREATE TABLE IF NOT EXISTS agents (
        id TEXT PRIMARY KEY,
        company_name TEXT NOT NULL,
        industry_category TEXT NOT NULL,
        webhook_url TEXT NOT NULL,
        interest_manifesto TEXT NOT NULL,
        interest_embedding TEXT NOT NULL, -- JSON-serialized vector
        trust_score REAL DEFAULT 1.0,
        is_active INTEGER DEFAULT 1,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    -- 2. Posts Table
    CREATE TABLE IF NOT EXISTS posts (
        id TEXT PRIMARY KEY,
        author_agent_id TEXT NOT NULL,
        title TEXT NOT NULL,
        content TEXT NOT NULL,
        post_embedding TEXT NOT NULL, -- JSON-serialized vector
        expires_at TIMESTAMP,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (author_agent_id) REFERENCES agents(id) ON DELETE CASCADE
    );

    -- 3. Routing Logs Table (Tracks Stage 1, 2, 3 metrics & feedback)
    CREATE TABLE IF NOT EXISTS routing_logs (
        id TEXT PRIMARY KEY,
        post_id TEXT NOT NULL,
        target_agent_id TEXT NOT NULL,
        stage1_passed INTEGER DEFAULT 1,
        vector_score REAL NOT NULL,
        rerank_score REAL NOT NULL,
        final_score REAL NOT NULL,
        status TEXT NOT NULL, -- 'NOTIFIED', 'ENGAGED', 'DROPPED', 'FAILED'
        reason TEXT,
        latency_ms REAL DEFAULT 0.0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (post_id) REFERENCES posts(id) ON DELETE CASCADE,
        FOREIGN KEY (target_agent_id) REFERENCES agents(id) ON DELETE CASCADE
    );

    -- 4. Comments Table (Autonomous Agent Responses)
    CREATE TABLE IF NOT EXISTS comments (
        id TEXT PRIMARY KEY,
        post_id TEXT NOT NULL,
        author_agent_id TEXT NOT NULL,
        content TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (post_id) REFERENCES posts(id) ON DELETE CASCADE,
        FOREIGN KEY (author_agent_id) REFERENCES agents(id) ON DELETE CASCADE
    );

    -- 5. Subscriptions Table (Explicit topic or agent subscriptions)
    CREATE TABLE IF NOT EXISTS subscriptions (
        id TEXT PRIMARY KEY,
        subscriber_agent_id TEXT NOT NULL,
        publisher_agent_id TEXT,
        topic_tag TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (subscriber_agent_id) REFERENCES agents(id) ON DELETE CASCADE
    );

    -- Indexes for high-throughput query performance
    CREATE INDEX IF NOT EXISTS idx_agents_active ON agents(is_active);
    CREATE INDEX IF NOT EXISTS idx_posts_author ON posts(author_agent_id);
    CREATE INDEX IF NOT EXISTS idx_routing_post ON routing_logs(post_id);
    CREATE INDEX IF NOT EXISTS idx_comments_post ON comments(post_id);
    """)

    conn.commit()
    conn.close()


# --- Agent Operations ---

def create_agent(company_name: str, industry_category: str, webhook_url: str,
                 interest_manifesto: str, interest_embedding: List[float],
                 agent_id: Optional[str] = None) -> Dict[str, Any]:
    conn = get_connection()
    new_id = agent_id or str(uuid.uuid4())
    embedding_json = json.dumps(interest_embedding)
    conn.execute("""
        INSERT INTO agents (id, company_name, industry_category, webhook_url, interest_manifesto, interest_embedding)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (new_id, company_name, industry_category, webhook_url, interest_manifesto, embedding_json))
    conn.commit()
    agent = get_agent_by_id(new_id)
    conn.close()
    return agent


def get_agent_by_id(agent_id: str) -> Optional[Dict[str, Any]]:
    conn = get_connection()
    row = conn.execute("SELECT * FROM agents WHERE id = ?", (agent_id,)).fetchone()
    conn.close()
    if row:
        d = dict(row)
        d['interest_embedding'] = json.loads(d['interest_embedding'])
        return d
    return None


def get_all_agents(active_only: bool = True) -> List[Dict[str, Any]]:
    conn = get_connection()
    query = "SELECT * FROM agents"
    if active_only:
        query += " WHERE is_active = 1"
    rows = conn.execute(query).fetchall()
    conn.close()
    result = []
    for r in rows:
        d = dict(r)
        d['interest_embedding'] = json.loads(d['interest_embedding'])
        result.append(d)
    return result


def update_agent_embedding(agent_id: str, new_embedding: List[float], trust_score_delta: float = 0.0):
    conn = get_connection()
    conn.execute("""
        UPDATE agents
        SET interest_embedding = ?,
            trust_score = MAX(0.1, MIN(2.0, trust_score + ?))
        WHERE id = ?
    """, (json.dumps(new_embedding), trust_score_delta, agent_id))
    conn.commit()
    conn.close()


# --- Post Operations ---

def create_post(author_agent_id: str, title: str, content: str,
                post_embedding: List[float], post_id: Optional[str] = None) -> Dict[str, Any]:
    conn = get_connection()
    new_id = post_id or str(uuid.uuid4())
    embedding_json = json.dumps(post_embedding)
    conn.execute("""
        INSERT INTO posts (id, author_agent_id, title, content, post_embedding)
        VALUES (?, ?, ?, ?, ?)
    """, (new_id, author_agent_id, title, content, embedding_json))
    conn.commit()
    post = get_post_by_id(new_id)
    conn.close()
    return post


def get_post_by_id(post_id: str) -> Optional[Dict[str, Any]]:
    conn = get_connection()
    row = conn.execute("""
        SELECT p.*, a.company_name as author_company_name, a.industry_category as author_industry
        FROM posts p
        JOIN agents a ON p.author_agent_id = a.id
        WHERE p.id = ?
    """, (post_id,)).fetchone()
    conn.close()
    if row:
        d = dict(row)
        d['post_embedding'] = json.loads(d['post_embedding'])
        return d
    return None


def get_all_posts(limit: int = 50) -> List[Dict[str, Any]]:
    conn = get_connection()
    rows = conn.execute("""
        SELECT p.*, a.company_name as author_company_name, a.industry_category as author_industry
        FROM posts p
        JOIN agents a ON p.author_agent_id = a.id
        ORDER BY p.created_at DESC
        LIMIT ?
    """, (limit,)).fetchall()
    conn.close()
    result = []
    for r in rows:
        d = dict(r)
        d['post_embedding'] = json.loads(d['post_embedding'])
        result.append(d)
    return result


# --- Comment Operations ---

def create_comment(post_id: str, author_agent_id: str, content: str) -> Dict[str, Any]:
    conn = get_connection()
    new_id = str(uuid.uuid4())
    conn.execute("""
        INSERT INTO comments (id, post_id, author_agent_id, content)
        VALUES (?, ?, ?, ?)
    """, (new_id, post_id, author_agent_id, content))
    conn.commit()
    row = conn.execute("""
        SELECT c.*, a.company_name as author_company_name, a.industry_category as author_industry
        FROM comments c
        JOIN agents a ON c.author_agent_id = a.id
        WHERE c.id = ?
    """, (new_id,)).fetchone()
    conn.close()
    return dict(row)


def get_comments_for_post(post_id: str) -> List[Dict[str, Any]]:
    conn = get_connection()
    rows = conn.execute("""
        SELECT c.*, a.company_name as author_company_name, a.industry_category as author_industry
        FROM comments c
        JOIN agents a ON c.author_agent_id = a.id
        WHERE c.post_id = ?
        ORDER BY c.created_at ASC
    """, (post_id,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# --- Routing Logs Operations ---

def log_routing_decision(post_id: str, target_agent_id: str, stage1_passed: bool,
                         vector_score: float, rerank_score: float, final_score: float,
                         status: str, reason: str, latency_ms: float = 0.0) -> str:
    conn = get_connection()
    log_id = str(uuid.uuid4())
    conn.execute("""
        INSERT INTO routing_logs (id, post_id, target_agent_id, stage1_passed, vector_score, rerank_score, final_score, status, reason, latency_ms)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (log_id, post_id, target_agent_id, 1 if stage1_passed else 0, vector_score, rerank_score, final_score, status, reason, latency_ms))
    conn.commit()
    conn.close()
    return log_id


def update_routing_status(post_id: str, target_agent_id: str, new_status: str):
    conn = get_connection()
    conn.execute("""
        UPDATE routing_logs
        SET status = ?
        WHERE post_id = ? AND target_agent_id = ?
    """, (new_status, post_id, target_agent_id))
    conn.commit()
    conn.close()


def get_routing_logs_for_post(post_id: str) -> List[Dict[str, Any]]:
    conn = get_connection()
    rows = conn.execute("""
        SELECT r.*, a.company_name as target_company_name, a.industry_category as target_industry
        FROM routing_logs r
        JOIN agents a ON r.target_agent_id = a.id
        WHERE r.post_id = ?
        ORDER BY r.final_score DESC
    """, (post_id,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_system_metrics() -> Dict[str, Any]:
    conn = get_connection()
    total_agents = conn.execute("SELECT COUNT(*) FROM agents").fetchone()[0]
    total_posts = conn.execute("SELECT COUNT(*) FROM posts").fetchone()[0]
    total_comments = conn.execute("SELECT COUNT(*) FROM comments").fetchone()[0]
    total_routed = conn.execute("SELECT COUNT(*) FROM routing_logs WHERE status = 'NOTIFIED'").fetchone()[0]
    total_engaged = conn.execute("SELECT COUNT(*) FROM routing_logs WHERE status = 'ENGAGED'").fetchone()[0]
    conn.close()
    engagement_rate = round((total_engaged / total_routed * 100), 1) if total_routed > 0 else 0.0
    return {
        "total_agents": total_agents,
        "total_posts": total_posts,
        "total_comments": total_comments,
        "total_notifications": total_routed,
        "total_engagements": total_engaged,
        "engagement_rate_percent": engagement_rate
    }
