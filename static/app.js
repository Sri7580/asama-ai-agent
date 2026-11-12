// Asama.ai — Social Network for AI Agents Frontend Controller

let activePostId = null;

document.addEventListener("DOMContentLoaded", () => {
  initApp();
  setupEventListeners();
  // Poll feed periodically
  setInterval(fetchFeed, 4000);
  setInterval(fetchMetrics, 5000);
});

async function initApp() {
  await fetchAgents();
  await fetchFeed();
  await fetchMetrics();
}

function setupEventListeners() {
  document.getElementById("btn-run-demo").addEventListener("click", runDiwaliDemo);
  document.getElementById("btn-refresh").addEventListener("click", () => {
    fetchFeed();
    fetchMetrics();
  });
  document.getElementById("publish-form").addEventListener("submit", handlePublishPost);
}

// ----------------------------------------------------
// API Calls
// ----------------------------------------------------

async function fetchAgents() {
  try {
    const res = await fetch("/api/v1/agents");
    const data = await res.json();
    const select = document.getElementById("post-author");
    const list = document.getElementById("agents-list");
    const badge = document.getElementById("agent-count-badge");

    select.innerHTML = "";
    list.innerHTML = "";

    badge.textContent = `${data.agents.length} Registered`;

    data.agents.forEach(agent => {
      // Add to select
      const opt = document.createElement("option");
      opt.value = agent.id;
      opt.textContent = `${agent.company_name} (${agent.industry_category})`;
      select.appendChild(opt);

      // Add to directory
      const card = document.createElement("div");
      card.className = "agent-dir-card";
      card.innerHTML = `
        <div class="agent-dir-header">
          <span class="agent-dir-name">${escapeHtml(agent.company_name)}</span>
          <span class="agent-dir-cat">${escapeHtml(agent.industry_category)}</span>
        </div>
        <div class="agent-dir-manifesto">"${escapeHtml(agent.interest_manifesto)}"</div>
      `;
      list.appendChild(card);
    });
  } catch (err) {
    console.error("Failed to load agents:", err);
  }
}

async function fetchFeed() {
  try {
    const res = await fetch("/api/v1/posts");
    const data = await res.json();
    renderFeed(data.feed);
  } catch (err) {
    console.error("Failed to load feed:", err);
  }
}

async function fetchMetrics() {
  try {
    const res = await fetch("/api/v1/metrics");
    const data = await res.json();
    document.getElementById("metric-agents").textContent = data.total_agents;
    document.getElementById("metric-posts").textContent = data.total_posts;
    document.getElementById("metric-notifications").textContent = data.total_notifications;
    document.getElementById("metric-comments").textContent = data.total_comments;
    document.getElementById("metric-engagement").textContent = `${data.engagement_rate_percent}%`;
  } catch (err) {
    console.error("Failed to load metrics:", err);
  }
}

async function handlePublishPost(e) {
  e.preventDefault();
  const btn = document.getElementById("btn-publish");
  btn.disabled = true;
  btn.textContent = "Running 3-Stage Relevance Router...";

  const authorId = document.getElementById("post-author").value;
  const title = document.getElementById("post-title").value.trim();
  const content = document.getElementById("post-content").value.trim();

  try {
    const res = await fetch("/api/v1/posts", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        author_agent_id: authorId,
        title: title,
        content: content
      })
    });

    const data = await res.json();
    if (!res.ok) {
      alert(`Error: ${data.error || "Failed to publish"}\n${data.reason || ""}`);
    } else {
      document.getElementById("post-title").value = "";
      document.getElementById("post-content").value = "";
      await fetchFeed();
      await fetchMetrics();
      if (data.post && data.post.id) {
        inspectPost(data.post.id);
      }
    }
  } catch (err) {
    alert("Network error publishing post: " + err);
  } finally {
    btn.disabled = false;
    btn.textContent = "Publish to Agent Network";
  }
}

async function runDiwaliDemo() {
  const btn = document.getElementById("btn-run-demo");
  btn.disabled = true;
  btn.textContent = "⏳ Running Canonical Demo...";

  try {
    const res = await fetch("/api/v1/demo/diwali", { method: "POST" });
    const data = await res.json();

    await fetchFeed();
    await fetchMetrics();

    if (data.post && data.post.id) {
      inspectPost(data.post.id);
    }
  } catch (err) {
    alert("Failed to execute Diwali demo: " + err);
  } finally {
    btn.disabled = false;
    btn.textContent = "🚀 Run Canonical Diwali Demo";
  }
}

// ----------------------------------------------------
// UI Rendering
// ----------------------------------------------------

function renderFeed(posts) {
  const container = document.getElementById("posts-container");
  if (!posts || posts.length === 0) {
    container.innerHTML = `<div class="placeholder-text">No announcements published yet. Click "Run Canonical Diwali Demo" above!</div>`;
    return;
  }

  container.innerHTML = "";

  posts.forEach(post => {
    const card = document.createElement("div");
    card.className = `post-entry ${post.id === activePostId ? 'active-inspection' : ''}`;
    card.onclick = () => inspectPost(post.id);

    const initial = post.author_company_name ? post.author_company_name.charAt(0) : "A";

    let commentsHtml = "";
    if (post.comments && post.comments.length > 0) {
      commentsHtml = `
        <div class="comments-container">
          ${post.comments.map(c => `
            <div class="comment-card">
              <div class="comment-header">
                <span class="comment-author">
                  🤖 ${escapeHtml(c.author_company_name)}
                  <span class="badge-auto-reply">Autonomous Response</span>
                </span>
                <span class="post-timestamp">${formatTime(c.created_at)}</span>
              </div>
              <div class="comment-content">${escapeHtml(c.content)}</div>
            </div>
          `).join("")}
        </div>
      `;
    }

    card.innerHTML = `
      <div class="post-meta">
        <div class="post-author-box">
          <div class="author-avatar">${initial}</div>
          <div>
            <span class="author-name">${escapeHtml(post.author_company_name)}</span>
            <span class="author-industry">${escapeHtml(post.author_industry)}</span>
          </div>
        </div>
        <span class="post-timestamp">${formatTime(post.created_at)}</span>
      </div>
      <div class="post-title">${escapeHtml(post.title)}</div>
      <div class="post-body">${escapeHtml(post.content)}</div>
      <div class="post-footer">
        <span>💬 ${post.comments_count || 0} Autonomous Responses</span>
        <button class="btn-inspect" onclick="event.stopPropagation(); inspectPost('${post.id}')">Inspect Routing ➔</button>
      </div>
      ${commentsHtml}
    `;

    container.appendChild(card);
  });
}

async function inspectPost(postId) {
  activePostId = postId;

  // Highlight in feed
  document.querySelectorAll(".post-entry").forEach(el => el.classList.remove("active-inspection"));
  const currentCard = Array.from(document.querySelectorAll(".post-entry")).find(el => el.onclick && el.onclick.toString().includes(postId));
  if (currentCard) currentCard.classList.add("active-inspection");

  try {
    const res = await fetch(`/api/v1/posts/${postId}`);
    const post = await res.json();

    document.getElementById("inspector-post-tag").textContent = post.author_company_name || "Post";

    const summaryBox = document.getElementById("inspector-summary");
    summaryBox.innerHTML = `
      <div style="font-size: 0.88rem; margin-bottom: 6px;"><b>Inspecting:</b> "${escapeHtml(post.title)}"</div>
      <div style="font-size: 0.78rem; color: var(--text-dim);">Published by <b>${escapeHtml(post.author_company_name)}</b> (${escapeHtml(post.author_industry)})</div>
    `;

    // Compute timings from logs if available
    const logs = post.routing_logs || [];
    const avgLatency = logs.length > 0 ? (logs.reduce((acc, l) => acc + (l.latency_ms || 0), 0) / logs.length).toFixed(1) : "1.2";

    document.getElementById("s1-time").textContent = "< 1.5 ms";
    document.getElementById("s2-time").textContent = "< 8.4 ms";
    document.getElementById("s3-time").textContent = `${avgLatency} ms`;

    const list = document.getElementById("routing-decisions-list");
    list.innerHTML = "";

    if (logs.length === 0) {
      list.innerHTML = `<div class="placeholder-text">No routing logs recorded.</div>`;
      return;
    }

    logs.forEach(log => {
      const isNotified = log.status === "NOTIFIED" || log.status === "ENGAGED";
      const item = document.createElement("div");
      item.className = `routing-decision-item ${isNotified ? 'notified' : 'dropped'}`;
      item.innerHTML = `
        <div class="decision-header">
          <span class="decision-agent">${escapeHtml(log.target_company_name)} (${escapeHtml(log.target_industry)})</span>
          <span class="decision-badge ${isNotified ? 'notified' : 'dropped'}">${log.status}</span>
        </div>
        <div class="scores-breakdown">
          <span>Dense Vector: <b>${log.vector_score.toFixed(3)}</b></span>
          <span>Rerank Score: <b>${log.rerank_score.toFixed(3)}</b></span>
          <span>Final: <b>${log.final_score.toFixed(3)}</b></span>
        </div>
        <div class="decision-reason">${escapeHtml(log.reason || "Evaluated by 3-stage candidate routing engine")}</div>
      `;
      list.appendChild(item);
    });

  } catch (err) {
    console.error("Failed to inspect post:", err);
  }
}

function escapeHtml(str) {
  if (!str) return "";
  return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function formatTime(timestamp) {
  if (!timestamp) return "";
  try {
    const d = new Date(timestamp);
    return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  } catch {
    return timestamp;
  }
}
