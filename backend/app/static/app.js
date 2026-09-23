const state = {
  accessToken: localStorage.getItem("ops_access_token") || "",
  refreshToken: localStorage.getItem("ops_refresh_token") || "",
  userId: localStorage.getItem("ops_user_id") || "",
  orgId: localStorage.getItem("ops_org_id") || "",
  orgName: localStorage.getItem("ops_org_name") || "",
  role: localStorage.getItem("ops_role") || "member",
  authMode: "signin",
  kbs: [],
  approvals: [],
  tickets: [],
  members: [],
  selectedKb: null,
  activeView: "overview",
  socket: null,
  chatContextVersion: 0,
};

/**
 * Single-flight guard for token refresh. The backend rotates refresh tokens and
 * revokes the previous ``jti``, so two concurrent refreshes would make the
 * second one fail and sign the user out. Every 401 shares one in-flight call.
 */
let refreshInFlight = null;

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
const initialChatMarkup = $("#chat-messages")?.innerHTML || "";

function workspaceContextKey() {
  return `${state.userId}\u0000${state.orgId}\u0000${state.role}`;
}

function isCurrentChatContext(contextKey, contextVersion) {
  return contextVersion === state.chatContextVersion && contextKey === workspaceContextKey();
}

/**
 * Clear all chat UI and invalidate in-flight work when the workspace identity
 * changes. Runs are deliberately not shared across users, organisations, or
 * roles, even though this demo keeps the visible transcript in the DOM.
 */
function resetAgentChat() {
  state.chatContextVersion += 1;
  if (state.socket) {
    state.socket.close();
    state.socket = null;
  }
  const messages = $("#chat-messages");
  if (messages) messages.innerHTML = initialChatMarkup;
  bindPromptSuggestions();
  const details = $("#run-details");
  if (details) {
    details.innerHTML = '<p class="empty-state">Your trace, retrieval count, and approval state will appear here after a run.</p>';
  }
  const prompt = $("#agent-prompt");
  if (prompt) prompt.value = "";
  const submit = $("#agent-form button[type=submit]");
  if (submit) setLoading(submit, false);
}

function iconMarkup(name, className = "icon") {
  return `<svg class="${className}" aria-hidden="true"><use href="#icon-${name}"></use></svg>`;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

const ANSWER_HIGHLIGHTS = [
  "3-5 个工作日",
  "7 个工作日",
  "idempotency key",
  "Retry-After",
  "transaction ID",
  "refund ID",
  "trace ID",
  "high priority",
  "waiting_approval",
  "organization_id",
  "content_hash",
  "content hash",
  "SQLAlchemy 2.0",
  "AsyncSession",
  "pgvector",
  "Recall@5",
  "MRR",
  "Celery",
  "Redis",
  "least privilege",
  "tenant scope",
  "dead-letter queue",
  "exponential backoff",
  "approved",
  "pending",
  "Authorization",
  "Bearer",
  "401",
  "429",
  "2xx",
  "P1",
  "P2",
  "p95",
];

function highlightAnswerTerms(escapedText) {
  const terms = ANSWER_HIGHLIGHTS
    .slice()
    .sort((left, right) => right.length - left.length)
    .map((term) => term.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
  if (!terms.length) return escapedText;
  return escapedText.replace(new RegExp(`(${terms.join("|")})`, "gi"), '<mark class="answer-highlight">$1</mark>');
}

function renderInlineAnswer(text) {
  const escaped = escapeHtml(text).replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
  return highlightAnswerTerms(escaped);
}

function renderAssistantContent(content) {
  const lines = String(content ?? "").split(/\r?\n/);
  let html = '<div class="answer-body">';
  let listType = null;
  let sectionOpen = false;

  const closeList = () => {
    if (listType) {
      html += `</${listType}>`;
      listType = null;
    }
  };
  const closeSection = () => {
    closeList();
    if (sectionOpen) {
      html += "</section>";
      sectionOpen = false;
    }
  };
  const openSection = (title, extraClass = "") => {
    closeSection();
    html += `<section class="answer-section ${extraClass}"><h4>${escapeHtml(title)}</h4>`;
    sectionOpen = true;
  };

  lines.forEach((rawLine) => {
    const line = rawLine.trim();
    if (!line) {
      closeList();
      return;
    }
    const lead = line.match(/^结论[:：]\s*(.*)$/);
    if (lead) {
      closeSection();
      html += `<div class="answer-lead"><span>结论</span><p>${renderInlineAnswer(lead[1])}</p></div>`;
      return;
    }
    const section = line.match(/^(关键点|处理步骤|来源)[:：]?\s*(.*)$/);
    if (section) {
      const sectionClass = section[1] === "来源" ? "answer-sources" : "";
      openSection(section[1], sectionClass);
      if (section[2]) html += `<p>${renderInlineAnswer(section[2])}</p>`;
      return;
    }
    const bullet = line.match(/^[-*]\s+(.+)$/);
    const numbered = line.match(/^\d+[.)]\s+(.+)$/);
    if (bullet || numbered) {
      const nextType = bullet ? "ul" : "ol";
      if (listType !== nextType) {
        closeList();
        html += `<${nextType} class="answer-list">`;
        listType = nextType;
      }
      html += `<li>${renderInlineAnswer((bullet || numbered)[1])}</li>`;
      return;
    }
    closeList();
    html += `<p>${renderInlineAnswer(line)}</p>`;
  });
  closeSection();
  return `${html}</div>`;
}

function authHeaders(json = false) {
  const headers = {};
  if (state.accessToken) headers["Authorization"] = `Bearer ${state.accessToken}`;
  if (state.orgId) headers["X-Org-ID"] = state.orgId;
  if (json) headers["Content-Type"] = "application/json";
  return headers;
}

/** Pull the human-readable reason out of the backend's error envelope. */
function errorMessage(payload, status) {
  if (typeof payload === "string" && payload.trim()) return payload;
  return payload?.error?.message || payload?.detail || payload?.message || `Request failed (${status})`;
}

function storeTokens(accessToken, refreshToken) {
  state.accessToken = accessToken;
  state.refreshToken = refreshToken;
  localStorage.setItem("ops_access_token", accessToken);
  if (refreshToken) localStorage.setItem("ops_refresh_token", refreshToken);
}

function clearTokens() {
  state.accessToken = "";
  state.refreshToken = "";
  localStorage.removeItem("ops_access_token");
  localStorage.removeItem("ops_refresh_token");
}

/**
 * Exchange the refresh token for a new pair. Returns the new access token, or
 * null when the refresh token is gone/revoked. Never throws — a failed refresh
 * is a normal sign-out, not an error to surface.
 */
function refreshAccessToken() {
  if (refreshInFlight) return refreshInFlight;
  if (!state.refreshToken) return Promise.resolve(null);
  refreshInFlight = (async () => {
    try {
      const response = await fetch("/v1/auth/refresh", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ refresh_token: state.refreshToken }),
      });
      if (!response.ok) return null;
      const payload = await response.json();
      storeTokens(payload.access_token, payload.refresh_token);
      return payload.access_token;
    } catch (_) {
      return null;
    } finally {
      refreshInFlight = null;
    }
  })();
  return refreshInFlight;
}

async function api(path, options = {}, { retry = true } = {}) {
  const isJsonBody = typeof options.body === "string";
  const response = await fetch(`/v1${path}`, {
    ...options,
    headers: { ...authHeaders(isJsonBody), ...(options.headers || {}) },
  });
  const contentType = response.headers.get("content-type") || "";
  const payload = contentType.includes("application/json") ? await response.json() : await response.text();

  // An expired access token is recoverable exactly once per request.
  if (response.status === 401 && retry && state.refreshToken) {
    const token = await refreshAccessToken();
    if (token) return api(path, options, { retry: false });
    signOut({ silent: true });
  }

  if (!response.ok) {
    const error = new Error(errorMessage(payload, response.status));
    error.status = response.status;
    throw error;
  }
  return payload;
}

function showToast(message, type = "success") {
  const toast = document.createElement("div");
  toast.className = `toast ${type}`;
  toast.textContent = message;
  $("#toast-region").appendChild(toast);
  window.setTimeout(() => toast.remove(), 4200);
}

function setLoading(button, loading, label) {
  if (!button) return;
  button.disabled = loading;
  if (loading) {
    button.dataset.originalMarkup = button.innerHTML;
    button.innerHTML = `<span>${escapeHtml(label || "Working...")}</span>`;
  } else if (button.dataset.originalMarkup) {
    button.innerHTML = button.dataset.originalMarkup;
    delete button.dataset.originalMarkup;
  }
}

function saveIdentity() {
  localStorage.setItem("ops_user_id", state.userId);
  localStorage.setItem("ops_org_id", state.orgId);
  localStorage.setItem("ops_org_name", state.orgName);
  localStorage.setItem("ops_role", state.role);
}

function clearWorkspaceState() {
  state.userId = "";
  state.orgId = "";
  state.orgName = "";
  state.role = "member";
  state.kbs = [];
  state.approvals = [];
  state.tickets = [];
  state.members = [];
  state.selectedKb = null;
  localStorage.removeItem("ops_user_id");
  localStorage.removeItem("ops_org_id");
  localStorage.removeItem("ops_org_name");
  localStorage.setItem("ops_role", state.role);
  resetAgentChat();
}

/** Drop all session state and return to the sign-in panel. */
function signOut({ silent = false } = {}) {
  const refreshToken = state.refreshToken;
  clearTokens();
  clearWorkspaceState();
  showAuth();
  if (refreshToken) {
    // Best effort: the local session is already gone either way.
    fetch("/v1/auth/logout", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ refresh_token: refreshToken }),
    }).catch(() => {});
  }
  if (!silent) showToast("已退出登录。");
}

function showWorkspace() {
  $("#onboarding").classList.add("hidden");
  $("#workspace").classList.remove("hidden");
  $("#user-label").textContent = state.userId;
  $("#user-avatar").textContent = state.userId.slice(0, 1).toUpperCase();
  $("#workspace-name").textContent = state.orgName || state.orgId.slice(0, 12);
  $("#workspace-role").textContent = state.role;
  $("#stat-role").textContent = state.role;
  $("#member-permission-note").textContent = ["owner", "admin"].includes(state.role)
    ? "Admins and owners can add or update teammates."
    : "Only admins and owners can change organization membership.";
  const demoButton = $("#demo-seed-button");
  if (demoButton) demoButton.disabled = state.role === "viewer";
  renderKnowledgeChecks();
}

/** Switch the auth panel between sign-in and register. */
function setAuthMode(mode) {
  state.authMode = mode;
  const registering = mode === "register";
  $("#auth-name-field").classList.toggle("hidden", !registering);
  $("#auth-org-field").classList.toggle("hidden", !registering);
  $("#auth-name").required = registering;
  $("#auth-org").required = registering;
  $("#onboarding-title").textContent = registering ? "创建工作区" : "登录到工作区";
  $("#auth-subtitle").textContent = registering
    ? "创建一个商户组织，并自动载入示例 runbook。"
    : "用邮箱和密码进入你的商户组织，管理知识库、Agent 运行、审批与工单。";
  $("#auth-submit").textContent = registering ? "Create workspace" : "Sign in";
  $("#auth-toggle").textContent = registering
    ? "已经有账号？返回登录"
    : "还没有账号？注册新工作区";
}

function showAuth() {
  $("#workspace").classList.add("hidden");
  $("#onboarding").classList.remove("hidden");
  setAuthMode("signin");
}

function formatDate(value) {
  if (!value) return "—";
  return new Intl.DateTimeFormat("en", { dateStyle: "medium", timeStyle: "short" }).format(new Date(value));
}

function statusPill(value) {
  const status = String(value || "unknown").toLowerCase();
  return `<span class="table-status ${escapeHtml(status)}">${escapeHtml(status.replaceAll("_", " "))}</span>`;
}

function navigate(view) {
  state.activeView = view;
  $$(".nav-item").forEach((button) => button.classList.toggle("active", button.dataset.view === view));
  $$(".view").forEach((section) => section.classList.toggle("active-view", section.id === `view-${view}`));
  const titles = { overview: "Overview", knowledge: "Knowledge bases", agent: "Agent workspace", approvals: "Approvals", tickets: "Tickets", team: "Team" };
  $("#page-title").textContent = titles[view] || "Overview";
  if (view === "knowledge") loadKnowledgeBases();
  if (view === "approvals") loadApprovals();
  if (view === "tickets") loadTickets();
  if (view === "team") loadMembers();
}

function renderStats() {
  $("#stat-kbs").textContent = state.kbs.length;
  const pending = state.approvals.filter((item) => item.status === "pending").length;
  $("#stat-approvals").textContent = pending;
  $("#stat-tickets").textContent = state.tickets.filter((item) => item.status === "open").length;
  $("#stat-role").textContent = state.role;
  const badge = $("#approval-badge");
  badge.textContent = pending;
  badge.classList.toggle("hidden", pending === 0);
}

function renderActivity() {
  const items = [
    ...state.approvals.filter((item) => item.status === "pending").slice(0, 2).map((item) => ({ icon: "check", title: "Approval needs review", detail: `${item.tool_name} · ${formatDate(item.decided_at)}` })),
    ...state.tickets.slice(0, 2).map((item) => ({ icon: "ticket", title: item.title, detail: `${item.priority} priority · ${item.status}` })),
  ];
  $("#activity-list").innerHTML = items.length
    ? items.map((item) => `<div class="activity-item"><span class="activity-icon">${iconMarkup(item.icon)}</span><div><strong>${escapeHtml(item.title)}</strong><span>${escapeHtml(item.detail)}</span></div></div>`).join("")
    : '<p class="empty-state">Your latest approvals and tickets will appear here.</p>';
}

async function loadKnowledgeBases() {
  try {
    state.kbs = await api("/knowledge-bases");
    renderKnowledgeBases();
    renderKnowledgeChecks();
    renderStats();
  } catch (error) {
    showToast(error.message, "error");
  }
}

function renderKnowledgeBases() {
  const grid = $("#kb-grid");
  if (!state.kbs.length) {
    grid.innerHTML = '<div class="panel"><p class="eyebrow">No collections yet</p><h3>Start with your first runbook</h3><p class="muted">Create a knowledge base, then upload your source documents.</p><button class="button primary" id="empty-new-kb" type="button">Create knowledge base</button></div>';
    $("#empty-new-kb")?.addEventListener("click", openKbModal);
    return;
  }
  grid.innerHTML = state.kbs.map((kb) => `<article class="kb-card"><div><p class="eyebrow">Merchant knowledge</p><h4>${escapeHtml(kb.name)}</h4><p>${escapeHtml(kb.description || "No description provided.")}</p></div><div class="kb-card-footer"><span>Collection ready</span><button class="kb-open" type="button" data-kb="${escapeHtml(kb.id)}">Manage sources ${iconMarkup("arrow")}</button></div></article>`).join("");
  $$(".kb-open").forEach((button) => button.addEventListener("click", () => selectKb(button.dataset.kb)));
}

function selectKb(id) {
  state.selectedKb = state.kbs.find((kb) => kb.id === id) || null;
  if (!state.selectedKb) return;
  $("#kb-detail").classList.remove("hidden");
  $("#selected-kb-name").textContent = state.selectedKb.name;
  $("#selected-kb-description").textContent = state.selectedKb.description || "No description provided.";
  $("#document-status").textContent = "";
  $("#kb-detail").scrollIntoView({ behavior: "smooth", block: "start" });
}

function renderKnowledgeChecks() {
  const wrapper = $("#kb-checks");
  if (!state.kbs.length) {
    wrapper.innerHTML = '<span class="muted small-text">No knowledge base selected</span>';
    return;
  }
  wrapper.innerHTML = state.kbs.map((kb) => `<label><input type="checkbox" value="${escapeHtml(kb.id)}"${kb.name === "Merchant operations demo" ? " checked" : ""} /> ${escapeHtml(kb.name)}</label>`).join("");
}

async function loadApprovals() {
  try {
    state.approvals = await api("/approvals");
    renderApprovals();
    renderStats();
    renderActivity();
  } catch (error) {
    showToast(error.message, "error");
  }
}

function renderApprovals() {
  const wrapper = $("#approvals-table");
  if (!state.approvals.length) {
    wrapper.innerHTML = '<div class="panel"><p class="empty-state">No approval requests in this organization.</p></div>';
    return;
  }
  wrapper.innerHTML = `<table><thead><tr><th>Tool</th><th>Arguments</th><th>Status</th><th>Decided</th><th></th></tr></thead><tbody>${state.approvals.map((approval) => {
    const args = Object.entries(approval.arguments || {}).map(([key, value]) => `${key}: ${value}`).join(" · ");
    const actions = approval.status === "pending" && ["owner", "admin"].includes(state.role)
      ? `<button class="table-action" data-approval="${escapeHtml(approval.id)}" data-decision="approved">Approve</button><button class="table-action" data-approval="${escapeHtml(approval.id)}" data-decision="rejected">Reject</button>`
      : "—";
    return `<tr><td><strong>${escapeHtml(approval.tool_name)}</strong><br><span class="muted small-text">Run ${escapeHtml(approval.run_id.slice(0, 8))}</span></td><td>${escapeHtml(args || "—")}</td><td>${statusPill(approval.status)}</td><td>${escapeHtml(approval.decided_by || "Pending")}</td><td>${actions}</td></tr>`;
  }).join("")}</tbody></table>`;
  $$("[data-approval]").forEach((button) => button.addEventListener("click", () => decideApproval(button.dataset.approval, button.dataset.decision)));
}

async function decideApproval(id, decision) {
  try {
    await api(`/approvals/${encodeURIComponent(id)}/decision`, { method: "POST", body: JSON.stringify({ decision }) });
    showToast(decision === "approved" ? "Ticket created successfully." : "Approval rejected.");
    await Promise.all([loadApprovals(), loadTickets()]);
  } catch (error) {
    showToast(error.message, "error");
  }
}

async function loadTickets() {
  try {
    state.tickets = await api("/tickets");
    renderTickets();
    renderStats();
    renderActivity();
  } catch (error) {
    showToast(error.message, "error");
  }
}

function renderTickets() {
  const wrapper = $("#tickets-table");
  if (!state.tickets.length) {
    wrapper.innerHTML = '<div class="panel"><p class="empty-state">No tickets have been created yet.</p></div>';
    return;
  }
  wrapper.innerHTML = `<table><thead><tr><th>Ticket</th><th>Priority</th><th>Status</th><th>Created by</th><th>Created</th></tr></thead><tbody>${state.tickets.map((ticket) => `<tr><td><strong>${escapeHtml(ticket.title)}</strong><br><span class="muted small-text">${escapeHtml(ticket.description.slice(0, 100))}</span></td><td>${statusPill(ticket.priority)}</td><td>${statusPill(ticket.status)}</td><td>${escapeHtml(ticket.created_by || "Agent")}</td><td>${escapeHtml(formatDate(ticket.created_at))}</td></tr>`).join("")}</tbody></table>`;
}

async function loadMembers() {
  try {
    state.members = await api(`/organizations/${encodeURIComponent(state.orgId)}/members`);
    const current = state.members.find((member) => member.user_id === state.userId);
    if (current && current.role !== state.role) {
      state.role = current.role;
      saveIdentity();
      resetAgentChat();
      showWorkspace();
    }
    renderMembers();
    return true;
  } catch (error) {
    showToast(error.message, "error");
    return false;
  }
}

async function seedDemoKnowledge({ silent = false } = {}) {
  const button = $("#demo-seed-button");
  setLoading(button, true, "Preparing...");
  try {
    await api("/demo/seed", { method: "POST" });
    await loadKnowledgeBases();
    renderKnowledgeChecks();
    if (!silent) showToast("Demo runbooks are ready for grounded answers.");
    return true;
  } catch (error) {
    if (!silent) showToast(error.message, "error");
    return false;
  } finally {
    setLoading(button, false);
  }
}

function renderMembers() {
  const wrapper = $("#members-table");
  if (!state.members.length) {
    wrapper.innerHTML = '<div class="panel"><p class="empty-state">No members found.</p></div>';
    return;
  }
  wrapper.innerHTML = `<table><thead><tr><th>Member</th><th>Email</th><th>Role</th></tr></thead><tbody>${state.members.map((member) => `<tr><td><strong>${escapeHtml(member.display_name)}</strong></td><td>${escapeHtml(member.email)}</td><td>${statusPill(member.role)}</td></tr>`).join("")}</tbody></table>`;
}

async function loadAll() {
  if (!state.accessToken) {
    showAuth();
    return;
  }
  // Identity comes from the token, not storage, so the org and role are
  // resolved fresh on every load and a stale cache cannot grant access.
  try {
    const me = await api("/auth/me");
    state.userId = me.id;
  } catch (error) {
    showToast(error.message, "error");
    signOut({ silent: true });
    return;
  }

  // The organisation is re-resolved from the server on every load rather than
  // trusted from localStorage. A cached id goes stale whenever the tenant is
  // recreated (fresh database, restored dump), and a dangling id is not caught
  // by the API layer: it is well-formed, so every request carries it as
  // X-Org-ID and fails authorisation instead of refreshing the session.
  let orgs;
  try {
    orgs = await api("/organizations");
  } catch (error) {
    showToast(error.message, "error");
    signOut({ silent: true });
    return;
  }
  if (!orgs.length) {
    signOut({ silent: true });
    showToast("该账号不属于任何组织。", "error");
    return;
  }
  // Fall back to the first membership when the cached org is gone; keep the
  // user's current selection when it is still one of their memberships.
  const selected = orgs.find((org) => org.id === state.orgId) || orgs[0];
  // The role is taken from the same membership row, so it needs no second
  // request: /organizations already returns the caller's role in each
  // organisation. Resolving it here rather than letting loadMembers() correct
  // it afterwards means the workspace is never painted with a cached role —
  // a stale "owner" carried over from a previous tenant would otherwise show
  // member-management controls that the server is going to reject anyway.
  if (
    selected.id !== state.orgId ||
    selected.name !== state.orgName ||
    selected.role !== state.role
  ) {
    state.orgId = selected.id;
    state.orgName = selected.name;
    state.role = selected.role;
    saveIdentity();
    // Runs are never shared across users, organisations, or roles.
    resetAgentChat();
  }

  await loadMembers();
  showWorkspace();
  await Promise.all([loadKnowledgeBases(), loadApprovals(), loadTickets()]);
  if (!state.kbs.length && state.role !== "viewer") await seedDemoKnowledge({ silent: true });
  renderStats();
  renderActivity();
}

function addMessage(role, content, hint = "") {
  const wrapper = $("#chat-messages");
  const item = document.createElement("div");
  item.className = `message ${role}`;
  const body = role === "assistant" ? renderAssistantContent(content) : `<p>${escapeHtml(content)}</p>`;
  item.innerHTML = `<div class="message-avatar">${role === "user" ? escapeHtml(state.userId.slice(0, 2).toUpperCase()) : "AI"}</div><div><p class="message-meta">${role === "user" ? "You" : "Operations agent"}</p>${body}${hint ? `<span class="message-hint">${escapeHtml(hint)}</span>` : ""}</div>`;
  wrapper.appendChild(item);
  wrapper.scrollTop = wrapper.scrollHeight;
}

function bindPromptSuggestions() {
  $("#chat-messages")?.querySelectorAll("[data-prompt]").forEach((button) => button.addEventListener("click", () => {
    $("#agent-prompt").value = button.dataset.prompt;
    $("#agent-prompt").focus();
  }));
}

function renderRunDetails(run, events = []) {
  const latest = events[events.length - 1];
  const retrievalEvent = [...events].reverse().find((event) => event.type === "retrieval.completed");
  const retrievedCount = retrievalEvent?.count ?? latest?.count ?? "—";
  $("#run-details").innerHTML = `<div class="run-row"><span>Status</span><span>${statusPill(run.status)}</span></div><div class="run-row"><span>Trace ID</span><span title="${escapeHtml(run.trace_id)}">${escapeHtml(run.trace_id.slice(0, 12))}…</span></div><div class="run-row"><span>Retrieved</span><span>${escapeHtml(retrievedCount)} chunks</span></div><div class="run-row"><span>Approval</span><span>${escapeHtml(run.approval_id ? `${run.approval_id.slice(0, 8)}…` : "Not required")}</span></div><div class="run-row"><span>Estimated cost</span><span>$${Number(run.estimated_cost_usd || 0).toFixed(5)}</span></div>`;
}

function streamRun(run, contextKey, contextVersion) {
  if (!run?.id || !window.WebSocket) return;
  if (state.socket) state.socket.close();
  const protocol = location.protocol === "https:" ? "wss" : "ws";
  const query = new URLSearchParams({ run_id: run.id, org_id: state.orgId });
  // The token travels as the first Sec-WebSocket-Protocol value, which the
  // server prefers over the query parameter so it stays out of URLs and logs.
  const socket = new WebSocket(
    `${protocol}://${location.host}/v1/ws/agent?${query.toString()}`,
    [state.accessToken],
  );
  state.socket = socket;
  const events = [];
  socket.onmessage = (event) => {
    if (!isCurrentChatContext(contextKey, contextVersion)) {
      socket.close();
      return;
    }
    try {
      const data = JSON.parse(event.data);
      events.push(data);
      renderRunDetails(run, events);
      if (data.type === "run.completed") socket.close();
    } catch (_) { /* Ignore malformed event payloads. */ }
  };
}

async function runAgent(event) {
  event.preventDefault();
  const form = event.currentTarget;
  const prompt = $("#agent-prompt").value.trim();
  if (!prompt) return;
  const contextKey = workspaceContextKey();
  const contextVersion = state.chatContextVersion;
  const kbIds = $$("#kb-checks input:checked").map((input) => input.value);
  addMessage("user", prompt);
  $("#agent-prompt").value = "";
  const submit = form.querySelector("button[type=submit]");
  setLoading(submit, true, "Running...");
  try {
    const run = await api("/agent/runs", { method: "POST", body: JSON.stringify({ prompt, knowledge_base_ids: kbIds }) });
    if (!isCurrentChatContext(contextKey, contextVersion)) return;
    renderRunDetails(run);
    streamRun(run, contextKey, contextVersion);
    const hint = run.status === "waiting_approval" ? `Approval ${run.approval_id?.slice(0, 8) || "requested"} is waiting for an admin review.` : `Trace ${run.trace_id.slice(0, 12)}…`;
    addMessage("assistant", run.answer || "The run is in progress. Watch the run details for updates.", hint);
    await Promise.all([loadApprovals(), loadTickets()]);
  } catch (error) {
    if (!isCurrentChatContext(contextKey, contextVersion)) return;
    addMessage("assistant", `The run could not be completed: ${error.message}`);
    showToast(error.message, "error");
  } finally {
    if (isCurrentChatContext(contextKey, contextVersion)) setLoading(submit, false);
  }
}

function openKbModal() { $("#kb-modal").classList.remove("hidden"); }
function closeModals() { $$(".modal-backdrop").forEach((modal) => { if (modal.id !== "onboarding") modal.classList.add("hidden"); }); }

/** POST /v1/auth/login or /v1/auth/register, then resolve the workspace. */
async function submitAuth(event) {
  event.preventDefault();
  const form = event.currentTarget;
  const button = $("#auth-submit");
  const email = $("#auth-email").value.trim();
  const password = $("#auth-password").value;
  const registering = state.authMode === "register";

  const body = registering
    ? {
        email,
        password,
        display_name: $("#auth-name").value.trim() || email,
        organization_name: $("#auth-org").value.trim(),
      }
    : { email, password };

  // Bare paths, like every caller of api(): the /v1 prefix is added at fetch time.
  const endpoint = registering ? "/auth/register" : "/auth/login";
  setLoading(button, true, registering ? "Creating..." : "Signing in...");
  try {
    const response = await fetch(`/v1${endpoint}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const payload = await response.json().catch(() => null);
    if (!response.ok) {
      throw new Error(errorMessage(payload, response.status));
    }
    storeTokens(payload.access_token, payload.refresh_token);
    form.reset();
    await loadAll();
    showToast(registering ? "Workspace created." : "Signed in.");
  } catch (error) {
    showToast(error.message, "error");
  } finally {
    setLoading(button, false);
  }
}

async function createKnowledgeBase(event) {
  event.preventDefault();
  const form = event.currentTarget;
  const data = new FormData(form);
  const button = form.querySelector("button");
  setLoading(button, true, "Creating...");
  try {
    await api("/knowledge-bases", { method: "POST", body: JSON.stringify({ name: data.get("name"), description: data.get("description") || "" }) });
    form.reset(); closeModals(); await loadKnowledgeBases(); showToast("Knowledge base created.");
  } catch (error) { showToast(error.message, "error"); }
  finally { setLoading(button, false); }
}

async function uploadDocument(event) {
  const file = event.target.files?.[0];
  if (!file || !state.selectedKb) return;
  const status = $("#document-status");
  status.textContent = `Uploading ${file.name}...`;
  const body = new FormData(); body.append("file", file);
  try {
    const result = await api(`/documents?kb_id=${encodeURIComponent(state.selectedKb.id)}`, { method: "POST", body, headers: {} });
    status.textContent = result.queued ? `${file.name} queued for indexing.` : `${file.name} indexed.`;
    showToast("Document ingestion started.");
  } catch (error) { status.textContent = error.message; showToast(error.message, "error"); }
  event.target.value = "";
}

async function ingestUrl(event) {
  event.preventDefault();
  const form = event.currentTarget; const button = form.querySelector("button"); const url = new FormData(form).get("url");
  setLoading(button, true, "Indexing...");
  try {
    await api("/documents/from-url", { method: "POST", body: JSON.stringify({ kb_id: state.selectedKb.id, url }) });
    closeModals(); form.reset(); $("#document-status").textContent = "URL queued for indexing."; showToast("URL ingestion started.");
  } catch (error) { showToast(error.message, "error"); }
  finally { setLoading(button, false); }
}

async function addMember(event) {
  event.preventDefault();
  const form = event.currentTarget; const data = new FormData(form); const button = form.querySelector("button");
  setLoading(button, true, "Saving...");
  try {
    await api(`/organizations/${encodeURIComponent(state.orgId)}/members`, { method: "POST", body: JSON.stringify({ email: data.get("email"), display_name: data.get("display_name"), password: data.get("password"), role: data.get("role") }) });
    form.reset(); await loadMembers(); showToast("Teammate saved.");
  } catch (error) { showToast(error.message, "error"); }
  finally { setLoading(button, false); }
}

function wireEvents() {
  $$("[data-view]").forEach((button) => button.addEventListener("click", () => navigate(button.dataset.view)));
  $$("[data-go]").forEach((button) => button.addEventListener("click", () => navigate(button.dataset.go)));
  $$("[data-refresh]").forEach((button) => button.addEventListener("click", () => ({ approvals: loadApprovals, tickets: loadTickets }[button.dataset.refresh])()));
  $("#auth-form").addEventListener("submit", submitAuth);
  $("#auth-toggle").addEventListener("click", () => setAuthMode(state.authMode === "register" ? "signin" : "register"));
  $("#new-kb-button").addEventListener("click", openKbModal);
  $("#kb-form").addEventListener("submit", createKnowledgeBase);
  $("#document-file").addEventListener("change", uploadDocument);
  $("#url-ingest-button").addEventListener("click", () => $("#url-modal").classList.remove("hidden"));
  $("#url-form").addEventListener("submit", ingestUrl);
  $("#agent-form").addEventListener("submit", runAgent);
  $("#demo-seed-button").addEventListener("click", () => seedDemoKnowledge());
  bindPromptSuggestions();
  $("#member-form").addEventListener("submit", addMember);
  $("#close-kb-detail").addEventListener("click", () => $("#kb-detail").classList.add("hidden"));
  $("#refresh-all").addEventListener("click", () => loadAll());
  $("#disconnect").addEventListener("click", () => signOut());
  $$(".close-modal").forEach((button) => button.addEventListener("click", closeModals));
  $$(".modal-backdrop").forEach((modal) => modal.addEventListener("click", (event) => { if (event.target === modal && modal.id !== "onboarding") closeModals(); }));
}

async function boot() {
  wireEvents();
  setAuthMode("signin");
  if (!state.accessToken) { showAuth(); return; }
  await loadAll();
}

boot();
