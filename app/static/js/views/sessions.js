// Sidebar session list: search/filter, select, inline rename, and delete.
// Also owns the session data actions used by the chat view.

import { state, on, emit } from "../store.js";
import { api } from "../api.js";
import { byId, el, clear, icon } from "../dom.js";

let els = {};

export function mountSessions() {
  els = {
    newSessionButton: byId("newSessionButton"),
    sessionSearch: byId("sessionSearch"),
    sessionList: byId("sessionList"),
    sessionTitle: byId("sessionTitle"),
  };

  els.newSessionButton.addEventListener("click", () => createSession());
  els.sessionSearch.addEventListener("input", () => {
    state.sessionFilter = els.sessionSearch.value;
    renderSessions();
  });

  on("sessions", renderSessions);
}

// ----------------------------- data actions -----------------------------

export async function loadSessions(selectFirst = false) {
  if (!state.currentUser) return;
  try {
    state.sessions = await api.listSessions();
  } catch {
    return; // 401/403 already surfaced via events
  }
  emit("sessions");

  if (!state.sessions.length) {
    await createSession();
    return;
  }
  const activeExists = state.sessions.some((s) => s.id === state.activeSessionId);
  if (selectFirst || !activeExists) {
    await selectSession(state.sessions[0].id, true);
  }
}

export async function reloadSessions() {
  if (!state.currentUser) return;
  try {
    state.sessions = await api.listSessions();
    emit("sessions");
  } catch {
    /* ignore */
  }
}

export async function createSession() {
  if (!state.currentUser) return;
  let session;
  try {
    session = await api.createSession();
  } catch {
    return;
  }
  applySessionDetail(session);
  await reloadSessions();
}

export async function selectSession(id, force = false) {
  if (!id) return;
  if (id === state.activeSessionId && !force) {
    emit("sessions");
    return;
  }
  let session;
  try {
    session = await api.getSession(id);
  } catch {
    await loadSessions(true);
    return;
  }
  applySessionDetail(session);
}

function applySessionDetail(session) {
  state.activeSessionId = session.id;
  state.renamingSessionId = "";
  state.conversationSummary = session.conversation_summary || "";
  state.summarizedMessageCount = session.compacted_message_count || 0;
  state.messages = (session.messages || []).map((m) => ({
    role: m.role,
    content: m.content,
    contexts: m.contexts || [],
    usedRag: Boolean(m.used_rag),
    route: m.route || "",
    routeReason: m.route_reason || "",
  }));

  const lastAssistant = [...state.messages].reverse().find((m) => m.role === "assistant");
  state.latestRoute = lastAssistant
    ? `${lastAssistant.usedRag ? "RAG" : "Direct"} · ${lastAssistant.route || "unknown"}`
    : "None";
  state.lastReferences = lastAssistant
    ? { contexts: lastAssistant.contexts || [], usedRag: lastAssistant.usedRag }
    : { contexts: [], usedRag: true };

  emit("sessions");
  emit("messages");
  emit("references");
  emit("meta");
  emit("session-selected");
  emit("focus-prompt");
}

async function doRename(id, title) {
  try {
    const updated = await api.renameSession(id, title);
    state.sessions = state.sessions.map((s) =>
      s.id === id ? { ...s, title: updated.title } : s,
    );
  } catch {
    /* leave the old title in place */
  }
  emit("sessions");
}

async function doDelete(session) {
  const label = session.title || "this chat";
  if (!window.confirm(`Delete "${label}"? This cannot be undone.`)) return;

  const response = await api.deleteSession(session.id);
  if (!response.ok && response.status !== 404) return;

  const wasActive = session.id === state.activeSessionId;
  state.sessions = state.sessions.filter((s) => s.id !== session.id);

  if (wasActive) {
    state.activeSessionId = "";
    if (state.sessions.length) {
      await selectSession(state.sessions[0].id, true);
    } else {
      await createSession();
    }
  } else {
    emit("sessions");
  }
}

// ----------------------------- rendering -----------------------------

function renderSessions() {
  const active = state.sessions.find((s) => s.id === state.activeSessionId);
  els.sessionTitle.textContent = active ? active.title : "Conversation";

  clear(els.sessionList);

  if (!state.sessions.length) {
    els.sessionList.append(el("div", { class: "empty", text: "No conversations yet." }));
    return;
  }

  const filter = state.sessionFilter.trim().toLowerCase();
  const visible = filter
    ? state.sessions.filter((s) => (s.title || "").toLowerCase().includes(filter))
    : state.sessions;

  if (!visible.length) {
    els.sessionList.append(el("div", { class: "empty", text: "No matching chats." }));
    return;
  }

  for (const session of visible) {
    els.sessionList.append(renderSessionItem(session));
  }
}

function renderSessionItem(session) {
  const isActive = session.id === state.activeSessionId;

  if (state.renamingSessionId === session.id) {
    const input = el("input", {
      class: "session-rename",
      value: session.title,
      maxlength: "80",
      "aria-label": "Rename chat",
    });
    const commit = async () => {
      if (state.renamingSessionId !== session.id) return;
      const title = input.value.trim();
      state.renamingSessionId = "";
      if (title && title !== session.title) {
        await doRename(session.id, title);
      } else {
        emit("sessions");
      }
    };
    input.addEventListener("keydown", (event) => {
      if (event.key === "Enter") {
        event.preventDefault();
        input.blur();
      } else if (event.key === "Escape") {
        state.renamingSessionId = "";
        emit("sessions");
      }
    });
    input.addEventListener("blur", commit);
    const item = el("div", { class: `session-item${isActive ? " active" : ""}` }, [input]);
    requestAnimationFrame(() => {
      input.focus();
      input.select();
    });
    return item;
  }

  const title = el("span", { class: "session-title", text: session.title || "New chat" });
  const renameBtn = el(
    "button",
    {
      class: "session-act",
      type: "button",
      title: "Rename",
      "aria-label": "Rename chat",
      onClick: (event) => {
        event.stopPropagation();
        state.renamingSessionId = session.id;
        emit("sessions");
      },
    },
    [icon("edit")],
  );
  const deleteBtn = el(
    "button",
    {
      class: "session-act danger",
      type: "button",
      title: "Delete",
      "aria-label": "Delete chat",
      onClick: (event) => {
        event.stopPropagation();
        doDelete(session);
      },
    },
    [icon("trash")],
  );
  const actions = el("div", { class: "session-actions" }, [renameBtn, deleteBtn]);

  const item = el(
    "div",
    {
      class: `session-item${isActive ? " active" : ""}`,
      role: "button",
      tabindex: "0",
      onClick: () => selectSession(session.id),
    },
    [title, actions],
  );
  item.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      selectSession(session.id);
    }
  });
  return item;
}
