// Chat column: message log, composer, and the references panel rendering.

import { state, on, emit } from "../store.js";
import { api } from "../api.js";
import { byId, el, clear, icon } from "../dom.js";
import { renderMarkdown } from "../markdown.js";
import { createSession, reloadSessions } from "./sessions.js";

let els = {};

export function mountChat() {
  els = {
    chatLog: byId("chatLog"),
    composer: byId("composer"),
    prompt: byId("prompt"),
    sendButton: byId("sendButton"),
    referenceList: byId("referenceList"),
  };

  els.composer.addEventListener("submit", onSubmit);
  els.prompt.addEventListener("input", autoGrow);
  els.prompt.addEventListener("keydown", (event) => {
    if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
      event.preventDefault();
      els.composer.requestSubmit();
    }
  });

  on("messages", renderMessages);
  on("references", renderReferences);
  on("focus-prompt", () => els.prompt.focus());
}

function onSubmit(event) {
  event.preventDefault();
  const question = els.prompt.value.trim();
  if (!question || state.busy) return;
  els.prompt.value = "";
  autoGrow();
  ask(question);
}

function autoGrow() {
  els.prompt.style.height = "auto";
  els.prompt.style.height = `${Math.min(els.prompt.scrollHeight, 200)}px`;
}

function setBusy(busy) {
  state.busy = busy;
  els.sendButton.disabled = busy;
  els.sendButton.classList.toggle("busy", busy);
}

async function ask(question) {
  if (!state.activeSessionId) {
    await createSession();
    if (!state.activeSessionId) return;
  }

  const assistant = {
    role: "assistant",
    content: "",
    contexts: [],
    usedRag: false,
    route: "",
    routeReason: "",
    loading: true,
  };
  state.messages.push({ role: "user", content: question }, assistant);
  emit("messages");
  state.lastReferences = { contexts: [], usedRag: true };
  emit("references");
  setBusy(true);

  try {
    const data = await api.ask({
      question,
      session_id: state.activeSessionId,
      recall_top_k: state.recallTopK ?? null,
      top_k: state.topK ?? null,
    });

    if (data.session_id) state.activeSessionId = data.session_id;
    assistant.content = data.answer;
    assistant.contexts = data.contexts || [];
    assistant.usedRag = Boolean(data.used_rag);
    assistant.route = data.route || "";
    assistant.routeReason = data.route_reason || "";
    assistant.loading = false;

    state.conversationSummary = data.conversation_summary || state.conversationSummary;
    state.summarizedMessageCount += data.compacted_history_messages || 0;
    state.latestRoute = `${assistant.usedRag ? "RAG" : "Direct"} · ${assistant.route || "unknown"}`;
    state.lastReferences = { contexts: assistant.contexts, usedRag: assistant.usedRag };

    emit("references");
    emit("meta");
    await reloadSessions();
  } catch (error) {
    assistant.content = error.message;
    assistant.loading = false;
    assistant.error = true;
  } finally {
    setBusy(false);
    emit("messages");
  }
}

// ----------------------------- rendering -----------------------------

function renderMessages() {
  clear(els.chatLog);

  if (!state.messages.length) {
    els.chatLog.append(renderEmptyState());
    return;
  }
  for (const message of state.messages) {
    els.chatLog.append(renderMessage(message));
  }
  els.chatLog.scrollTop = els.chatLog.scrollHeight;
}

function renderEmptyState() {
  return el("div", { class: "empty-state" }, [
    el("span", { class: "brand-mark" }, [icon("sparkle")]),
    el("h2", { text: "Ask your knowledge base" }),
    el("p", {
      text:
        "Recent turns are kept, older context is compacted into memory, and each " +
        "answer retrieves Markdown references when retrieval is used.",
    }),
  ]);
}

function renderMessage(message) {
  const wrapper = el("article", { class: `message ${message.role}` });
  wrapper.append(
    el("div", { class: "msg-role", text: message.role === "user" ? "You" : "Assistant" }),
  );

  const bubble = el("div", { class: "bubble" });
  if (message.loading) {
    bubble.classList.add("loading");
    bubble.append(el("span", { text: "Thinking" }), el("span", { class: "dots" }));
  } else if (message.error) {
    bubble.classList.add("error");
    bubble.textContent = message.content;
  } else {
    bubble.classList.add("markdown");
    bubble.innerHTML = renderMarkdown(message.content);
  }
  wrapper.append(bubble);

  if (message.role === "assistant" && !message.loading && !message.error) {
    wrapper.append(renderRouteMeta(message));
    if (message.contexts?.length) {
      wrapper.append(renderInlineRefs(message.contexts));
    }
  }
  return wrapper;
}

function renderRouteMeta(message) {
  const meta = el("div", { class: "msg-route" });
  meta.append(
    el("span", {
      class: `route-pill${message.usedRag ? " rag" : ""}`,
      text: `${message.usedRag ? "RAG" : "Direct"} · ${message.route || "unknown"}`,
    }),
  );
  if (message.routeReason) {
    meta.append(el("span", { text: message.routeReason }));
  }
  return meta;
}

function renderInlineRefs(contexts) {
  const details = el("details", { class: "inline-refs", open: true });
  details.append(el("summary", {}, [`References (${contexts.length})`]));
  for (const context of contexts) {
    details.append(renderReferenceCard(context, 420));
  }
  return el("div", { class: "msg-refs" }, [details]);
}

function renderReferences() {
  clear(els.referenceList);
  const { contexts, usedRag } = state.lastReferences || { contexts: [], usedRag: true };

  if (!usedRag) {
    els.referenceList.append(
      el("div", { class: "empty", text: "No retrieval was used for this answer." }),
    );
    return;
  }
  if (!contexts.length) {
    els.referenceList.append(
      el("div", { class: "empty", text: "References from retrieval will appear here." }),
    );
    return;
  }
  for (const context of contexts) {
    els.referenceList.append(renderReferenceCard(context, 800));
  }
}

function renderReferenceCard(item, maxLength) {
  const left = el("div", {}, [el("div", { class: "ref-source", text: item.source })]);
  const sub = [`chunk ${item.chunk_id}`, item.content_type || "text"];
  const headings =
    Array.isArray(item.headings) && item.headings.length ? item.headings.join(" › ") : "";
  if (headings) sub.push(headings);
  left.append(el("div", { class: "ref-sub", text: sub.join(" · ") }));

  const hasRerank = item.rerank_score !== null && item.rerank_score !== undefined;
  const score = el("div", {
    class: "ref-score",
    text: hasRerank
      ? `rerank ${Number(item.rerank_score).toFixed(3)}`
      : `vector ${Number(item.score).toFixed(3)}`,
    title: hasRerank ? `vector score ${Number(item.score).toFixed(4)}` : "",
  });

  const meta = el("div", { class: "ref-meta" }, [left, score]);
  const body = item.text || "";
  const text = el("pre", {
    class: "ref-text",
    text: body.length > maxLength ? `${body.slice(0, maxLength)}…` : body,
  });
  return el("div", { class: "ref-card" }, [meta, text]);
}
