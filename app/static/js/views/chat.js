// Chat column: message log, composer, and the references panel rendering.

import { state, on, emit } from "../store.js?v=20260620-degraded-retrieval";
import { api } from "../api.js?v=20260620-degraded-retrieval";
import { byId, el, clear, icon } from "../dom.js?v=20260620-degraded-retrieval";
import { renderMarkdown } from "../markdown.js?v=20260620-degraded-retrieval";
import { createSession, reloadSessions } from "./sessions.js?v=20260620-degraded-retrieval";

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
    retrievalDegraded: false,
    qdrantDegraded: false,
    rerankerDegraded: false,
    degradationReason: "",
    loading: true,
  };
  state.messages.push({ role: "user", content: question }, assistant);
  emit("messages");
  state.lastReferences = {
    contexts: [],
    usedRag: true,
    retrievalDegraded: false,
    degradationReason: "",
  };
  emit("references");
  setBusy(true);

  try {
    const data = await api.ask({
      question,
      session_id: state.activeSessionId,
      bm25_top_k: state.bm25TopK ?? null,
      recall_top_k: state.recallTopK ?? null,
      rrf_top_k: state.rrfTopK ?? null,
      top_k: state.topK ?? null,
    });

    if (data.session_id) state.activeSessionId = data.session_id;
    assistant.content = data.answer;
    assistant.contexts = data.contexts || [];
    assistant.usedRag = Boolean(data.used_rag);
    assistant.route = data.route || "";
    assistant.routeReason = data.route_reason || "";
    assistant.retrievalDegraded = Boolean(data.retrieval_degraded);
    assistant.qdrantDegraded = Boolean(data.qdrant_degraded);
    assistant.rerankerDegraded = Boolean(data.reranker_degraded);
    assistant.degradationReason = data.degradation_reason || "";
    assistant.loading = false;

    state.conversationSummary = data.conversation_summary || state.conversationSummary;
    state.summarizedMessageCount += data.compacted_history_messages || 0;
    state.latestRoute = `${assistant.usedRag ? "RAG" : "Direct"} · ${assistant.route || "unknown"}${
      assistant.retrievalDegraded ? " · Degraded" : ""
    }`;
    state.lastReferences = {
      contexts: assistant.contexts,
      usedRag: assistant.usedRag,
      retrievalDegraded: assistant.retrievalDegraded,
      degradationReason: assistant.degradationReason,
    };

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
    if (message.retrievalDegraded) {
      wrapper.append(renderDegradationNotice(message));
    }
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
  if (message.retrievalDegraded) {
    meta.append(el("span", { class: "route-pill warn", text: "Degraded" }));
  }
  return meta;
}

function renderDegradationNotice(message) {
  return el("div", {
    class: "degradation-notice",
    text: degradationText(message),
  });
}

function degradationText(item) {
  if (item.degradationReason) return item.degradationReason;
  if (item.qdrantDegraded && item.rerankerDegraded) {
    return "Qdrant/vector recall and reranker degraded; using BM25-only unre-ranked references.";
  }
  if (item.qdrantDegraded) {
    return "Qdrant/vector recall degraded; using BM25-only references.";
  }
  if (item.rerankerDegraded) {
    return "Reranker degraded; using unre-ranked RRF references.";
  }
  return "Retrieval degraded; using fallback references.";
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
  const fallback = {
    contexts: [],
    usedRag: true,
    retrievalDegraded: false,
    degradationReason: "",
  };
  const { contexts, usedRag, retrievalDegraded, degradationReason } =
    state.lastReferences || fallback;

  if (!usedRag) {
    els.referenceList.append(
      el("div", { class: "empty", text: "No retrieval was used for this answer." }),
    );
    return;
  }
  if (retrievalDegraded) {
    els.referenceList.append(
      el("div", {
        class: "degradation-notice references",
        text: degradationReason || "Retrieval degraded; using fallback references.",
      }),
    );
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
  const sourceLabel = item.retrieval_source || "retrieval";
  const sub = [`chunk ${item.chunk_id}`, item.content_type || "text", sourceLabel];
  const headings =
    Array.isArray(item.headings) && item.headings.length ? item.headings.join(" › ") : "";
  if (headings) sub.push(headings);
  left.append(el("div", { class: "ref-sub", text: sub.join(" · ") }));

  const hasRerank = item.rerank_score !== null && item.rerank_score !== undefined;
  const scoreParts = [
    item.rrf_score !== null && item.rrf_score !== undefined
      ? `rrf ${Number(item.rrf_score).toFixed(4)}`
      : "",
    item.vector_score !== null && item.vector_score !== undefined
      ? `vector ${Number(item.vector_score).toFixed(4)}`
      : "",
    item.bm25_score !== null && item.bm25_score !== undefined
      ? `bm25 ${Number(item.bm25_score).toFixed(3)}`
      : "",
  ].filter(Boolean);
  const score = el("div", {
    class: "ref-score",
    text: hasRerank
      ? `rerank ${Number(item.rerank_score).toFixed(3)}`
      : scoreParts[0] || `score ${Number(item.score).toFixed(3)}`,
    title: scoreParts.join(" · "),
  });

  const meta = el("div", { class: "ref-meta" }, [left, score]);
  const body = item.text || "";
  const text = el("pre", {
    class: "ref-text",
    text: body.length > maxLength ? `${body.slice(0, maxLength)}…` : body,
  });
  return el("div", { class: "ref-card" }, [meta, text]);
}
