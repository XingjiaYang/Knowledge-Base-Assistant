// Chat column: message log, composer, and the references panel rendering.

import { state, on, emit } from "../store.js?v=20260626-code-search";
import { api } from "../api.js?v=20260626-code-search";
import { byId, el, clear, icon } from "../dom.js?v=20260626-code-search";
import { renderMarkdown } from "../markdown.js?v=20260626-code-search";
import { createSession, reloadSessions } from "./sessions.js?v=20260626-code-search";

let els = {};
let codeCy = null;

export function mountChat() {
  els = {
    chatLog: byId("chatLog"),
    composer: byId("composer"),
    prompt: byId("prompt"),
    sendButton: byId("sendButton"),
    referenceList: byId("referenceList"),
    sidePanelTitle: byId("sidePanelTitle"),
    chatModeButton: byId("chatModeButton"),
    codeModeButton: byId("codeModeButton"),
  };

  els.composer.addEventListener("submit", onSubmit);
  els.chatModeButton.addEventListener("click", () => setMode("chat"));
  els.codeModeButton.addEventListener("click", () => setMode("code"));
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
  renderMode();
}

function onSubmit(event) {
  event.preventDefault();
  const question = els.prompt.value.trim();
  if (!question || state.busy) return;
  els.prompt.value = "";
  autoGrow();
  if (state.activeMode === "code") {
    searchCode(question);
  } else {
    ask(question);
  }
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

function setMode(mode) {
  if (state.activeMode === mode) return;
  state.activeMode = mode;
  if (mode === "code") ensureCodeRepositories();
  renderMode();
  emit("messages");
  emit("references");
  els.prompt.focus();
}

function renderMode() {
  const codeMode = state.activeMode === "code";
  els.chatModeButton.classList.toggle("active", !codeMode);
  els.codeModeButton.classList.toggle("active", codeMode);
  els.prompt.placeholder = codeMode
    ? `Search ${selectedCodeRepositoryLabel()} code symbols...`
    : "Ask a question about the indexed knowledge base…";
  if (els.sidePanelTitle) {
    els.sidePanelTitle.innerHTML = codeMode
      ? '<svg class="icon"><use href="#i-graph"/></svg> Call Graph'
      : '<svg class="icon"><use href="#i-book"/></svg> References';
  }
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
    embeddingDegraded: false,
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
      rag_only: Boolean(state.ragOnly),
    });

    if (data.session_id) state.activeSessionId = data.session_id;
    assistant.content = data.answer;
    assistant.contexts = data.contexts || [];
    assistant.usedRag = Boolean(data.used_rag);
    assistant.route = data.route || "";
    assistant.routeReason = data.route_reason || "";
    assistant.retrievalDegraded = Boolean(data.retrieval_degraded);
    assistant.embeddingDegraded = Boolean(data.embedding_degraded);
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

async function searchCode(query) {
  if (!state.activeSessionId) {
    await createSession();
    if (!state.activeSessionId) return;
  }
  await ensureCodeRepositories();
  state.codeQuery = query;
  state.codeResults = [];
  state.codeFiles = [];
  state.codeGraph = null;
  state.codeGraphFunction = "";
  state.codeError = "";
  state.codeSearchLoading = true;
  const assistant = {
    role: "assistant",
    content: "",
    contexts: [],
    usedRag: true,
    route: "code_search",
    routeReason: "Code search over selected repositories.",
    retrievalDegraded: false,
    qdrantDegraded: false,
    rerankerDegraded: false,
    degradationReason: "",
    loading: true,
    codeResults: [],
    codeFiles: [],
    codeQuery: query,
  };
  state.messages.push({ role: "user", content: query }, assistant);
  setBusy(true);
  emit("messages");
  emit("references");

  try {
    const data = await api.codeSearch({
      query,
      session_id: state.activeSessionId,
      final_top_k: state.topK ?? null,
      repository_ids: selectedCodeRepositoryIds(),
    });
    if (data.session_id) state.activeSessionId = data.session_id;
    state.codeFiles = data.files || [];
    state.codeResults = data.functions || [];
    state.codeGraph = data.graph || null;
    state.codeGraphFunction = data.graph?.function_name || query;
    assistant.content = data.answer || formatCodeAnswer(state.codeResults);
    assistant.contexts = data.contexts || [];
    assistant.codeFiles = state.codeFiles;
    assistant.codeResults = state.codeResults;
    assistant.loading = false;
    state.lastReferences = {
      contexts: assistant.contexts,
      usedRag: true,
      retrievalDegraded: false,
      degradationReason: "",
    };
    state.latestRoute = "Code · code_search";
    emit("references");
    emit("meta");
    await reloadSessions();
  } catch (error) {
    state.codeError = error.message;
    assistant.content = error.message;
    assistant.loading = false;
    assistant.error = true;
  } finally {
    state.codeSearchLoading = false;
    setBusy(false);
    emit("messages");
  }
}

function formatCodeAnswer(results) {
  if (!results?.length) return "No matching code functions found.";
  return [
    "Found these code entry points:",
    ...results.slice(0, 10).map((item, index) =>
      `${index + 1}. \`${item.qualified_name || item.name}\` in ` +
      `\`${item.path}:${item.start_line}\``,
    ),
  ].join("\n");
}

async function loadCallGraph(functionName, repositoryId = state.selectedCodeRepositoryId) {
  if (!functionName || state.busy) return;
  state.codeGraphFunction = functionName;
  state.codeGraph = { loading: true, callers: [], nodes: [], edges: [] };
  emit("references");

  try {
    const data = await api.codeCallGraph(
      functionName,
      3,
      repositoryId ? [repositoryId] : selectedCodeRepositoryIds(),
    );
    state.codeGraph = data;
  } catch (error) {
    state.codeGraph = {
      function_name: functionName,
      error: error.message,
      callers: [],
      nodes: [],
      edges: [],
    };
  } finally {
    emit("references");
  }
}

async function ensureCodeRepositories() {
  if (state.codeRepositoriesLoaded || state.codeRepositoriesLoading || !state.currentUser) {
    return;
  }
  state.codeRepositoriesLoading = true;
  emit("messages");
  try {
    const data = await api.codeRepositories();
    state.codeRepositories = data.repositories || [];
    state.codeRepositoriesLoaded = true;
    state.codeError = "";
    const ids = new Set(state.codeRepositories.map((repository) => repository.id));
    if (state.selectedCodeRepositoryId && !ids.has(state.selectedCodeRepositoryId)) {
      state.selectedCodeRepositoryId = "";
    }
    if (!state.selectedCodeRepositoryId && data.default_repository_ids?.length === 1) {
      state.selectedCodeRepositoryId = data.default_repository_ids[0];
    }
  } catch (error) {
    state.codeError = error.message;
  } finally {
    state.codeRepositoriesLoading = false;
    renderMode();
    emit("messages");
  }
}

async function indexSelectedCodeRepositories() {
  await ensureCodeRepositories();
  if (state.codeIndexing || state.busy) return;

  state.codeIndexing = true;
  state.codeIndexStatus = "Indexing code...";
  state.codeError = "";
  setBusy(true);
  emit("messages");

  try {
    const data = await api.codeIndex({
      repository_ids: selectedCodeRepositoryIds(),
      rebuild: true,
    });
    state.codeRepositories = data.repositories || state.codeRepositories;
    state.codeRepositoriesLoaded = true;
    state.codeIndexStatus =
      `Indexed ${data.files || 0} files, ` +
      `${data.functions || 0} symbols, ${data.call_edges || 0} calls.`;
    if (state.codeQuery) {
      await searchCode(state.codeQuery);
    }
  } catch (error) {
    state.codeError = error.message;
    state.codeIndexStatus = "";
  } finally {
    state.codeIndexing = false;
    setBusy(false);
    emit("messages");
  }
}

function selectedCodeRepositoryIds() {
  return state.selectedCodeRepositoryId ? [state.selectedCodeRepositoryId] : [];
}

function selectedCodeRepositoryLabel() {
  const selected = (state.codeRepositories || []).find(
    (repository) => repository.id === state.selectedCodeRepositoryId,
  );
  if (selected) return selected.name;
  if ((state.codeRepositories || []).length === 1) return state.codeRepositories[0].name;
  return "code folders";
}

// ----------------------------- rendering -----------------------------

function renderMessages() {
  clear(els.chatLog);

  if (state.activeMode === "code") {
    renderCodeSearch();
    return;
  }

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

function renderCodeSearch() {
  if (!state.messages.length && !state.codeQuery && !state.codeSearchLoading && !state.codeError) {
    els.chatLog.append(
      el("div", { class: "empty-state" }, [
        el("span", { class: "brand-mark" }, [icon("code")]),
        el("h2", { text: `Search ${selectedCodeRepositoryLabel()} code` }),
        renderCodeRepositoryPicker(),
        el("p", { text: "Function-level results and call graphs appear here." }),
      ]),
    );
    return;
  }

  const shell = el("div", { class: "code-search-shell" });
  shell.append(renderCodeRepositoryPicker(true));

  for (const message of state.messages) {
    shell.append(renderCodeMessage(message));
  }

  if (state.codeError && !state.messages.some((message) => message.error)) {
    shell.append(el("div", { class: "bubble error", text: state.codeError }));
  }
  els.chatLog.append(shell);
  els.chatLog.scrollTop = els.chatLog.scrollHeight;
}

function renderCodeMessage(message) {
  const node = renderMessage(message);
  if (message.role !== "assistant" || message.loading || message.error) return node;

  if (message.codeFiles?.length) {
    node.append(renderCodeFileHits(message.codeFiles));
  }
  if (message.codeResults?.length) {
    const list = el("div", { class: "code-result-list" });
    for (const item of message.codeResults) {
      list.append(renderCodeResultCard(item));
    }
    node.append(list);
  }
  return node;
}

function renderCodeRepositoryPicker(compact = false) {
  const repos = state.codeRepositories || [];
  const selected = state.selectedCodeRepositoryId
    ? repos.find((repository) => repository.id === state.selectedCodeRepositoryId)
    : null;
  const indexedCount = selected
    ? selected.indexed_files || 0
    : repos.reduce((total, repository) => total + (repository.indexed_files || 0), 0);
  const buttonLabel = indexedCount > 0 ? "Reindex" : "Index";
  const select = el(
    "select",
    {
      class: compact ? "code-repo-select compact" : "code-repo-select",
      disabled: state.codeRepositoriesLoading || !repos.length,
      onChange: (event) => {
        state.selectedCodeRepositoryId = event.target.value;
        state.codeGraph = null;
        state.codeGraphFunction = "";
        renderMode();
        if (state.codeQuery) {
          searchCode(state.codeQuery);
        } else {
          emit("messages");
          emit("references");
        }
      },
    },
    [
      el("option", {
        value: "",
        text: repos.length ? "All code folders" : "No code folders",
        selected: !state.selectedCodeRepositoryId,
      }),
      ...repos.map((repository) =>
        el("option", {
          value: repository.id,
          text: `${repository.name} (${repository.indexed_files || 0})`,
          title: repository.source_root,
          selected: repository.id === state.selectedCodeRepositoryId,
        }),
      ),
    ],
  );
  select.value = state.selectedCodeRepositoryId || "";
  return el("div", { class: "code-repo-picker" }, [
    el("label", { class: "code-repo-field" }, [
      el("span", { class: "field-label", text: "Code folder" }),
      select,
    ]),
    el("div", { class: "code-index-row" }, [
      el(
        "button",
        {
          class: "btn btn-ghost btn-sm",
          type: "button",
          disabled: state.codeIndexing || state.busy || !repos.length,
          onClick: indexSelectedCodeRepositories,
        },
        [icon("refresh"), state.codeIndexing ? "Indexing" : buttonLabel],
      ),
      el("span", {
        class: "code-index-status",
        text: state.codeIndexStatus || `${indexedCount} indexed files`,
      }),
    ]),
  ]);
}

function renderCodeFileHits(files) {
  const wrap = el("div", { class: "code-file-hits" });
  for (const file of files.slice(0, 6)) {
    const label = [file.repository_name || file.repository_id, file.path]
      .filter(Boolean)
      .join(" / ");
    wrap.append(
      el("span", {
        class: "code-file-chip",
        title: [file.source_root, file.path].filter(Boolean).join("/"),
        text: `${label} · ${Number(file.score).toFixed(3)}`,
      }),
    );
  }
  return wrap;
}

function renderCodeResultCard(item) {
  const graphButton = el(
    "button",
    {
      class: "btn btn-ghost btn-sm",
      type: "button",
      onClick: () => loadCallGraph(item.qualified_name || item.name, item.repository_id),
    },
    [icon("graph"), "View call graph"],
  );
  const displayPath = [item.repository_name || item.repository_id, item.path]
    .filter(Boolean)
    .join(" / ");
  const meta = [
    displayPath,
    `${item.start_line}-${item.end_line}`,
    item.kind || "function",
  ].filter(Boolean);
  const score = `vector ${Number(item.vector_score ?? item.score).toFixed(3)}`;

  return el("article", { class: "code-result-card" }, [
    el("div", { class: "code-result-top" }, [
      el("div", {}, [
        el("h3", { text: item.qualified_name || item.name }),
        el("div", { class: "ref-sub", text: meta.join(" · ") }),
      ]),
      el("div", { class: "ref-score", text: score }),
    ]),
    item.signature
      ? el("pre", { class: "code-signature", text: item.signature })
      : null,
    el("pre", { class: "code-snippet", text: item.snippet || "" }),
    el("div", { class: "code-actions" }, [graphButton]),
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
  const isCode = message.route === "code_search";
  meta.append(
    el("span", {
      class: `route-pill${message.usedRag || isCode ? " rag" : ""}`,
      text: `${isCode ? "Code" : message.usedRag ? "RAG" : "Direct"} · ${
        message.route || "unknown"
      }`,
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
  if (item.embeddingDegraded && item.qdrantDegraded) {
    return "Embedding and Qdrant/vector recall degraded; using BM25-only references.";
  }
  if (item.embeddingDegraded) {
    return "Embedding health degraded; using BM25-only references.";
  }
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
  if (codeCy) {
    codeCy.destroy();
    codeCy = null;
  }

  if (state.activeMode === "code") {
    renderCodeGraphPanel();
    return;
  }

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

function renderCodeGraphPanel() {
  const graph = state.codeGraph;
  if (!graph) {
    els.referenceList.append(
      el("div", { class: "empty", text: "Select a function to view its call graph." }),
    );
    return;
  }
  if (graph.loading) {
    els.referenceList.append(
      el("div", { class: "bubble loading" }, [
        el("span", { text: "Loading graph" }),
        el("span", { class: "dots" }),
      ]),
    );
    return;
  }
  if (graph.error) {
    els.referenceList.append(el("div", { class: "alert alert-error", text: graph.error }));
    return;
  }

  const title = graph.function_name || state.codeGraphFunction;
  els.referenceList.append(
    el("div", { class: "code-graph-summary" }, [
      el("div", { class: "ref-source", text: title }),
      el("div", {
        class: "ref-sub",
        text: `${graph.nodes?.length || 0} nodes · ${graph.edges?.length || 0} edges`,
      }),
    ]),
  );

  if (graph.callers?.length) {
    const callers = el("div", { class: "code-callers" }, [
      el("div", { class: "popover-title", text: "Callers" }),
    ]);
    for (const caller of graph.callers.slice(0, 12)) {
      callers.append(
        el("button", {
          class: "caller-chip",
          type: "button",
          text: caller.label || caller.id,
          title: caller.path || "",
          onClick: () => loadCallGraph(caller.id, caller.repository_id),
        }),
      );
    }
    els.referenceList.append(callers);
  }

  if (!graph.nodes?.length) {
    els.referenceList.append(el("div", { class: "empty", text: "No graph nodes found." }));
    return;
  }

  const graphEl = el("div", { class: "code-graph-canvas" });
  els.referenceList.append(graphEl);
  requestAnimationFrame(() => renderCytoscapeGraph(graphEl, graph));
  els.referenceList.append(renderGraphEdges(graph.edges || []));
}

function renderCytoscapeGraph(container, graph) {
  if (!window.cytoscape) {
    container.append(el("div", { class: "empty", text: "Graph renderer unavailable." }));
    return;
  }

  const theme = getComputedStyle(document.documentElement);
  const accent = theme.getPropertyValue("--accent").trim() || "#0f766e";
  const accentBorder = theme.getPropertyValue("--accent-border").trim() || "#b9e0d9";
  const muted = theme.getPropertyValue("--text-subtle").trim() || "#8a939e";
  const text = theme.getPropertyValue("--text").trim() || "#18212b";
  const surface = theme.getPropertyValue("--surface").trim() || "#ffffff";
  const border = theme.getPropertyValue("--border").trim() || "#d7dde4";

  const elements = layoutGraphElements(normalizeGraphElements(graph), container);

  codeCy = window.cytoscape({
    container,
    elements,
    layout: { name: "preset", fit: true, padding: 24 },
    style: [
      {
        selector: "node",
        style: {
          shape: "roundrectangle",
          "background-color": surface,
          "border-color": accentBorder,
          "border-width": 1.5,
          color: text,
          label: "data(label)",
          "font-size": 11,
          "font-weight": 600,
          "text-wrap": "wrap",
          "text-max-width": 124,
          "text-valign": "center",
          "text-halign": "center",
          width: 148,
          height: 54,
          padding: "8px",
        },
      },
      {
        selector: 'node[type = "class"]',
        style: {
          "background-color": "#f4efff",
          "border-color": "#a855f7",
        },
      },
      {
        selector: 'node[type = "component"]',
        style: {
          "background-color": "#fff7ed",
          "border-color": "#f97316",
        },
      },
      {
        selector: 'node[type = "file"]',
        style: {
          "background-color": "#eff6ff",
          "border-color": "#3b82f6",
        },
      },
      {
        selector: 'node[type = "function"]',
        style: {
          "background-color": "#ecfdf5",
          "border-color": accent,
        },
      },
      {
        selector: 'node[indexed = "false"]',
        style: {
          "background-color": "#f8fafc",
          "border-color": border,
          color: muted,
        },
      },
      {
        selector: "edge",
        style: {
          width: 2,
          "line-color": muted,
          "target-arrow-color": muted,
          "target-arrow-shape": "triangle",
          "curve-style": "taxi",
          "taxi-direction": "downward",
          "taxi-turn": 36,
          label: "data(label)",
          "font-size": 8,
          "text-background-color": surface,
          "text-background-opacity": 0.75,
        },
      },
      {
        selector: 'edge[type = "calls"]',
        style: {
          "line-color": accent,
          "target-arrow-color": accent,
        },
      },
      {
        selector: 'edge[type = "imports"]',
        style: {
          "line-color": "#3b82f6",
          "target-arrow-color": "#3b82f6",
        },
      },
      {
        selector: 'edge[type = "extends"]',
        style: {
          "line-color": "#a855f7",
          "target-arrow-color": "#a855f7",
        },
      },
      {
        selector: 'edge[type = "uses"]',
        style: {
          "line-color": "#f59e0b",
          "target-arrow-color": "#f59e0b",
        },
      },
    ],
    minZoom: 0.35,
    maxZoom: 2.5,
    wheelSensitivity: 0.2,
  });
}

function normalizeGraphElements(graph) {
  if (Array.isArray(graph.elements) && graph.elements.length) {
    return graph.elements.map((element) => {
      if (element.group === "nodes") {
        return {
          ...element,
          data: {
            ...element.data,
            label: compactNodeLabel(element.data?.label || element.data?.id),
            kind: element.data?.kind || "external",
            type: graphNodeType(element.data),
            indexed: String(element.data?.indexed !== false),
          },
        };
      }
      return {
        ...element,
        data: {
          ...element.data,
          label: edgeDisplayLabel(element.data),
          type: graphEdgeType(element.data),
        },
      };
    });
  }

  return [
    ...(graph.nodes || []).map((node) => ({
      group: "nodes",
      data: {
        id: node.id,
        label: compactNodeLabel(node.label || node.id),
        kind: node.kind || "external",
        type: graphNodeType(node),
        indexed: String(node.indexed !== false),
        filePath: node.filePath || node.path || "",
        codeSnippet: node.codeSnippet || "",
        description: node.description || "",
      },
    })),
    ...(graph.edges || []).map((edge, index) => ({
      group: "edges",
      data: {
        id: `${edge.source}->${edge.target}:${index}`,
        source: edge.source,
        target: edge.target,
        label: edgeDisplayLabel(edge),
        type: graphEdgeType(edge),
      },
    })),
  ];
}

function layoutGraphElements(elements, container) {
  const nodes = elements.filter((item) => item.group === "nodes");
  const edges = elements.filter((item) => item.group === "edges");
  const nodeIds = new Set(nodes.map((node) => node.data.id));
  const incoming = new Map(nodes.map((node) => [node.data.id, []]));
  const outgoing = new Map(nodes.map((node) => [node.data.id, []]));

  for (const edge of edges) {
    if (!nodeIds.has(edge.data.source) || !nodeIds.has(edge.data.target)) continue;
    incoming.get(edge.data.target)?.push(edge.data.source);
    outgoing.get(edge.data.source)?.push(edge.data.target);
  }

  const ranks = new Map();
  let roots = nodes
    .map((node) => node.data.id)
    .filter((id) => !(incoming.get(id)?.length));
  if (!roots.length && nodes.length) roots = [nodes[0].data.id];

  const queue = [];
  for (const id of roots) {
    if (ranks.has(id)) continue;
    ranks.set(id, 0);
    queue.push(id);
  }
  while (queue.length) {
    const id = queue.shift();
    const rank = ranks.get(id) || 0;
    for (const target of outgoing.get(id) || []) {
      if (ranks.has(target)) continue;
      ranks.set(target, rank + 1);
      queue.push(target);
    }
  }
  for (const node of nodes) {
    if (!ranks.has(node.data.id)) ranks.set(node.data.id, 0);
  }

  const layers = new Map();
  for (const node of nodes) {
    const rank = ranks.get(node.data.id) || 0;
    if (!layers.has(rank)) layers.set(rank, []);
    layers.get(rank).push(node);
  }

  const width = Math.max(container.clientWidth || 320, 320);
  const layerGap = 116;
  const nodeGap = 172;
  for (const rank of [...layers.keys()].sort((a, b) => a - b)) {
    const layer = layers.get(rank);
    const layerWidth = Math.max(0, (layer.length - 1) * nodeGap);
    layer.forEach((node, index) => {
      node.position = {
        x: width / 2 - layerWidth / 2 + index * nodeGap,
        y: 48 + rank * layerGap,
      };
    });
  }

  return [...nodes, ...edges];
}

function graphNodeType(data) {
  const type = data?.type || data?.kind;
  if (["file", "function", "class", "component"].includes(type)) return type;
  return "function";
}

function graphEdgeType(data) {
  const type = data?.type || "";
  if (["imports", "calls", "extends", "uses"].includes(type)) return type;
  return "calls";
}

function edgeDisplayLabel(data) {
  if (data?.label) return data.label;
  if (Array.isArray(data?.lines) && data.lines.length) return data.lines.join(", ");
  return graphEdgeType(data);
}

function renderGraphEdges(edges) {
  const wrap = el("div", { class: "graph-edge-list" });
  if (!edges.length) {
    wrap.append(el("div", { class: "empty", text: "No graph relationships found." }));
    return wrap;
  }
  for (const edge of edges.slice(0, 30)) {
    wrap.append(
      el("div", { class: "graph-edge-item" }, [
        el("span", { text: compactNodeLabel(edge.source) }),
        el("span", { text: "→" }),
        el("span", { text: compactNodeLabel(edge.target) }),
      ]),
    );
  }
  return wrap;
}

function compactNodeLabel(label) {
  const text = String(label || "");
  const parts = text.split(/::|\./);
  return parts[parts.length - 1] || text;
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
