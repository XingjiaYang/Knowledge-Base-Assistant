// Status indicator, health/model metadata, and the retrieval-settings popover.

import { state, on, emit, setStatus } from "../store.js?v=20260626-code-search";
import { api } from "../api.js?v=20260626-code-search";
import { byId, clear, icon } from "../dom.js?v=20260626-code-search";

let els = {};

export function mountMeta() {
  els = {
    status: byId("status"),
    statusText: byId("statusText"),
    settingsToggle: byId("settingsToggle"),
    settingsPopover: byId("settingsPopover"),
    ragOnlyMode: byId("ragOnlyMode"),
    bm25TopK: byId("bm25TopK"),
    recallTopK: byId("recallTopK"),
    rrfTopK: byId("rrfTopK"),
    topK: byId("topK"),
    codeRepository: byId("settingsCodeRepository"),
    codeIndexButton: byId("settingsCodeIndexButton"),
    codeIndexStatus: byId("settingsCodeIndexStatus"),
    modelName: byId("modelName"),
    collectionName: byId("collectionName"),
    maxTokens: byId("maxTokens"),
    rerankerName: byId("rerankerName"),
    latestRoute: byId("latestRoute"),
    compactChip: byId("compactChip"),
  };

  els.settingsToggle.addEventListener("click", (event) => {
    event.stopPropagation();
    togglePopover();
  });
  // Close the popover on outside click or Escape.
  document.addEventListener("click", (event) => {
    if (els.settingsPopover.hidden) return;
    if (!els.settingsPopover.contains(event.target) && event.target !== els.settingsToggle) {
      closePopover();
    }
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") closePopover();
  });

  els.ragOnlyMode.addEventListener("change", () => {
    state.ragOnly = els.ragOnlyMode.checked;
  });
  els.bm25TopK.addEventListener("input", () => {
    state.bm25TopK = els.bm25TopK.value ? Number(els.bm25TopK.value) : null;
  });
  els.recallTopK.addEventListener("input", () => {
    state.recallTopK = els.recallTopK.value ? Number(els.recallTopK.value) : null;
  });
  els.rrfTopK.addEventListener("input", () => {
    state.rrfTopK = els.rrfTopK.value ? Number(els.rrfTopK.value) : null;
  });
  els.topK.addEventListener("input", () => {
    state.topK = els.topK.value ? Number(els.topK.value) : null;
  });
  els.codeRepository.addEventListener("change", () => {
    state.selectedCodeRepositoryId = els.codeRepository.value;
    renderCodeIndexSettings();
    emit("messages");
  });
  els.codeIndexButton.addEventListener("click", indexSelectedCodeRepositories);

  on("meta", renderMeta);
  on("authenticated", () => {
    ensureCodeRepositories();
  });
}

function togglePopover() {
  if (els.settingsPopover.hidden) {
    els.settingsPopover.hidden = false;
    els.settingsToggle.setAttribute("aria-expanded", "true");
    ensureCodeRepositories();
    renderCodeIndexSettings();
  } else {
    closePopover();
  }
}

function closePopover() {
  els.settingsPopover.hidden = true;
  els.settingsToggle.setAttribute("aria-expanded", "false");
}

export async function refreshHealth() {
  if (state.currentUser?.must_change_password) {
    setStatus("Password change required", "bad");
    return;
  }
  const live = await api.live();
  if (!live) {
    setStatus("Offline", "bad");
    return;
  }
  if (!state.currentUser) {
    setStatus("Login required", "bad");
    return;
  }
  try {
    const data = await api.healthDetails();
    state.health = data;
    const ready = Boolean(data.qdrant && data.llm);
    setStatus(ready ? "Ready" : "Degraded", ready ? "ok" : "bad");
  } catch {
    setStatus("Offline", "bad");
  }
}

function renderMeta() {
  els.status.className = `status ${state.statusKind}`.trim();
  els.statusText.textContent = state.statusText;
  els.latestRoute.textContent = state.latestRoute || "None";
  els.compactChip.hidden = !state.conversationSummary;
  els.ragOnlyMode.checked = Boolean(state.ragOnly);

  const data = state.health;
  if (!data) return;

  els.modelName.textContent = data.llm_model || "—";
  els.collectionName.textContent = data.collection || "—";
  els.maxTokens.textContent = data.llm_max_tokens ? `${data.llm_max_tokens} tokens` : "—";
  els.rerankerName.textContent = data.reranker_enabled
    ? data.reranker_model || "Enabled"
    : "Disabled";

  applyNumberSetting(els.bm25TopK, "bm25TopK", data.bm25_top_k, data.api_recall_top_k_max);
  applyNumberSetting(els.recallTopK, "recallTopK", data.recall_top_k, data.api_recall_top_k_max);
  applyNumberSetting(els.rrfTopK, "rrfTopK", data.rrf_top_k, data.api_recall_top_k_max);
  applyNumberSetting(els.topK, "topK", data.retrieve_top_k, data.api_top_k_max);
  renderCodeIndexSettings();
}

function applyNumberSetting(input, stateKey, value, max) {
  if (max) input.max = String(max);
  if (!value) return;
  input.placeholder = String(value);
  if (!input.value) {
    input.value = String(value);
    state[stateKey] = value;
  }
}

async function ensureCodeRepositories() {
  if (!state.currentUser || state.currentUser.must_change_password) return;
  if (state.codeRepositoriesLoaded || state.codeRepositoriesLoading) return;
  state.codeRepositoriesLoading = true;
  renderCodeIndexSettings();
  try {
    const data = await api.codeRepositories();
    state.codeRepositories = data.repositories || [];
    state.codeRepositoriesLoaded = true;
    const ids = new Set(state.codeRepositories.map((repository) => repository.id));
    if (state.selectedCodeRepositoryId && !ids.has(state.selectedCodeRepositoryId)) {
      state.selectedCodeRepositoryId = "";
    }
    if (!state.selectedCodeRepositoryId && data.default_repository_ids?.length === 1) {
      state.selectedCodeRepositoryId = data.default_repository_ids[0];
    }
    state.codeError = "";
  } catch (error) {
    state.codeError = error.message || "Unable to load code folders.";
  } finally {
    state.codeRepositoriesLoading = false;
    renderCodeIndexSettings();
    emit("messages");
  }
}

async function indexSelectedCodeRepositories() {
  await ensureCodeRepositories();
  if (state.codeIndexing || !state.codeRepositories.length) return;

  state.codeIndexing = true;
  state.codeIndexStatus = "Indexing code...";
  renderCodeIndexSettings();
  emit("messages");

  try {
    const data = await api.codeIndex({
      repository_ids: selectedCodeRepositoryIds(),
      rebuild: true,
    });
    state.codeRepositories = data.repositories || state.codeRepositories;
    state.codeRepositoriesLoaded = true;
    state.codeIndexStatus =
      `Indexed ${data.files || 0} files, ${data.functions || 0} symbols, `
      + `${data.call_edges || 0} calls.`;
    state.codeError = "";
  } catch (error) {
    state.codeError = error.message || "Code indexing failed.";
    state.codeIndexStatus = "";
  } finally {
    state.codeIndexing = false;
    renderCodeIndexSettings();
    emit("messages");
  }
}

function renderCodeIndexSettings() {
  if (!els.codeRepository || !els.codeIndexButton || !els.codeIndexStatus) return;

  const repos = state.codeRepositories || [];
  clear(els.codeRepository);
  const all = document.createElement("option");
  all.value = "";
  all.textContent = "All code folders";
  els.codeRepository.append(all);
  for (const repository of repos) {
    const option = document.createElement("option");
    option.value = repository.id;
    option.textContent =
      `${repository.name || repository.id} (${repository.indexed_files || 0})`;
    els.codeRepository.append(option);
  }
  els.codeRepository.value = state.selectedCodeRepositoryId || "";
  els.codeRepository.disabled = state.codeRepositoriesLoading || state.codeIndexing;

  const indexedCount = selectedIndexedFileCount(repos);
  const buttonLabel = state.codeIndexing
    ? "Indexing"
    : indexedCount > 0
      ? "Reindex"
      : "Index";
  clear(els.codeIndexButton);
  els.codeIndexButton.append(icon("refresh"), document.createTextNode(buttonLabel));
  els.codeIndexButton.disabled =
    state.codeRepositoriesLoading
    || state.codeIndexing
    || !repos.length
    || !state.currentUser
    || Boolean(state.currentUser?.must_change_password);

  if (state.codeIndexStatus) {
    els.codeIndexStatus.textContent = state.codeIndexStatus;
  } else if (state.codeRepositoriesLoading) {
    els.codeIndexStatus.textContent = "Loading folders...";
  } else if (state.codeError) {
    els.codeIndexStatus.textContent = state.codeError;
  } else if (!repos.length) {
    els.codeIndexStatus.textContent = "No code folders";
  } else {
    els.codeIndexStatus.textContent = `${indexedCount} indexed files`;
  }
}

function selectedCodeRepositoryIds() {
  return state.selectedCodeRepositoryId ? [state.selectedCodeRepositoryId] : [];
}

function selectedIndexedFileCount(repos) {
  if (state.selectedCodeRepositoryId) {
    const selected = repos.find(
      (repository) => repository.id === state.selectedCodeRepositoryId,
    );
    return selected?.indexed_files || 0;
  }
  return repos.reduce(
    (total, repository) => total + (repository.indexed_files || 0),
    0,
  );
}
