// Status indicator, health/model metadata, and the retrieval-settings popover.

import { state, on, emit, setStatus } from "../store.js?v=20260620-rag-only";
import { api } from "../api.js?v=20260620-rag-only";
import { byId } from "../dom.js?v=20260620-rag-only";

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

  on("meta", renderMeta);
}

function togglePopover() {
  if (els.settingsPopover.hidden) {
    els.settingsPopover.hidden = false;
    els.settingsToggle.setAttribute("aria-expanded", "true");
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
