// Status indicator, health/model metadata, and the retrieval-settings popover.

import { state, on, emit, setStatus } from "../store.js";
import { api } from "../api.js";
import { byId } from "../dom.js";

let els = {};

export function mountMeta() {
  els = {
    status: byId("status"),
    statusText: byId("statusText"),
    settingsToggle: byId("settingsToggle"),
    settingsPopover: byId("settingsPopover"),
    recallTopK: byId("recallTopK"),
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

  els.recallTopK.addEventListener("input", () => {
    state.recallTopK = els.recallTopK.value ? Number(els.recallTopK.value) : null;
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

  const data = state.health;
  if (!data) return;

  els.modelName.textContent = data.llm_model || "—";
  els.collectionName.textContent = data.collection || "—";
  els.maxTokens.textContent = data.llm_max_tokens ? `${data.llm_max_tokens} tokens` : "—";
  els.rerankerName.textContent = data.reranker_enabled
    ? data.reranker_model || "Enabled"
    : "Disabled";

  if (data.api_recall_top_k_max) els.recallTopK.max = String(data.api_recall_top_k_max);
  if (data.recall_top_k) {
    els.recallTopK.placeholder = String(data.recall_top_k);
    if (!els.recallTopK.value) {
      els.recallTopK.value = String(data.recall_top_k);
      state.recallTopK = data.recall_top_k;
    }
  }
  if (data.api_top_k_max) els.topK.max = String(data.api_top_k_max);
  if (data.retrieve_top_k) {
    els.topK.placeholder = String(data.retrieve_top_k);
    if (!els.topK.value) {
      els.topK.value = String(data.retrieve_top_k);
      state.topK = data.retrieve_top_k;
    }
  }
}
