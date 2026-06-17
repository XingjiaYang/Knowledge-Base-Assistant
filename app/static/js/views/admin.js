// Admin dashboard: list users, create, CSV import, toggle role, reset
// password, clear chat data, and delete users.

import { state, on, emit } from "../store.js?v=20260617-hybrid-retrieval";
import { api } from "../api.js?v=20260617-hybrid-retrieval";
import { byId, el, clear, icon } from "../dom.js?v=20260617-hybrid-retrieval";
import { loadSessions } from "./sessions.js?v=20260617-hybrid-retrieval";

let els = {};

// Render a prominent feedback box. Empty message hides it (`.alert:empty`).
function setAlert(node, message, kind = "error") {
  clear(node);
  node.className = "alert";
  if (!message) return;
  node.classList.add(`alert-${kind}`);
  node.append(icon(kind === "success" ? "check" : "alert"), el("span", { text: message }));
}

export function mountAdmin() {
  els = {
    createForm: byId("adminCreateForm"),
    newUsername: byId("adminNewUsername"),
    newPassword: byId("adminNewPassword"),
    newIsAdmin: byId("adminNewIsAdmin"),
    createStatus: byId("adminCreateStatus"),
    csvForm: byId("adminCsvForm"),
    csvFile: byId("adminCsvFile"),
    csvStatus: byId("adminCsvStatus"),
    llmSection: byId("adminLlmSettingsSection"),
    llmForm: byId("adminLlmSettingsForm"),
    llmProvider: byId("adminLlmProvider"),
    llmBaseUrl: byId("adminLlmBaseUrl"),
    llmModel: byId("adminLlmModel"),
    llmApiKey: byId("adminLlmApiKey"),
    llmStatus: byId("adminLlmSettingsStatus"),
    error: byId("adminError"),
    refreshButton: byId("adminRefreshButton"),
    userList: byId("adminUserList"),
  };

  els.createForm.addEventListener("submit", onCreate);
  els.csvForm.addEventListener("submit", onImportCsv);
  els.llmForm.addEventListener("submit", onSaveLLMSettings);
  els.refreshButton.addEventListener("click", loadAdminUsers);

  on("admin", renderAdminUsers);
  on("open-admin", () => {
    setAlert(els.createStatus, "");
    setAlert(els.csvStatus, "");
    setAlert(els.llmStatus, "");
    setAlert(els.error, "");
    loadAdminUsers();
    loadLLMSettings();
  });
}

export async function loadAdminUsers() {
  if (!state.currentUser?.is_admin) return;
  try {
    state.adminUsers = await api.listUsers();
  } catch (error) {
    setAlert(els.error, error.message || "Unable to load users.", "error");
    return;
  }
  emit("admin");
}

async function loadLLMSettings() {
  const isSuperuser = Boolean(state.currentUser?.is_superuser);
  els.llmSection.hidden = !isSuperuser;
  if (!isSuperuser) return;
  try {
    state.llmSettings = await api.getLLMSettings();
    renderLLMSettings();
  } catch (error) {
    setAlert(els.llmStatus, error.message || "Unable to load LLM settings.", "error");
  }
}

async function onCreate(event) {
  event.preventDefault();
  const username = els.newUsername.value.trim();
  const password = els.newPassword.value;
  if (!username || !password) {
    setAlert(els.createStatus, "Username and password are required.", "error");
    return;
  }
  try {
    await api.createUser(username, password, els.newIsAdmin.checked);
    els.newUsername.value = "";
    els.newPassword.value = "";
    els.newIsAdmin.checked = false;
    setAlert(els.createStatus, `User "${username}" created.`, "success");
    await loadAdminUsers();
  } catch (error) {
    setAlert(els.createStatus, error.message, "error");
  }
}

async function onImportCsv(event) {
  event.preventDefault();
  setAlert(els.csvStatus, "");
  const file = els.csvFile.files[0];
  if (!file) {
    setAlert(els.csvStatus, "Choose a CSV file first.", "error");
    return;
  }
  try {
    const text = await file.text();
    const data = await api.importCsv(text);
    els.csvFile.value = "";
    setAlert(els.csvStatus, `Created ${data.created || 0} users.`, "success");
    await loadAdminUsers();
  } catch (error) {
    setAlert(els.csvStatus, error.message, "error");
  }
}

async function onSaveLLMSettings(event) {
  event.preventDefault();
  const payload = {
    provider: els.llmProvider.value,
    base_url: els.llmBaseUrl.value.trim(),
    model: els.llmModel.value.trim(),
  };
  const apiKey = els.llmApiKey.value.trim();
  if (apiKey) payload.api_key = apiKey;
  if (!payload.base_url || !payload.model) {
    setAlert(els.llmStatus, "API URL and model are required.", "error");
    return;
  }
  try {
    state.llmSettings = await api.updateLLMSettings(payload);
    els.llmApiKey.value = "";
    renderLLMSettings();
    setAlert(els.llmStatus, "LLM settings saved.", "success");
  } catch (error) {
    setAlert(els.llmStatus, error.message, "error");
  }
}

async function toggleRole(user, isAdmin, checkbox) {
  try {
    await api.setUserRole(user.id, isAdmin);
    await loadAdminUsers();
    setAlert(els.error, "");
  } catch (error) {
    setAlert(els.error, error.message, "error");
    checkbox.checked = Boolean(user.is_admin);
  }
}

async function resetPassword(user) {
  const password = window.prompt(`New password for ${user.username}`);
  if (!password) return;
  try {
    await api.setUserPassword(user.id, password);
    await loadAdminUsers();
    setAlert(
      els.error,
      `Password reset for ${user.username}. They must change it at next login.`,
      "success",
    );
  } catch (error) {
    setAlert(els.error, error.message, "error");
  }
}

async function clearData(user) {
  if (!window.confirm(`Clear all chat data for ${user.username}?`)) return;
  try {
    await api.clearUserSessions(user.id);
    if (user.id === state.currentUser?.id) await loadSessions(true);
    await loadAdminUsers();
    setAlert(els.error, `Cleared chat data for ${user.username}.`, "success");
  } catch (error) {
    setAlert(els.error, error.message, "error");
  }
}

async function deleteUser(user) {
  if (!window.confirm(`Delete ${user.username}? This removes their account and chats.`)) return;
  try {
    await api.deleteUser(user.id);
    await loadAdminUsers();
    setAlert(els.error, `Deleted ${user.username}.`, "success");
  } catch (error) {
    setAlert(els.error, error.message, "error");
  }
}

// ----------------------------- rendering -----------------------------

function renderAdminUsers() {
  clear(els.userList);
  if (!state.adminUsers.length) {
    els.userList.append(el("div", { class: "empty", text: "No users yet." }));
    return;
  }
  for (const user of state.adminUsers) {
    els.userList.append(renderUser(user));
  }
}

function renderLLMSettings() {
  const settings = state.llmSettings;
  if (!settings) return;
  els.llmProvider.value = settings.provider || "openai_compatible";
  els.llmBaseUrl.value = settings.base_url || "";
  els.llmModel.value = settings.model || "";
  const keyState = settings.api_key_configured ? "key configured" : "no key configured";
  setAlert(
    els.llmStatus,
    `Using ${settings.source || ".env"} settings, ${keyState}.`,
    "success",
  );
}

function renderUser(user) {
  const isSelf = user.id === state.currentUser?.id;
  const isSuperuser = Boolean(user.is_superuser);

  const tags = el("div", { class: "admin-user-tags" });
  if (isSuperuser) tags.append(el("span", { class: "tag admin", text: "Superuser" }));
  if (user.is_admin) tags.append(el("span", { class: "tag admin", text: "Admin" }));
  tags.append(el("span", { class: "tag", text: `${user.session_count || 0} sessions` }));
  tags.append(el("span", { class: "tag", text: `${user.message_count || 0} messages` }));
  if (user.must_change_password) {
    tags.append(el("span", { class: "tag warn", text: "must change password" }));
  }

  const identity = el("div", {}, [
    el("div", { class: "admin-user-name", text: user.username }),
    tags,
  ]);

  const roleCheckbox = el("input", { type: "checkbox", disabled: isSelf || isSuperuser });
  roleCheckbox.checked = Boolean(user.is_admin);
  roleCheckbox.addEventListener("change", () =>
    toggleRole(user, roleCheckbox.checked, roleCheckbox),
  );
  const roleLabel = el("label", { class: "checkbox" }, [roleCheckbox, "Admin"]);

  const top = el("div", { class: "admin-user-top" }, [identity, roleLabel]);

  const actions = el("div", { class: "admin-user-actions" }, [
    el("button", { class: "btn btn-ghost btn-sm", type: "button", onClick: () => resetPassword(user) }, [
      icon("key"),
      "Reset",
    ]),
    el("button", { class: "btn btn-ghost btn-sm", type: "button", onClick: () => clearData(user) }, [
      icon("trash"),
      "Clear",
    ]),
    el(
      "button",
      {
        class: "btn btn-danger btn-sm",
        type: "button",
        disabled: isSelf || isSuperuser,
        onClick: () => deleteUser(user),
      },
      [icon("close"), "Delete"],
    ),
  ]);

  return el("div", { class: "admin-user" }, [top, actions]);
}
