// Thin HTTP layer. Attaches the bearer token, surfaces 401/403 as events,
// and returns parsed JSON (throwing on non-OK with the server's detail).

import { state, loadToken, emit } from "./store.js?v=20260617-hybrid-retrieval";

async function raw(path, options = {}, { auth = true } = {}) {
  const headers = { ...(options.headers || {}) };
  if (auth) {
    const token = state.authToken || loadToken();
    if (token) headers.Authorization = `Bearer ${token}`;
  }

  const response = await fetch(path, { ...options, headers });

  if (auth && response.status === 401) {
    emit("unauthorized");
  } else if (response.status === 403) {
    const data = await response.clone().json().catch(() => ({}));
    if (data.detail === "Password change required.") emit("password-required");
  }
  return response;
}

async function send(path, options = {}, opts = {}) {
  const response = await raw(path, options, opts);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(data.detail || `Request failed (${response.status})`);
  }
  return data;
}

const jsonBody = (body) => ({
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

export const api = {
  // auth
  login: (username, password) =>
    send("/auth/login", { method: "POST", ...jsonBody({ username, password }) }, { auth: false }),
  me: () => send("/auth/me"),
  logout: () => raw("/auth/logout", { method: "POST" }),
  changePassword: (currentPassword, newPassword) =>
    send("/auth/password", {
      method: "POST",
      ...jsonBody({ current_password: currentPassword, new_password: newPassword }),
    }),

  // health
  live: () => fetch("/health").then((r) => r.ok).catch(() => false),
  healthDetails: () => send("/health/details"),

  // chat sessions
  listSessions: () => send("/sessions"),
  createSession: () => send("/sessions", { method: "POST" }),
  getSession: (id) => send(`/sessions/${id}`),
  renameSession: (id, title) =>
    send(`/sessions/${id}`, { method: "PATCH", ...jsonBody({ title }) }),
  deleteSession: (id) => raw(`/sessions/${id}`, { method: "DELETE" }),

  // rag
  ask: (payload) => send("/rag", { method: "POST", ...jsonBody(payload) }),

  // admin
  listUsers: () => send("/admin/users"),
  createUser: (username, password, isAdmin) =>
    send("/admin/users", {
      method: "POST",
      ...jsonBody({ username, password, is_admin: isAdmin }),
    }),
  importCsv: (csvText) =>
    send("/admin/users/import-csv", { method: "POST", ...jsonBody({ csv_text: csvText }) }),
  setUserRole: (id, isAdmin) =>
    send(`/admin/users/${id}/role`, { method: "PATCH", ...jsonBody({ is_admin: isAdmin }) }),
  setUserPassword: (id, password) =>
    send(`/admin/users/${id}/password`, { method: "POST", ...jsonBody({ password }) }),
  clearUserSessions: (id) => send(`/admin/users/${id}/sessions`, { method: "DELETE" }),
  deleteUser: (id) => send(`/admin/users/${id}`, { method: "DELETE" }),
};
