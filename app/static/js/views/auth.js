// Authentication: login gate, password-change gate, user card, and which
// top-level screen is visible. Auth actions emit events that main.js wires
// to data loading (sessions, health, admin) — avoiding cross-view imports.

import { state, emit, on, saveToken, clearToken, resetState, setStatus } from "../store.js?v=20260620-degraded-retrieval";
import { api } from "../api.js?v=20260620-degraded-retrieval";
import { byId } from "../dom.js?v=20260620-degraded-retrieval";

let els = {};

export function mountAuth() {
  els = {
    loginGate: byId("loginGate"),
    passwordGate: byId("passwordGate"),
    appShell: byId("appShell"),
    adminDashboard: byId("adminDashboard"),

    loginForm: byId("loginForm"),
    loginUsername: byId("loginUsername"),
    loginPassword: byId("loginPassword"),
    loginButton: byId("loginButton"),
    loginError: byId("loginError"),

    passwordChangeForm: byId("passwordChangeForm"),
    passwordChangeTitle: byId("passwordChangeTitle"),
    passwordChangeHelp: byId("passwordChangeHelp"),
    passwordCurrent: byId("passwordCurrent"),
    passwordNew: byId("passwordNew"),
    passwordChangeButton: byId("passwordChangeButton"),
    passwordCancelButton: byId("passwordCancelButton"),
    passwordLogoutButton: byId("passwordLogoutButton"),
    passwordChangeError: byId("passwordChangeError"),

    userName: byId("userName"),
    userRole: byId("userRole"),
    userCard: byId("userCard"),
    adminDashboardButton: byId("adminDashboardButton"),
    changePasswordButton: byId("changePasswordButton"),
    logoutButton: byId("logoutButton"),

    adminChangePasswordButton: byId("adminChangePasswordButton"),
    backToChatButton: byId("backToChatButton"),
    adminLogoutButton: byId("adminLogoutButton"),
  };

  els.loginForm.addEventListener("submit", (event) => {
    event.preventDefault();
    const username = els.loginUsername.value.trim();
    const password = els.loginPassword.value;
    if (!username || !password) {
      els.loginError.textContent = "Username and password are required.";
      return;
    }
    doLogin(username, password);
  });

  els.passwordChangeForm.addEventListener("submit", (event) => {
    event.preventDefault();
    const current = els.passwordCurrent.value;
    const next = els.passwordNew.value;
    if (!current || !next) {
      els.passwordChangeError.textContent =
        "Current password and new password are required.";
      return;
    }
    doChangePassword(current, next);
  });

  els.changePasswordButton.addEventListener("click", openPasswordChange);
  els.adminChangePasswordButton.addEventListener("click", openPasswordChange);
  els.passwordCancelButton.addEventListener("click", closePasswordChange);

  for (const btn of [els.logoutButton, els.passwordLogoutButton, els.adminLogoutButton]) {
    btn.addEventListener("click", doLogout);
  }

  els.adminDashboardButton.addEventListener("click", showAdmin);
  els.backToChatButton.addEventListener("click", showChat);

  on("auth", renderScreens);
}

async function doLogin(username, password) {
  els.loginButton.disabled = true;
  els.loginError.textContent = "";
  try {
    const data = await api.login(username, password);
    saveToken(data.token);
    state.authToken = data.token;
    state.currentUser = data.user;
    state.currentView = "chat";
    state.passwordChangeOpen = false;
    els.loginPassword.value = "";
    emit("auth");
    if (data.user.must_change_password) {
      setStatus("Password change required", "bad");
      return;
    }
    emit("authenticated");
  } catch (error) {
    els.loginError.textContent = error.message;
  } finally {
    els.loginButton.disabled = false;
  }
}

async function doLogout() {
  try {
    await api.logout();
  } catch {
    /* best effort */
  }
  signOutLocal("");
}

export function signOutLocal(message = "") {
  resetState();
  clearToken();
  els.loginError.textContent = message;
  els.passwordChangeError.textContent = "";
  els.passwordCurrent.value = "";
  els.passwordNew.value = "";
  state.statusText = "Login required";
  state.statusKind = "bad";
  emit("auth");
  emit("sessions");
  emit("messages");
  emit("references");
  emit("admin");
  emit("meta");
}

function openPasswordChange() {
  state.passwordChangeOpen = true;
  els.passwordChangeError.textContent = "";
  els.passwordCurrent.value = "";
  els.passwordNew.value = "";
  emit("auth");
}

function closePasswordChange() {
  if (state.currentUser?.must_change_password) return;
  state.passwordChangeOpen = false;
  els.passwordChangeError.textContent = "";
  els.passwordCurrent.value = "";
  els.passwordNew.value = "";
  emit("auth");
}

async function doChangePassword(current, next) {
  els.passwordChangeButton.disabled = true;
  els.passwordChangeError.textContent = "";
  try {
    const user = await api.changePassword(current, next);
    state.currentUser = user;
    state.passwordChangeOpen = false;
    els.passwordCurrent.value = "";
    els.passwordNew.value = "";
    emit("auth");
    emit("authenticated");
    if (state.currentView === "admin") emit("open-admin");
  } catch (error) {
    els.passwordChangeError.textContent = error.message;
  } finally {
    els.passwordChangeButton.disabled = false;
  }
}

function showAdmin() {
  if (!state.currentUser?.is_admin) return;
  state.currentView = "admin";
  emit("auth");
  emit("open-admin");
}

function showChat() {
  state.currentView = "chat";
  emit("auth");
  emit("focus-prompt");
}

function renderScreens() {
  const signedIn = Boolean(state.currentUser);
  const mustChange = Boolean(state.currentUser?.must_change_password);
  const showPwGate = signedIn && (mustChange || state.passwordChangeOpen);
  const showAdminView =
    signedIn && !showPwGate && state.currentUser.is_admin && state.currentView === "admin";
  const showApp = signedIn && !showPwGate && !showAdminView;

  els.loginGate.hidden = signedIn;
  els.passwordGate.hidden = !showPwGate;
  els.appShell.hidden = !showApp;
  els.adminDashboard.hidden = !showAdminView;
  els.userCard.hidden = !signedIn;
  els.adminDashboardButton.hidden = !signedIn || !state.currentUser.is_admin;
  els.passwordCancelButton.hidden = mustChange;

  els.passwordChangeTitle.textContent = mustChange
    ? "Change Password Required"
    : "Change Password";
  els.passwordChangeHelp.textContent = mustChange
    ? "This account is using an initial or reset password. Update it before continuing."
    : "Update your account password.";

  if (state.currentUser) {
    els.userName.textContent = state.currentUser.username;
    els.userRole.textContent = state.currentUser.is_admin ? "Administrator" : "User";
    if (showPwGate) els.passwordCurrent.focus();
  } else {
    els.userName.textContent = "";
    els.userRole.textContent = "";
    els.loginUsername.focus();
  }
}
