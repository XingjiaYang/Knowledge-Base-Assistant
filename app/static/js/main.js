// Entry point: mount views, wire cross-cutting events, manage responsive
// drawers, and restore the persisted auth session on load.

import { state, on, emit, loadToken, setStatus } from "./store.js";
import { api } from "./api.js";
import { byId } from "./dom.js";
import { mountAuth, signOutLocal } from "./views/auth.js";
import { mountSessions, loadSessions } from "./views/sessions.js";
import { mountChat } from "./views/chat.js";
import { mountMeta, refreshHealth } from "./views/meta.js";
import { mountAdmin } from "./views/admin.js";

mountAuth();
mountMeta();
mountSessions();
mountChat();
mountAdmin();
mountLayout();

// Global auth signals raised by the API layer.
on("unauthorized", () => signOutLocal("Session expired. Sign in again."));
on("password-required", () => {
  if (state.currentUser) {
    state.currentUser = { ...state.currentUser, must_change_password: true };
  }
  state.passwordChangeOpen = false;
  setStatus("Password change required", "bad");
  emit("auth");
});

// After a successful login / restore / password change.
on("authenticated", async () => {
  await loadSessions(true);
  await refreshHealth();
});

function mountLayout() {
  const sidebar = byId("sidebar");
  const refs = byId("refsPanel");
  const scrim = byId("scrim");
  const isMobile = () => window.matchMedia("(max-width: 760px)").matches;

  const showScrim = () => {
    scrim.hidden = false;
    scrim.classList.add("show");
  };
  const closeDrawers = () => {
    sidebar.classList.remove("open");
    refs.classList.remove("open");
    scrim.classList.remove("show");
    scrim.hidden = true;
  };

  byId("sidebarToggle").addEventListener("click", () => {
    if (sidebar.classList.contains("open")) {
      closeDrawers();
    } else {
      sidebar.classList.add("open");
      showScrim();
    }
  });
  byId("refsToggle").addEventListener("click", () => {
    refs.classList.add("open");
    showScrim();
  });
  byId("refsClose").addEventListener("click", closeDrawers);
  scrim.addEventListener("click", closeDrawers);

  // Selecting a session on a phone should close the sidebar drawer.
  on("session-selected", () => {
    if (isMobile()) closeDrawers();
  });
}

async function restoreAuthSession() {
  state.authToken = loadToken();
  if (!state.authToken) {
    signOutLocal("");
    return;
  }
  try {
    state.currentUser = await api.me();
    state.currentView = "chat";
    emit("auth");
    if (state.currentUser.must_change_password) {
      setStatus("Password change required", "bad");
      return;
    }
    emit("authenticated");
  } catch {
    signOutLocal("Session expired. Sign in again.");
  }
}

restoreAuthSession();
