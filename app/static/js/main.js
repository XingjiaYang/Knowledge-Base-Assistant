// Entry point: mount views, wire cross-cutting events, manage responsive
// drawers, and restore the persisted auth session on load.

import { state, on, emit, loadToken, setStatus } from "./store.js?v=20260617-hybrid-retrieval";
import { api } from "./api.js?v=20260617-hybrid-retrieval";
import { byId } from "./dom.js?v=20260617-hybrid-retrieval";
import { mountAuth, signOutLocal } from "./views/auth.js?v=20260617-hybrid-retrieval";
import { mountSessions, loadSessions } from "./views/sessions.js?v=20260617-hybrid-retrieval";
import { mountChat } from "./views/chat.js?v=20260617-hybrid-retrieval";
import { mountMeta, refreshHealth } from "./views/meta.js?v=20260617-hybrid-retrieval";
import { mountAdmin } from "./views/admin.js?v=20260617-hybrid-retrieval";

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
