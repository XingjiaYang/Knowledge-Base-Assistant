// Central application state + a tiny pub/sub bus + auth-token persistence.
// Views subscribe to named events and re-render the slice they own.

export const state = {
  // chat
  messages: [],
  conversationSummary: "",
  summarizedMessageCount: 0,
  busy: false,
  latestRoute: "None",
  lastReferences: {
    contexts: [],
    usedRag: true,
    retrievalDegraded: false,
    degradationReason: "",
  },
  bm25TopK: null,
  recallTopK: null,
  rrfTopK: null,
  topK: null,
  ragOnly: false,
  activeMode: "chat",
  codeResults: [],
  codeFiles: [],
  codeQuery: "",
  codeGraph: null,
  codeGraphFunction: "",
  codeSearchLoading: false,
  codeRepositories: [],
  codeRepositoriesLoaded: false,
  codeRepositoriesLoading: false,
  codeIndexing: false,
  codeIndexStatus: "",
  selectedCodeRepositoryId: "",
  codeError: "",
  // auth
  authToken: "",
  currentUser: null,
  passwordChangeOpen: false,
  // sessions
  sessions: [],
  activeSessionId: "",
  sessionFilter: "",
  renamingSessionId: "",
  // admin
  adminUsers: [],
  llmSettings: null,
  currentView: "chat", // "chat" | "admin"
  // meta
  health: null,
  statusText: "Checking…",
  statusKind: "",
};

const AUTH_TOKEN_KEY = "knowledgeBaseAssistantAuthToken";
export const loadToken = () => localStorage.getItem(AUTH_TOKEN_KEY) || "";
export const saveToken = (token) => localStorage.setItem(AUTH_TOKEN_KEY, token);
export const clearToken = () => localStorage.removeItem(AUTH_TOKEN_KEY);

const handlers = new Map();

export function on(event, fn) {
  if (!handlers.has(event)) handlers.set(event, new Set());
  handlers.get(event).add(fn);
  return () => handlers.get(event)?.delete(fn);
}

export function emit(event, payload) {
  const set = handlers.get(event);
  if (!set) return;
  for (const fn of [...set]) {
    try {
      fn(payload);
    } catch (error) {
      console.error(`Handler for "${event}" failed:`, error);
    }
  }
}

export function setStatus(text, kind = "") {
  state.statusText = text;
  state.statusKind = kind;
  emit("meta");
}

// Reset all client-side state to a signed-out baseline.
export function resetState() {
  Object.assign(state, {
    messages: [],
    conversationSummary: "",
    summarizedMessageCount: 0,
    busy: false,
    latestRoute: "None",
    lastReferences: {
      contexts: [],
      usedRag: true,
      retrievalDegraded: false,
      degradationReason: "",
    },
    bm25TopK: null,
    recallTopK: null,
    rrfTopK: null,
    topK: null,
    ragOnly: false,
    activeMode: "chat",
    codeResults: [],
    codeFiles: [],
    codeQuery: "",
    codeGraph: null,
    codeGraphFunction: "",
    codeSearchLoading: false,
    codeRepositories: [],
    codeRepositoriesLoaded: false,
    codeRepositoriesLoading: false,
    codeIndexing: false,
    codeIndexStatus: "",
    selectedCodeRepositoryId: "",
    codeError: "",
    authToken: "",
    currentUser: null,
    passwordChangeOpen: false,
    sessions: [],
    activeSessionId: "",
    sessionFilter: "",
    renamingSessionId: "",
    adminUsers: [],
    llmSettings: null,
    currentView: "chat",
    health: null,
  });
}
