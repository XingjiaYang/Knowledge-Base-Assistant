// Lightweight, dependency-free Markdown renderer for assistant answers.
// Supports fenced code, unordered/ordered lists, inline code, and bold.

export function escapeHtml(value) {
  return String(value || "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function renderInline(value) {
  return String(value || "")
    .split(/(`[^`]+`)/g)
    .map((part) => {
      if (part.startsWith("`") && part.endsWith("`")) {
        return `<code>${escapeHtml(part.slice(1, -1))}</code>`;
      }
      return escapeHtml(part)
        .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
        .replace(/__([^_]+)__/g, "<strong>$1</strong>");
    })
    .join("");
}

export function renderMarkdown(value) {
  const lines = String(value || "").replace(/\r\n/g, "\n").split("\n");
  const html = [];
  let inCode = false;
  let codeLines = [];
  let listType = "";

  const closeList = () => {
    if (!listType) return;
    html.push(`</${listType}>`);
    listType = "";
  };
  const openList = (type) => {
    if (listType === type) return;
    closeList();
    listType = type;
    html.push(`<${type}>`);
  };
  const closeCode = () => {
    html.push(`<pre><code>${escapeHtml(codeLines.join("\n"))}</code></pre>`);
    codeLines = [];
    inCode = false;
  };

  for (const line of lines) {
    const trimmed = line.trim();
    if (trimmed.startsWith("```")) {
      closeList();
      if (inCode) closeCode();
      else {
        inCode = true;
        codeLines = [];
      }
      continue;
    }
    if (inCode) {
      codeLines.push(line);
      continue;
    }
    if (!trimmed) {
      closeList();
      continue;
    }

    const bullet = line.match(/^\s*[-*]\s+(.+)$/);
    if (bullet) {
      openList("ul");
      html.push(`<li>${renderInline(bullet[1])}</li>`);
      continue;
    }
    const numbered = line.match(/^\s*\d+\.\s+(.+)$/);
    if (numbered) {
      openList("ol");
      html.push(`<li>${renderInline(numbered[1])}</li>`);
      continue;
    }

    closeList();
    html.push(`<p>${renderInline(line)}</p>`);
  }

  closeList();
  if (inCode) closeCode();
  return html.join("");
}
