// Minimal DOM helpers shared by the views. No framework, no build step.

const SVG_NS = "http://www.w3.org/2000/svg";

export const byId = (id) => document.getElementById(id);

export function clear(node) {
  while (node && node.firstChild) node.removeChild(node.firstChild);
}

/**
 * Create an element.
 * @param {string} tag
 * @param {object} props - class | text | html | hidden | disabled | value |
 *                         on<Event> handlers | any other attribute
 * @param {Array|Node|string} children
 */
export function el(tag, props = {}, children = []) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(props)) {
    if (value == null || value === false) continue;
    if (key === "class") node.className = value;
    else if (key === "text") node.textContent = value;
    else if (key === "html") node.innerHTML = value;
    else if (key === "hidden") node.hidden = Boolean(value);
    else if (key === "disabled") node.disabled = Boolean(value);
    else if (key === "value") node.value = value;
    else if (key.startsWith("on") && typeof value === "function") {
      node.addEventListener(key.slice(2).toLowerCase(), value);
    } else {
      node.setAttribute(key, value === true ? "" : value);
    }
  }
  const kids = Array.isArray(children) ? children : [children];
  for (const child of kids) {
    if (child == null || child === false) continue;
    node.append(child.nodeType ? child : document.createTextNode(String(child)));
  }
  return node;
}

// Build an inline SVG icon that references the sprite embedded in index.html.
export function icon(name) {
  const svg = document.createElementNS(SVG_NS, "svg");
  svg.setAttribute("class", "icon");
  const use = document.createElementNS(SVG_NS, "use");
  use.setAttribute("href", `#i-${name}`);
  svg.appendChild(use);
  return svg;
}
