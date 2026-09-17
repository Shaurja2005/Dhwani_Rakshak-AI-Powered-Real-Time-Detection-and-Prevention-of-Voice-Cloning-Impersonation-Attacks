// App shell: hash routing, settings (API key + tenant), toasts.

import { getSettings, saveSettings } from "./lib/api.js";
import { legend } from "./lib/components.js";
import * as admin from "./views/admin.js";
import * as demo from "./views/demo.js";
import * as forensics from "./views/forensics.js";
import * as live from "./views/live.js";

const routes = { live, forensics, admin, demo };
const view = document.getElementById("view");
let cleanup = null;

export function toast(message, kind = "info") {
  const el = document.createElement("div");
  el.className = `toast ${kind}`;
  el.setAttribute("role", kind === "error" ? "alert" : "status");
  el.textContent = message;
  document.getElementById("toasts").appendChild(el);
  setTimeout(() => el.remove(), kind === "error" ? 8000 : 4000);
}

function route() {
  const [name, query] = (location.hash.replace(/^#\/?/, "") || "live").split("?");
  const page = routes[name] ? name : "live";
  document.querySelectorAll("nav a").forEach((a) => a.setAttribute("aria-current", a.dataset.route === page ? "page" : "false"));
  if (typeof cleanup === "function") cleanup();
  view.innerHTML = "";
  if (!getSettings().apiKey) {
    openSettings();
    return;
  }
  cleanup = routes[page].mount(view, { toast, params: new URLSearchParams(query ?? "") });
}

function openSettings() {
  const dlg = document.getElementById("settings");
  const s = getSettings();
  dlg.querySelector("[name=apiKey]").value = s.apiKey;
  dlg.querySelector("[name=tenant]").value = s.tenant;
  dlg.showModal();
}

document.getElementById("settings-form").addEventListener("submit", (e) => {
  const f = new FormData(e.target);
  saveSettings({ apiKey: f.get("apiKey"), tenant: f.get("tenant") });
  route();
});
document.getElementById("open-settings").addEventListener("click", openSettings);
document.getElementById("legend").innerHTML = legend();
window.addEventListener("hashchange", route);
route();
