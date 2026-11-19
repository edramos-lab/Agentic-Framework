/**
 * Thin API client for the PaperOrchestra backend.
 */

const BASE = "/api";

async function request(path, opts = {}) {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json", ...opts.headers },
    ...opts,
  });
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(`${res.status} ${res.statusText}: ${body}`);
  }
  if (res.status === 204) return null;
  return res.json();
}

// --- Keys -----------------------------------------------------------------

export const listKeys = () => request("/keys");
export const setKey = (name, value) =>
  request("/keys", {
    method: "POST",
    body: JSON.stringify({ name, value }),
  });
export const deleteKey = (name) =>
  request(`/keys/${encodeURIComponent(name)}`, { method: "DELETE" });

// --- Uploads --------------------------------------------------------------

export const listUploads = () => request("/uploads");
export const uploadFiles = async (fileList) => {
  const form = new FormData();
  for (const f of fileList) form.append("files", f);
  const res = await fetch(`${BASE}/uploads`, { method: "POST", body: form });
  if (!res.ok) throw new Error(`Upload failed: ${res.statusText}`);
  return res.json();
};

// --- Pipeline -------------------------------------------------------------

export const startPipeline = (payload) =>
  request("/pipeline/start", {
    method: "POST",
    body: JSON.stringify(payload),
  });

export const listRuns = () => request("/pipeline/runs");
export const getRun = (id) => request(`/pipeline/runs/${id}`);
export const listArtifacts = (id) => request(`/pipeline/runs/${id}/artifacts`);

export function connectPipelineWS(runId, onEvent) {
  const proto = window.location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${window.location.host}/api/pipeline/ws/${runId}`);
  ws.onmessage = (e) => {
    try {
      onEvent(JSON.parse(e.data));
    } catch {
      /* ignore parse errors */
    }
  };
  ws.onerror = () => onEvent({ type: "error", message: "WebSocket error" });
  ws.onclose = () => onEvent({ type: "done" });
  return ws;
}
