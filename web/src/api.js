const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";

export async function ask(question) {
  const r = await fetch(`${API_BASE}/chat/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });
  if (!r.ok) {
    const t = await r.text();
    throw new Error(`API error: ${r.status} ${t}`);
  }
  return r.json();
}
