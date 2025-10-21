import React, { useState } from "react";
import { ask } from "./api";

export default function App() {
  const [q, setQ] = useState("");
  const [answer, setAnswer] = useState("");
  const [matches, setMatches] = useState([]);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState("");

  const onSubmit = async (e) => {
    e.preventDefault();
    setErr(""); setLoading(true);
    try {
      const res = await ask(q);
      setAnswer(res.answer);
      setMatches(res.matches || []);
    } catch (e) {
      setErr(e.message || "Error");
      setAnswer(""); setMatches([]);
    } finally { setLoading(false); }
  };

  return (
    <div style={{maxWidth: 900, margin: "2rem auto", fontFamily: "Inter, system-ui, sans-serif"}}>
      <h1>Video Highlights Chat</h1>
      <form onSubmit={onSubmit} style={{display: "flex", gap: 8}}>
        <input
          value={q}
          onChange={(e)=>setQ(e.target.value)}
          placeholder='Ask: "What happened after the person got out of the car?"'
          style={{flex:1, padding: 10, fontSize: 16}}
        />
        <button disabled={loading || !q.trim()} style={{padding: "10px 16px"}}>
          {loading ? "Searching..." : "Ask"}
        </button>
      </form>

      {err && <p style={{color:"crimson"}}>{err}</p>}

      {answer && (
        <>
          <h3>Answer</h3>
          <p>{answer}</p>
        </>
      )}

      {matches.length > 0 && (
        <>
          <h3>Matched Highlights</h3>
          <ul>
            {matches.map(m => (
              <li key={m.id} style={{marginBottom: 8}}>
                <strong>Video #{m.video_id}</strong> · [{m.ts_start_sec}s–{m.ts_end_sec}s] · score {m.score.toFixed(3)}
                <div>{m.llm_summary || m.description}</div>
              </li>
            ))}
          </ul>
        </>
      )}
    </div>
  );
}
