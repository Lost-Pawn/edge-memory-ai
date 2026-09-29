import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  ActivityEvent,
  AskResponse,
  EdgeApi,
  Memory,
  SearchResponse,
  SyncStatus,
} from "./api";
import { DEVICES } from "./config";

type Tab = "memories" | "search" | "ask" | "sync" | "activity";

export default function App() {
  const [deviceIdx, setDeviceIdx] = useState(0);
  const api = useMemo(() => new EdgeApi(DEVICES[deviceIdx].url), [deviceIdx]);
  const [tab, setTab] = useState<Tab>("memories");
  const [status, setStatus] = useState<SyncStatus | null>(null);
  const [events, setEvents] = useState<ActivityEvent[]>([]);

  // Poll status; stream activity over WebSocket.
  useEffect(() => {
    let alive = true;
    const poll = async () => {
      try {
        if (alive) setStatus(await api.status());
      } catch {
        if (alive) setStatus(null);
      }
    };
    poll();
    const t = setInterval(poll, 3000);
    return () => {
      alive = false;
      clearInterval(t);
    };
  }, [api]);

  useEffect(() => {
    setEvents([]);
    const ws = new WebSocket(api.wsUrl());
    ws.onmessage = (m) =>
      setEvents((prev) => [JSON.parse(m.data) as ActivityEvent, ...prev].slice(0, 200));
    return () => ws.close();
  }, [api]);

  return (
    <div className="app">
      <header>
        <div className="brand">
          <span className="logo">◆</span> EdgeMemoryAi
          <span className="sub">offline-first edge memory · Qdrant Edge</span>
        </div>
        <div className="device-picker">
          {DEVICES.map((d, i) => (
            <button
              key={d.name}
              className={i === deviceIdx ? "chip active" : "chip"}
              onClick={() => setDeviceIdx(i)}
            >
              {d.name}
            </button>
          ))}
          <StatusPill status={status} />
        </div>
      </header>

      <nav>
        {(["memories", "search", "ask", "sync", "activity"] as Tab[]).map((t) => (
          <button key={t} className={t === tab ? "tab active" : "tab"} onClick={() => setTab(t)}>
            {t}
          </button>
        ))}
      </nav>

      <main>
        {tab === "memories" && <Memories api={api} />}
        {tab === "search" && <Search api={api} />}
        {tab === "ask" && <Ask api={api} />}
        {tab === "sync" && <Sync api={api} status={status} />}
        {tab === "activity" && <Activity events={events} />}
      </main>
    </div>
  );
}

function StatusPill({ status }: { status: SyncStatus | null }) {
  if (!status) return <span className="pill down">unreachable</span>;
  return (
    <span className={status.online ? "pill up" : "pill offline"}>
      {status.online ? "online" : "offline"} · {status.pending} queued
    </span>
  );
}

function Memories({ api }: { api: EdgeApi }) {
  const [items, setItems] = useState<Memory[]>([]);
  const [text, setText] = useState("");
  const load = useCallback(() => api.listMemories().then(setItems).catch(() => {}), [api]);
  useEffect(() => {
    load();
  }, [load]);

  const add = async () => {
    if (!text.trim()) return;
    await api.ingest(text.trim());
    setText("");
    load();
  };

  return (
    <section>
      <div className="row">
        <input
          value={text}
          placeholder="Add a memory (try one with an email to see the PII policy)…"
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && add()}
        />
        <button className="primary" onClick={add}>
          Store
        </button>
      </div>
      <div className="cards">
        {items.map((m) => (
          <div className="card" key={m.id}>
            <div className="card-head">
              <span className={`tag ${m.sync_state}`}>{m.sync_state}</span>
              <span className={`tag sens-${m.sensitivity}`}>{m.sensitivity}</span>
              <span className="imp">imp {m.importance.toFixed(2)}</span>
              <button className="x" onClick={() => api.remove(m.id).then(load)}>
                ✕
              </button>
            </div>
            <div className="text">{m.text}</div>
            <div className="reason">{m.policy_reason}</div>
          </div>
        ))}
        {items.length === 0 && <p className="empty">No memories yet.</p>}
      </div>
    </section>
  );
}

function Search({ api }: { api: EdgeApi }) {
  const [q, setQ] = useState("");
  const [hybrid, setHybrid] = useState(true);
  const [res, setRes] = useState<SearchResponse | null>(null);
  const run = async () => q.trim() && setRes(await api.search(q.trim(), hybrid));
  return (
    <section>
      <div className="row">
        <input
          value={q}
          placeholder="Search local memory…"
          onChange={(e) => setQ(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && run()}
        />
        <label className="toggle">
          <input type="checkbox" checked={hybrid} onChange={(e) => setHybrid(e.target.checked)} />
          hybrid
        </label>
        <button className="primary" onClick={run}>
          Search
        </button>
      </div>
      {res && (
        <>
          <p className="meta">
            {res.hits.length} hits · {res.latency_ms} ms · {res.used_hybrid ? "dense+sparse (RRF)" : "dense only"}
          </p>
          <div className="cards">
            {res.hits.map((h) => (
              <div className="card" key={h.memory.id}>
                <div className="card-head">
                  <span className="score">{h.score.toFixed(4)}</span>
                  <span className="tag">{h.source_shard}</span>
                </div>
                <div className="text">{h.memory.text}</div>
              </div>
            ))}
          </div>
        </>
      )}
    </section>
  );
}

function Ask({ api }: { api: EdgeApi }) {
  const [q, setQ] = useState("");
  const [res, setRes] = useState<AskResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const run = async () => {
    if (!q.trim()) return;
    setLoading(true);
    try {
      setRes(await api.ask(q.trim()));
    } finally {
      setLoading(false);
    }
  };
  return (
    <section>
      <div className="row">
        <input
          value={q}
          placeholder="Ask a question over local memory…"
          onChange={(e) => setQ(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && run()}
        />
        <button className="primary" onClick={run} disabled={loading}>
          {loading ? "…" : "Ask"}
        </button>
      </div>
      {res && (
        <div className="answer">
          <div className="answer-body">{res.answer}</div>
          <div className="meta">{res.used_llm ? "generated by local LLM" : "extractive (no LLM loaded)"}</div>
          <div className="cards">
            {res.citations.map((h, i) => (
              <div className="card" key={h.memory.id}>
                <span className="tag">[{i + 1}]</span> {h.memory.text}
              </div>
            ))}
          </div>
        </div>
      )}
    </section>
  );
}

function Sync({ api, status }: { api: EdgeApi; status: SyncStatus | null }) {
  const [busy, setBusy] = useState(false);
  const offline = status ? !status.online : false;
  const toggle = async () => {
    setBusy(true);
    try {
      await api.setOffline(!offline);
    } finally {
      setBusy(false);
    }
  };
  return (
    <section>
      <div className="stats">
        <Stat label="pending" value={status?.pending ?? "–"} cls="pending" />
        <Stat label="synced" value={status?.synced ?? "–"} cls="synced" />
        <Stat label="local-only" value={status?.local_only ?? "–"} cls="local_only" />
        <Stat label="conflicts" value={status?.conflicts ?? "–"} cls="conflict" />
        <Stat label="failed" value={status?.failed ?? "–"} cls="failed" />
      </div>
      <div className="row">
        <button className="primary" onClick={() => api.syncNow()}>
          Sync now
        </button>
        <button className={offline ? "warn" : "ghost"} onClick={toggle} disabled={busy}>
          {offline ? "Go online" : "Simulate offline"}
        </button>
      </div>
      {status?.last_error && <p className="err">last error: {status.last_error}</p>}
    </section>
  );
}

function Stat({ label, value, cls }: { label: string; value: number | string; cls: string }) {
  return (
    <div className={`stat ${cls}`}>
      <div className="v">{value}</div>
      <div className="l">{label}</div>
    </div>
  );
}

function Activity({ events }: { events: ActivityEvent[] }) {
  const ref = useRef<HTMLDivElement>(null);
  return (
    <section>
      <div className="log" ref={ref}>
        {events.map((e, i) => (
          <div className={`log-row ${e.type}`} key={i}>
            <span className="log-type">{e.type}</span>
            <span className="log-msg">{e.message}</span>
            <span className="log-time">{new Date(e.at * 1000).toLocaleTimeString()}</span>
          </div>
        ))}
        {events.length === 0 && <p className="empty">Waiting for events…</p>}
      </div>
    </section>
  );
}
