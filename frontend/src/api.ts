// Thin API client for an edge device. The base URL is configurable so the same
// dashboard can point at edge-A, edge-B, or a remote host.

export interface Memory {
  id: string;
  text: string;
  device_id: string;
  version: number;
  sensitivity: string;
  importance: number;
  sync_state: string;
  policy_reason: string;
  access_count: number;
  updated_at: number;
  deleted: boolean;
  metadata: Record<string, string>;
}

export interface SearchHit {
  memory: Memory;
  score: number;
  source_shard: string;
}

export interface SearchResponse {
  hits: SearchHit[];
  latency_ms: number;
  query: string;
  used_hybrid: boolean;
}

export interface SyncStatus {
  device_id: string;
  online: boolean;
  pending: number;
  synced: number;
  local_only: number;
  conflicts: number;
  failed: number;
  last_sync_at: number | null;
  last_error: string | null;
}

export interface ActivityEvent {
  type: string;
  message: string;
  at: number;
  data: Record<string, unknown>;
}

export interface AskResponse {
  answer: string;
  citations: SearchHit[];
  used_llm: boolean;
}

export class EdgeApi {
  constructor(public base: string) {}

  private async json<T>(path: string, init?: RequestInit): Promise<T> {
    const res = await fetch(`${this.base}${path}`, {
      headers: { "Content-Type": "application/json" },
      ...init,
    });
    if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
    return res.json() as Promise<T>;
  }

  listMemories(includeDeleted = false) {
    return this.json<Memory[]>(`/memories?include_deleted=${includeDeleted}&limit=200`);
  }
  ingest(text: string, metadata: Record<string, string> = {}) {
    return this.json<Memory>(`/memories`, {
      method: "POST",
      body: JSON.stringify({ text, metadata }),
    });
  }
  update(id: string, text: string) {
    return this.json<Memory>(`/memories/${id}`, {
      method: "PATCH",
      body: JSON.stringify({ text }),
    });
  }
  remove(id: string) {
    return this.json<Memory>(`/memories/${id}`, { method: "DELETE" });
  }
  search(query: string, hybrid: boolean) {
    return this.json<SearchResponse>(`/search`, {
      method: "POST",
      body: JSON.stringify({ query, hybrid, limit: 10 }),
    });
  }
  ask(question: string) {
    return this.json<AskResponse>(`/ask`, {
      method: "POST",
      body: JSON.stringify({ question, top_k: 5 }),
    });
  }
  status() {
    return this.json<SyncStatus>(`/sync/status`);
  }
  syncNow() {
    return this.json<Record<string, number>>(`/sync/now`, { method: "POST" });
  }
  setOffline(offline: boolean) {
    return this.json<{ forced_offline: boolean }>(`/sync/offline`, {
      method: "POST",
      body: JSON.stringify({ offline }),
    });
  }
  activity() {
    return this.json<ActivityEvent[]>(`/activity?limit=100`);
  }
  wsUrl() {
    return this.base.replace(/^http/, "ws") + "/ws";
  }
}
