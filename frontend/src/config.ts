// Resolve edge device endpoints. Defaults derive from the host the dashboard is served
// from, so it works both locally and on the deployed server without rebuilds.

const host = window.location.hostname || "localhost";

export interface Device {
  name: string;
  url: string;
}

export const DEVICES: Device[] = [
  { name: "edge-A", url: import.meta.env.VITE_EDGE_A_URL ?? `http://${host}:8001` },
  { name: "edge-B", url: import.meta.env.VITE_EDGE_B_URL ?? `http://${host}:8002` },
];

export const CLOUD_URL = import.meta.env.VITE_CLOUD_URL ?? `http://${host}:8000`;
