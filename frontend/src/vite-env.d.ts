/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_EDGE_A_URL?: string;
  readonly VITE_EDGE_B_URL?: string;
  readonly VITE_CLOUD_URL?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
