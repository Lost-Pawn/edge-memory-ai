#!/usr/bin/env bash
#
# EdgeMemoryAi one-shot setup.
#
# Builds the images, downloads the models EXPLICITLY (embeddings + local LLM),
# and brings up the full stack: Qdrant Server + cloud brain + two edge devices +
# the React dashboard.
#
# Usage:
#   ./setup.sh            # full setup (build -> download models -> up)
#   ./setup.sh --no-llm   # skip the ~2GB GGUF LLM download (RAG falls back to extractive)
#   ./setup.sh down       # stop and remove the stack
#   ./setup.sh logs       # follow logs
#
set -euo pipefail

cd "$(dirname "$0")"

RED=$'\033[31m'; GRN=$'\033[32m'; YEL=$'\033[33m'; NC=$'\033[0m'
say() { echo "${GRN}==>${NC} $*"; }
warn() { echo "${YEL}!! ${NC} $*"; }
die() { echo "${RED}xx ${NC} $*" >&2; exit 1; }

# ---- subcommands -----------------------------------------------------------
case "${1:-up}" in
  down) exec docker compose down ;;
  logs) exec docker compose logs -f ;;
esac

WITH_LLM=1
[ "${1:-}" = "--no-llm" ] && WITH_LLM=0

# ---- preflight -------------------------------------------------------------
command -v docker >/dev/null || die "docker not found. Install Docker first."
docker compose version >/dev/null 2>&1 || die "docker compose v2 not found."
docker info >/dev/null 2>&1 || die "docker daemon not running."

# ---- build -----------------------------------------------------------------
say "Building backend + frontend images (llama-cpp compile can take a few minutes)…"
docker compose build cloud frontend

# ---- models (EXPLICIT — nothing downloads on its own) ----------------------
if [ ! -d models/models--Qdrant--clip-ViT-B-32-text ]; then
  say "Downloading embedding models (~350MB, CLIP + BM25)…"
  docker compose run --rm edge-a edge-memory download-models
else
  say "Embedding models already cached — skipping."
fi

if [ "$WITH_LLM" = "1" ]; then
  if ! ls models/llm/*.gguf >/dev/null 2>&1; then
    say "Downloading local LLM GGUF (~2GB)…"
    docker compose run --rm edge-a edge-memory download-llm
  else
    say "LLM already cached — skipping."
  fi
else
  warn "Skipping LLM download (--no-llm). RAG will use the extractive fallback."
fi

# ---- up --------------------------------------------------------------------
say "Starting the full stack…"
docker compose up -d

sleep 8
say "Health check:"
for svc in "cloud:8000" "edge-a:8001" "edge-b:8002"; do
  name=${svc%%:*}; port=${svc##*:}
  if curl -sf "http://localhost:${port}/health" >/dev/null; then
    echo "   ${GRN}ok${NC}   ${name} (:${port})"
  else
    echo "   ${RED}down${NC} ${name} (:${port})"
  fi
done
curl -sf "http://localhost:6333/healthz" >/dev/null && echo "   ${GRN}ok${NC}   qdrant (:6333)"
curl -sf "http://localhost:5173/" >/dev/null && echo "   ${GRN}ok${NC}   dashboard (:5173)"

cat <<EOF

${GRN}EdgeMemoryAi is up.${NC}
  Dashboard : http://localhost:5173
  Edge A    : http://localhost:8001/docs
  Edge B    : http://localhost:8002/docs
  Cloud     : http://localhost:8000/docs
  Qdrant    : http://localhost:6333/dashboard

Viewing from another machine: open the ports on your firewall, or SSH-tunnel:
  ssh -L 5173:localhost:5173 -L 8001:localhost:8001 -L 8002:localhost:8002 <user>@<host>

Stop with: ./setup.sh down
EOF
