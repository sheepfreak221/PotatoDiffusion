#!/usr/bin/env bash
# PotatoDiffusion starten
cd "$(dirname "$0")"
exec python3 app.py "$@"
