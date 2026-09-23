#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
exec "${PI05_CONSOLE_PYTHON:-/home/gm/data/LoongForge/env/bin/python}" service.py stop
