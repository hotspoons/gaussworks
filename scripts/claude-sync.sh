#!/usr/bin/env bash
# Keep Claude Code's state (~/.claude, ~/.claude.json) alive across container
# restarts: the home dir is ephemeral, /workspace is the PVC.
#
#   claude-sync.sh restore   # if ~/.claude is missing/empty, copy the backup back
#   claude-sync.sh sync      # one rsync home -> PVC (refuses if home looks empty)
#   claude-sync.sh start     # restore, then loop `sync` every $CLAUDE_SYNC_INTERVAL s (default 300)
#
# Call `start` from the pod init script; it daemonises itself and is idempotent.
set -euo pipefail
SRC="$HOME/.claude"; SRC_JSON="$HOME/.claude.json"
DST=${CLAUDE_BACKUP_DIR:-/workspace/.claude-backup}
INTERVAL=${CLAUDE_SYNC_INTERVAL:-300}
LOCK=/tmp/claude-sync.pid
log() { echo "[claude-sync] $(date +%FT%T) $*"; }

nonempty() { [ -d "$1" ] && [ -n "$(ls -A "$1" 2>/dev/null)" ]; }

restore() {
    if nonempty "$SRC/projects" || nonempty "$SRC/sessions"; then
        log "home state present, no restore"; return 0
    fi
    if nonempty "$DST/.claude"; then
        mkdir -p "$SRC"
        rsync -a "$DST/.claude/" "$SRC/"
        [ -f "$DST/.claude.json" ] && [ ! -f "$SRC_JSON" ] && cp -p "$DST/.claude.json" "$SRC_JSON"
        log "restored $(du -sh "$SRC" | cut -f1) from $DST"
    else
        log "nothing to restore ($DST empty)"
    fi
}

sync_once() {
    # never let an empty/fresh home wipe the backup
    if ! nonempty "$SRC/projects" && ! nonempty "$SRC/sessions"; then
        log "home looks empty, skipping sync"; return 0
    fi
    mkdir -p "$DST/.claude"
    rsync -a --delete \
        --exclude 'debug/' --exclude 'shell-snapshots/' --exclude 'ide/' \
        --exclude 'session-env/' --exclude '*.lock' \
        "$SRC/" "$DST/.claude/"
    [ -f "$SRC_JSON" ] && cp -p "$SRC_JSON" "$DST/.claude.json"
    date +%s > "$DST/.last-sync"
}

case "${1:-}" in
    restore) restore ;;
    sync)    sync_once; log "synced" ;;
    start)
        restore
        if [ -f "$LOCK" ] && kill -0 "$(cat "$LOCK")" 2>/dev/null; then
            log "daemon already running (pid $(cat "$LOCK"))"; exit 0
        fi
        setsid nohup bash -c "
            echo \$\$ > '$LOCK'
            while true; do '$0' sync >> /workspace/logs/claude-sync.log 2>&1 || true; sleep $INTERVAL; done" \
            >/dev/null 2>&1 < /dev/null &
        sleep 0.5; log "daemon started (pid $(cat "$LOCK"), every ${INTERVAL}s -> $DST)"
        ;;
    *) echo "usage: $0 restore|sync|start"; exit 2 ;;
esac
