#!/usr/bin/env bash
# Dev helper: puts every review-tray garment on the connected tablet back to CAPTURED, restarts the app so
# batch processing runs again, waits, and prints the results. Uses only adb + Python's sqlite3.
# Usage (repo root):  bash scripts/reprocess-on-tablet.sh [wait-seconds]
set -euo pipefail
export MSYS_NO_PATHCONV=1
ADB="${ADB:-$HOME/AppData/Local/Android/Sdk/platform-tools/adb.exe}"
PKG=dev.wardrobe.app
WAIT="${1:-90}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
# Path as the (Windows) Python sees it.
DB="$(cygpath -w "$TMP/wardrobe.db" 2>/dev/null || echo "$TMP/wardrobe.db")"

pull_db() {
  rm -f "$TMP"/wardrobe.db*
  for f in wardrobe.db wardrobe.db-wal wardrobe.db-shm; do
    "$ADB" exec-out run-as $PKG cat databases/$f > "$TMP/$f" 2>/dev/null || true
  done
}

"$ADB" shell am force-stop $PKG
pull_db
python - "$DB" <<'EOF'
import sqlite3, sys
c = sqlite3.connect(sys.argv[1])
c.execute("update capture_entry set status='CAPTURED', cutoutPathsJson=NULL, suggestedTagsJson=NULL")
c.commit()
c.execute("PRAGMA wal_checkpoint(TRUNCATE)")
c.execute("PRAGMA journal_mode=DELETE")
c.close()
EOF
"$ADB" push "$DB" /data/local/tmp/wardrobe.db >/dev/null
"$ADB" shell "run-as $PKG sh -c 'cp /data/local/tmp/wardrobe.db databases/wardrobe.db && rm -f databases/wardrobe.db-wal databases/wardrobe.db-shm'"
"$ADB" shell rm /data/local/tmp/wardrobe.db
"$ADB" logcat -c
"$ADB" shell am start -W -n $PKG/.MainActivity >/dev/null
echo "processing... (waiting ${WAIT}s)"
sleep "$WAIT"
"$ADB" logcat -d | grep -E "CaptureProcessor|ProcessCaptureWorker|AndroidRuntime|OutOfMemory" | tail -20 || true
pull_db
python - "$DB" <<'EOF'
import json, sqlite3, sys
c = sqlite3.connect(sys.argv[1])
for i, status, cut, tags in c.execute("select id, status, cutoutPathsJson, suggestedTagsJson from capture_entry"):
    print(f"entry {i}: {status}")
    if cut:
        d = json.loads(cut)
        for v, info in d.get("views", {}).items():
            print(f"  {v}: {info}")
        for piece, views in d.get("pieces", {}).items():
            print(f"  piece {piece}: " + ", ".join(f"{v} area {x['areaFrac']:.3f} align {x['alignErrorPx']:.0f}px" for v, x in views.items()))
    if tags:
        for piece, t in json.loads(tags).items():
            summary = {k: (v.get("value") if k != "colour" else [c["name"] for c in v["value"]]) for k, v in t.items() if isinstance(v, dict)}
            print(f"  tags {piece}: {summary}")
EOF
