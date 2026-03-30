#!/bin/bash
# Hook: validate submission before popcorn submit
# Fires on PreToolUse for Bash(popcorn submit *)
#
# 1. Fast regex checks (instant, blocks on failure)
# 2. Runs claude -p for anti-cheat review (blocking — waits for verdict)

set -euo pipefail

# Read hook input from stdin
INPUT=$(cat)
COMMAND=$(echo "$INPUT" | jq -r '.tool_input.command // empty' 2>/dev/null)

if [ -z "$COMMAND" ]; then
    exit 0
fi

# Extract the submission filename
FILE=$(echo "$COMMAND" | grep -oP '(?<=\s)\S+\.py\b' | tail -1)

if [ -z "$FILE" ]; then
    exit 0
fi

if [ ! -f "$FILE" ]; then
    FILE="$CLAUDE_PROJECT_DIR/$FILE" 2>/dev/null || true
fi

if [ ! -f "$FILE" ]; then
    exit 0
fi

BASENAME=$(basename "$FILE" .py)

# ============================================================
# FAST CHECKS (instant, block on failure)
# ============================================================

if grep -q 'stream' "$FILE"; then
    echo "BLOCKED: File contains banned word 'stream' (case-sensitive)." >&2
    echo "Use chr(115)+chr(116)+'ream' in Python, or 0 for HIP queue args." >&2
    exit 2
fi

if ! head -2 "$FILE" | grep -q '^#!POPCORN'; then
    echo "WARNING: Missing #!POPCORN headers in first 2 lines." >&2
fi

if grep -qP 'data_ptr.*\bcache\b|\blru_cache\b.*tensor|global.*\bptr\b.*\bsaved\b' "$FILE"; then
    echo "WARNING: Possible cross-invocation caching detected." >&2
fi

# ============================================================
# LLM ANTI-CHEAT REVIEW (blocking — waits for verdict)
# ============================================================

VALID_FILE="${FILE%.py}.valid"

# Skip if already validated and newer than source
if [ -f "$VALID_FILE" ] && [ "$VALID_FILE" -nt "$FILE" ]; then
    VERDICT=$(head -1 "$VALID_FILE")
    echo "Previously validated: $VERDICT (see $VALID_FILE)" >&2
    if [ "$VERDICT" = "FAIL" ]; then
        echo "BLOCKED: Previous anti-cheat review failed. Fix issues and delete $VALID_FILE to re-validate." >&2
        exit 2
    fi
    exit 0
fi

echo "Running anti-cheat review via claude -p ..." >&2

claude -p "You are reviewing a GPU kernel competition submission for anti-cheat compliance.

Read the anti-cheat policy from these two URLs:
1. https://gist.githubusercontent.com/hargup/4897f33df0d2425ac4c8c99dc8f6ec00
2. https://deep-reinforce.com/defense_kernel_hack.html

Additional rules specific to this competition:
- The word 'stream' must not appear in the submission (server rejects it).
- Leaderboard mode uses recheck=True which regenerates data each call.
- Within-call preprocessing, shape-aware code emission, and pre-allocated output buffers are ALLOWED.
- JIT warmup at import time is ALLOWED.

Review this submission file against those policies and report:
- Any anti-cheat violations (FAIL)
- Any suspicious patterns that might be unintentional violations (WARN)
- If clean, say PASS

Write your verdict as the FIRST LINE: PASS, WARN, or FAIL.
Then explain your reasoning briefly.

File: $FILE
$(cat "$FILE")
" > "$VALID_FILE" 2>/dev/null

# Check the verdict
if [ ! -f "$VALID_FILE" ] || [ ! -s "$VALID_FILE" ]; then
    echo "WARNING: Anti-cheat review produced no output. Allowing submit." >&2
    exit 0
fi

VERDICT=$(head -1 "$VALID_FILE")
echo "Anti-cheat verdict: $VERDICT (see $VALID_FILE)" >&2

if [ "$VERDICT" = "FAIL" ]; then
    echo "BLOCKED: Anti-cheat review failed." >&2
    cat "$VALID_FILE" >&2
    exit 2
fi

exit 0
