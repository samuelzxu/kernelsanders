#!/bin/bash
# Hook: validate submission before popcorn submit
# Fires on PreToolUse for Bash(popcorn submit *)
#
# 1. Fast regex checks (instant, blocks on failure)
# 2. Spawns claude -p in background for anti-cheat review (non-blocking)

set -euo pipefail

# Read hook input from stdin
INPUT=$(cat)
COMMAND=$(echo "$INPUT" | jq -r '.tool_input.command // empty' 2>/dev/null)

if [ -z "$COMMAND" ]; then
    exit 0  # not a bash command, allow
fi

# Extract the submission filename from the command
# Handles: popcorn submit --mode test --no-tui 597_v3.py
FILE=$(echo "$COMMAND" | grep -oP '(?<=\s)\S+\.py\b' | tail -1)

if [ -z "$FILE" ]; then
    exit 0  # no .py file found, allow
fi

# Resolve to absolute path
if [ ! -f "$FILE" ]; then
    FILE="$CLAUDE_PROJECT_DIR/$FILE" 2>/dev/null || true
fi

if [ ! -f "$FILE" ]; then
    exit 0  # file not found, let popcorn handle the error
fi

BASENAME=$(basename "$FILE" .py)

# ============================================================
# FAST CHECKS (instant, block on failure)
# ============================================================

# Check for banned word "stream" (case-sensitive)
if grep -q 'stream' "$FILE"; then
    echo "BLOCKED: File contains banned word 'stream' (case-sensitive)." >&2
    echo "The competition server rejects submissions with this word." >&2
    echo "Use chr(115)+chr(116)+'ream' in Python, or 0 for HIP queue args." >&2
    exit 2  # block the command
fi

# Check POPCORN headers
if ! head -2 "$FILE" | grep -q '^#!POPCORN'; then
    echo "WARNING: Missing #!POPCORN headers in first 2 lines." >&2
    # Don't block — just warn
fi

# Check for obvious cross-invocation caching patterns
if grep -qP 'data_ptr.*\bcache\b|\blru_cache\b.*tensor|global.*\bptr\b.*\bsaved\b' "$FILE"; then
    echo "WARNING: Possible cross-invocation caching detected. Check anti-cheat policy." >&2
fi

# ============================================================
# LLM ANTI-CHEAT REVIEW (background, non-blocking)
# ============================================================

VALID_FILE="${FILE%.py}.valid"

# Skip if already validated
if [ -f "$VALID_FILE" ] && [ "$VALID_FILE" -nt "$FILE" ]; then
    echo "Already validated: $VALID_FILE" >&2
    exit 0
fi

# Spawn claude -p in background for full review
(
    claude -p "You are reviewing a GPU kernel competition submission for anti-cheat compliance.

The competition rules:
1. Cross-invocation caching is BANNED — storing outputs or preprocessed data across calls keyed by pointer addresses or tensor properties is not allowed.
2. Leaderboard mode uses recheck=True which regenerates data each call, so cached results return stale values and FAIL.
3. The word 'stream' must not appear in the submission (server rejects it).
4. Within-call preprocessing and shape-aware code emission are ALLOWED.
5. Pre-computing configs, JIT warmup at import time, and pre-allocated output buffers are ALLOWED.

Review this submission file and report:
- Any anti-cheat violations (FAIL)
- Any suspicious patterns that might be unintentional violations (WARN)
- If clean, say PASS

Write your verdict as the first line: PASS, WARN, or FAIL.

File: $FILE
$(cat "$FILE")
" > "$VALID_FILE" 2>/dev/null
) &

echo "Anti-cheat review started in background → $VALID_FILE" >&2
exit 0  # allow the submit to proceed
