#!/usr/bin/env bash
#
# Feature 5 Integration Test: Age & Decay Rate Processing (-D <interval>)
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BIN="$PROJECT_ROOT/bin/geif"
TMP_DIR="$(mktemp -d /tmp/geif_test_decay_XXXXXX)"
trap 'rm -rf "$TMP_DIR"' EXIT

WINE_CSV="/home/timo_savinen_elisa_fi/docs/tools/geif/prod_plan/winequality-red.csv"

echo "================================================================="
echo "FEATURE 5 TEST: Age & Decay Rate Processing (-D <interval>)"
echo "================================================================="

# Step 1: Train reference model with -C 12
echo "Step 1: Train reference multi-category model with -C 12..."
$BIN -l "$WINE_CSV" -H -f ';' -C 12 -t 15 -s 480 -w "$TMP_DIR/model_ref.json"

# Verify updated timestamps exist in model summary
echo "Step 2: Inspect model summary (-r -q) for updated timestamps..."
SUMMARY=$($BIN -r "$TMP_DIR/model_ref.json" -q)
echo "$SUMMARY"
if ! echo "$SUMMARY" | grep -q "updated="; then
    echo "ERROR: Model summary missing updated timestamp!"
    exit 1
fi
echo "  [PASS] Sub-forests report valid updated epoch timestamps."

# Step 3: Standalone Age Pruning with -D <interval>
echo "Test 3: Standalone Age Pruning (-r ... -D 7d -w ...)..."
# Create a modified model where categories '3' and '4' have artificial timestamps 15 days ago
NOW=$(date +%s)
FIFTEEN_DAYS_AGO=$((NOW - 15 * 86400))

# Use python or jq/sed to inject older timestamp for category 3 and 4
python3 -c "
import json
with open('$TMP_DIR/model_ref.json', 'r') as f:
    data = json.load(f)
for forest in data.get('forests', []):
    if forest.get('category') in ['3', '4']:
        forest['last_updated'] = $FIFTEEN_DAYS_AGO
with open('$TMP_DIR/model_aged.json', 'w') as f:
    json.dump(data, f, indent=2)
"

# Run age pruning with -D 7d (drop categories older than 7 days)
$BIN -r "$TMP_DIR/model_aged.json" -D 7d -w "$TMP_DIR/model_pruned_7d.json"

PRUNED_SUMMARY=$($BIN -r "$TMP_DIR/model_pruned_7d.json" -q)
echo "$PRUNED_SUMMARY"

# Sub-forests 3 and 4 should be pruned; 5, 6, 7, 8 should remain (4 sub-forests)
SUB_COUNT=$(echo "$PRUNED_SUMMARY" | grep "Sub-forests:" | awk '{print $2}')
if [ "$SUB_COUNT" -ne 4 ]; then
    echo "ERROR: Expected 4 sub-forests after -D 7d pruning, got $SUB_COUNT!"
    exit 1
fi
if echo "$PRUNED_SUMMARY" | grep -q "Sub-forest '3'"; then
    echo "ERROR: Sub-forest '3' should have been pruned!"
    exit 1
fi
if echo "$PRUNED_SUMMARY" | grep -q "Sub-forest '4'"; then
    echo "ERROR: Sub-forest '4' should have been pruned!"
    exit 1
fi
echo "  [PASS] -D 7d correctly dropped stale sub-forests '3' and '4', retaining 4 active sub-forests."

# Test 4: Incremental Learning with Age Pruning (-r -l -D -w)
echo "Test 4: Incremental Learning with Age Refresh and Pruning (-r -l -D 7d -w)..."
# Start with model_aged where 3 and 4 are 15 days old, and 5, 6, 7, 8 are fresh.
# We now feed new data that contains ONLY category '3' rows.
# Category 3 should be refreshed to NOW, while category 4 remains 15 days old.
awk -F';' '$12 == "3" { print; if (++n == 5) exit }' "$WINE_CSV" > "$TMP_DIR/cat3_fresh.csv"

# Run incremental training with -r, -l, -D 7d, -w
$BIN -r "$TMP_DIR/model_aged.json" -l "$TMP_DIR/cat3_fresh.csv" -f ';' -D 7d -w "$TMP_DIR/model_cat3_refreshed.json" -v

REFRESHED_SUMMARY=$($BIN -r "$TMP_DIR/model_cat3_refreshed.json" -q)
echo "$REFRESHED_SUMMARY"

# Category 3 should now be RETAINED (because refreshed to now), Category 4 should be PRUNED!
# Total sub-forests should be 5 (3, 5, 6, 7, 8)
REFRESHED_COUNT=$(echo "$REFRESHED_SUMMARY" | grep "Sub-forests:" | awk '{print $2}')
if [ "$REFRESHED_COUNT" -ne 5 ]; then
    echo "ERROR: Expected 5 sub-forests (cat 3 refreshed, cat 4 pruned), got $REFRESHED_COUNT!"
    exit 1
fi
if ! echo "$REFRESHED_SUMMARY" | grep -q "Sub-forest '3'"; then
    echo "ERROR: Sub-forest '3' should have been retained after being refreshed!"
    exit 1
fi
if echo "$REFRESHED_SUMMARY" | grep -q "Sub-forest '4'"; then
    echo "ERROR: Sub-forest '4' was not fed and is older than 7d; should have been pruned!"
    exit 1
fi
echo "  [PASS] Incremental training successfully refreshed category '3' timestamp and pruned stale category '4'."

# Test 5: Verify various time format strings (-D 30d, -D 24h, -D 3600s, -D 2w)
echo "Test 5: Verify diverse time unit syntax parsing..."
$BIN -r "$TMP_DIR/model_aged.json" -D 24h -w "$TMP_DIR/model_24h.json"
$BIN -r "$TMP_DIR/model_aged.json" -D 3600s -w "$TMP_DIR/model_3600s.json"
$BIN -r "$TMP_DIR/model_aged.json" -D 2w -w "$TMP_DIR/model_2w.json"
$BIN -r "$TMP_DIR/model_aged.json" -D 30 -w "$TMP_DIR/model_30days.json"
echo "  [PASS] Time unit suffixes ('h', 's', 'w', 'd', bare integer) parsed without errors."

# Test 6: Scoring against model pruned by age flags unknown category as outlier
echo "Test 6: Score against aged-out category (triggers -N outlier)..."
awk -F';' '$12 == "4" { print; if (++n == 1) exit }' "$WINE_CSV" > "$TMP_DIR/query_cat4.csv"

set +e
$BIN -r "$TMP_DIR/model_pruned_7d.json" -a "$TMP_DIR/query_cat4.csv" -f ';' -N "NEW;cat=%c;outlier=%o" -o "$TMP_DIR/cat4_out.csv"
EXIT_CODE=$?
set -e

if [ "$EXIT_CODE" -ne 2 ]; then
    echo "ERROR: Expected exit code 2 for aged-out category, got $EXIT_CODE!"
    exit 1
fi
OUT_LINE=$(cat "$TMP_DIR/cat4_out.csv")
if [ "$OUT_LINE" != "NEW;cat=4;outlier=1" ]; then
    echo "ERROR: Unexpected format for aged-out category: '$OUT_LINE'!"
    exit 1
fi
echo "  [PASS] Queries against pruned/aged-out categories correctly alert via -N with exit code 2."

echo ">>> ALL FEATURE 5 TESTS PASSED SUCCESSFULLY! <<<"
