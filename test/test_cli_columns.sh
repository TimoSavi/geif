#!/usr/bin/env bash
# test_cli_columns.sh - Test Suite for Feature 2: Column Selection & Row Labeling (-I, -U, -L)
set -euo pipefail

BIN="./bin/geif"
WINE_CSV="/home/timo_savinen_elisa_fi/docs/tools/geif/prod_plan/winequality-red.csv"

if [[ ! -f "$WINE_CSV" ]]; then
    echo "Error: Dataset '$WINE_CSV' not found!"
    exit 1
fi

TMP_DIR=$(mktemp -d /tmp/geif_test_cols_XXXXXX)
trap 'rm -rf "$TMP_DIR"' EXIT

echo "================================================================="
echo "FEATURE 2 TEST: Column Selection & Row Labeling (-I, -U, -L)"
echo "================================================================="

# Test 1: Ignore Column 12 (-I 12)
echo "Test 1: Train with -I 12 (Ignore column 12 - quality)..."
$BIN -l "$WINE_CSV" -H -f ';' -I 12 -w "$TMP_DIR/model_i12.json" -i 30 -s 64 -v 2>&1 | grep "11 feature dimensions"
echo "  [PASS] Forest trained with 11 features (col 12 ignored)."

# Test 2: Inspect Model Query (-q)
echo "Test 2: Query model metadata (-r -q)..."
SUMMARY=$($BIN -r "$TMP_DIR/model_i12.json" -q)
echo "$SUMMARY" | grep "Dimensions:          11"
echo "$SUMMARY" | grep "Ignore Columns (-I):  12"
echo "  [PASS] Model metadata correctly persisted: 11 dimensions, Ignore: 12."

# Test 3: Include Range (-U 2-10)
echo "Test 3: Train with -U 2-10 (Include only columns 2 through 10)..."
$BIN -l "$WINE_CSV" -H -f ';' -U 2-10 -w "$TMP_DIR/model_u.json" -i 30 -s 64 -v 2>&1 | grep "9 feature dimensions"
SUMMARY_U=$($BIN -r "$TMP_DIR/model_u.json" -q)
echo "$SUMMARY_U" | grep "Dimensions:          9"
echo "$SUMMARY_U" | grep "Include Columns (-U): 2-10"
echo "  [PASS] Range inclusion (-U 2-10) correctly trained 9 feature dimensions."

# Test 4: Combined Label (-L 1) and Ignore (-I 12)
echo "Test 4: Train with -L 1 -I 12 (Col 1 label, col 12 ignore, cols 2-11 features)..."
$BIN -l "$WINE_CSV" -H -f ';' -L 1 -I 12 -w "$TMP_DIR/model_l_i.json" -i 30 -s 64 -v 2>&1 | grep "10 feature dimensions"
SUMMARY_LI=$($BIN -r "$TMP_DIR/model_l_i.json" -q)
echo "$SUMMARY_LI" | grep "Dimensions:          10"
echo "$SUMMARY_LI" | grep "Label Columns (-L):  1"
echo "$SUMMARY_LI" | grep "Ignore Columns (-I):  12"
echo "  [PASS] Combined -L 1 -I 12 resolved to 10 feature dimensions."

# Test 5: Verify CEIF JSON compatibility fields
echo "Test 5: Verify globals JSON fields for CEIF compatibility..."
grep -E '"globals"\s*:' "$TMP_DIR/model_l_i.json" > /dev/null
grep -E '"labelDims"\s*:\s*"1"' "$TMP_DIR/model_l_i.json" > /dev/null
grep -E '"ignoreDims"\s*:\s*"12"' "$TMP_DIR/model_l_i.json" > /dev/null
echo "  [PASS] CEIF globals compatibility block found in model JSON."

# Test 6: Score with automatic column spec restoration from model JSON
echo "Test 6: Score test CSV without passing -L / -I on CLI (auto-restore from model)..."
set +e
$BIN -r "$TMP_DIR/model_l_i.json" -a "$WINE_CSV" -H -f ';' -o "$TMP_DIR/scores.csv" -T 0.65
RC=$?
set -e
if [[ "$RC" -ne 0 && "$RC" -ne 2 ]]; then
    echo "Error: Expected exit code 0 or 2, got $RC"
    exit 1
fi
ROW_COUNT=$(wc -l < "$TMP_DIR/scores.csv")
if [[ "$ROW_COUNT" -ne 1599 ]]; then
    echo "Error: Expected 1599 scored rows, got $ROW_COUNT"
    exit 1
fi
echo "  [PASS] Scored 1599 rows using model-persisted column specs (Exit code: $RC)."

# Test 7: Error handling for zero active dimensions
echo "Test 7: Verify graceful error when all columns are ignored or excluded..."
set +e
ERR_OUT=$($BIN -l "$WINE_CSV" -H -f ';' -I 1-12 -w "$TMP_DIR/empty.json" 2>&1)
set -e
if echo "$ERR_OUT" | grep -q "no active feature dimensions remaining"; then
    echo "  [PASS] Correctly rejected configuration with 0 active features."
else
    echo "Error: Failed to catch zero active feature dimensions! Output: $ERR_OUT"
    exit 1
fi

echo ">>> ALL FEATURE 2 TESTS PASSED SUCCESSFULLY! <<<"
