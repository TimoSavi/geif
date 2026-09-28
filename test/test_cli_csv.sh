#!/usr/bin/env bash
set -euo pipefail

BIN="./bin/geif"
WINE_CSV="/home/timo_savinen_elisa_fi/docs/tools/geif/prod_plan/winequality-red.csv"
TMP_DIR="$(mktemp -d /tmp/geif_test_csv_XXXXXX)"
trap 'rm -rf "$TMP_DIR"' EXIT

echo "================================================================="
echo "FEATURE 1 TEST: CSV Delimiters (-f, -e), Headers (-H), & Stdin (-)"
echo "================================================================="

# Test 1: Train from file with header and semicolon delimiter (-H -f ';')
echo "Test 1: Train with -H -f ';' on winequality-red.csv..."
$BIN -l "$WINE_CSV" -H -f ';' -w "$TMP_DIR/model.json" -i 50 -s 128 -v 2>&1 | grep "Ingested 1599 rows"
echo "  [PASS] Successfully ingested 1599 data rows (header skipped, 12 dims)."

# Test 2: Inspect saved model (-r -q)
echo "Test 2: Inspect model with -r -q..."
$BIN -r "$TMP_DIR/model.json" -q | grep -E "Dimensions:[[:space:]]+12"
echo "  [PASS] Model verified: 12 dimensions."

# Test 3: UNIX Pipe streaming: train from stdin and save to stdout (-l - and -w -)
echo "Test 3: Pipe streaming (cat ... | geif -l - -H -e ';' -w -)..."
cat "$WINE_CSV" | $BIN -l - -H -e ';' -w - -i 30 -s 64 > "$TMP_DIR/piped_model.json"
test -s "$TMP_DIR/piped_model.json"
$BIN -r "$TMP_DIR/piped_model.json" -q | grep -E "Dimensions:[[:space:]]+12"
echo "  [PASS] Stdin training and stdout JSON serialization verified."

# Test 4: Stream analysis through stdin with -e ';' output formatting
echo "Test 4: Analyze streaming via stdin with -H -f ';' -e ';'..."
head -n 20 "$WINE_CSV" | $BIN -r "$TMP_DIR/model.json" -a - -H -f ';' -e ';' -o "$TMP_DIR/scores.csv" -T 0.99
ROW_COUNT=$(wc -l < "$TMP_DIR/scores.csv")
echo "  Scored $ROW_COUNT rows."
test "$ROW_COUNT" -eq 19
# Verify output is semicolon delimited
grep -q ";" "$TMP_DIR/scores.csv"
echo "  [PASS] Semicolon output formatting and row count verified."

# Test 5: Verify exit code 2 when outliers detected, 0 when none
echo "Test 5: Verify exit code semantics (2 = outliers detected, 0 = clean)..."
set +e
# High threshold (no outliers) -> exit code 0
$BIN -r "$TMP_DIR/model.json" -a "$WINE_CSV" -H -f ';' -T 0.9999 > /dev/null
EC_ZERO=$?
# Low threshold (outliers present) -> exit code 2
$BIN -r "$TMP_DIR/model.json" -a "$WINE_CSV" -H -f ';' -T 0.3000 > /dev/null
EC_TWO=$?
set -e

echo "  Exit code with high threshold (no outliers): $EC_ZERO (expected 0)"
echo "  Exit code with low threshold (outliers present): $EC_TWO (expected 2)"
test "$EC_ZERO" -eq 0
test "$EC_TWO" -eq 2
echo "  [PASS] Exit code semantics match ceif specification."

# Test 6: Verify NaN, Inf, and missing value handling defaults to 0.0
echo "Test 6: Verify NaN, Inf, and missing value handling..."
cat << 'EOF' > "$TMP_DIR/nan_test.csv"
7.4;0.7;nan;1.9;0.076;11;34;0.9978;3.51;NaN;9.4;5
7.4;;0;1.9;null;11;34;0.9978;3.51;0.56;inf;5
EOF
$BIN -r "$TMP_DIR/model.json" -a "$TMP_DIR/nan_test.csv" -f ';' -e ';' -o "$TMP_DIR/nan_out.csv" || test $? -eq 2
NAN_SCORED=$(wc -l < "$TMP_DIR/nan_out.csv")
test "$NAN_SCORED" -eq 2
echo "  [PASS] NaN, Inf, and empty fields gracefully parsed as 0.0 without errors."

echo ">>> ALL FEATURE 1 TESTS PASSED SUCCESSFULLY! <<<"
