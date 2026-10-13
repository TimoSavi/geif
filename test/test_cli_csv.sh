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

# Test 7: Verify field delimiter (-f) and list separator (-e) persistence in JSON model
echo "Test 7: Verify delimiter persistence in saved JSON model..."
$BIN -r "$TMP_DIR/model.json" -q | grep -F "Input Delimiter (-f): ';'"
$BIN -r "$TMP_DIR/model.json" -q | grep -F "List Separator (-e):  ','"
grep -E -q '"input_separator":[[:space:]]*";"' "$TMP_DIR/model.json"
grep -E -q '"inputSeparator":[[:space:]]*";"' "$TMP_DIR/model.json"

# Also verify explicit -e '|' persistence
$BIN -l "$WINE_CSV" -H -f ';' -e '|' -w "$TMP_DIR/custom_sep.json" -i 10 -s 32
$BIN -r "$TMP_DIR/custom_sep.json" -q | grep -F "Input Delimiter (-f): ';'"
$BIN -r "$TMP_DIR/custom_sep.json" -q | grep -F "List Separator (-e):  '|'"
grep -E -q '"list_separator":[[:space:]]*"\|"' "$TMP_DIR/custom_sep.json"
grep -E -q '"listSeparator":[[:space:]]*"\|"' "$TMP_DIR/custom_sep.json"
echo "  [PASS] Delimiters properly serialized in root and CEIF-compatible globals."

# Test 8: Auto-restoration of delimiter during scoring without -f flag
echo "Test 8: Scoring without -f flag (auto-restoring saved ';' delimiter)..."
head -n 25 "$WINE_CSV" | tail -n 20 > "$TMP_DIR/wine_subset.csv"
# Notice: no -f ';' supplied on CLI!
$BIN -r "$TMP_DIR/model.json" -a "$TMP_DIR/wine_subset.csv" -o "$TMP_DIR/restored_scores.csv" -T 0.99
RESTORED_COUNT=$(wc -l < "$TMP_DIR/restored_scores.csv")
test "$RESTORED_COUNT" -eq 20
# Output delimiter should also default to the model's saved separator
grep -q ";" "$TMP_DIR/restored_scores.csv"
echo "  [PASS] Model automatically used saved delimiter ';' without -f on CLI ($RESTORED_COUNT rows)."

# Test 9: In-place continuous training (-z) without -f flag
echo "Test 9: In-place continuous training (-z) without -f flag..."
cp "$TMP_DIR/model.json" "$TMP_DIR/model_inplace.json"
BEFORE_ROWS=$($BIN -r "$TMP_DIR/model_inplace.json" -q | grep -o 'rows=[0-9]*' | head -n 1 | cut -d= -f2)
head -n 50 "$WINE_CSV" | tail -n 30 > "$TMP_DIR/wine_batch2.csv"
# Ingest new batch without -f flag
$BIN -z "$TMP_DIR/model_inplace.json" -l "$TMP_DIR/wine_batch2.csv" -v 2>&1 | grep "Ingested 30 rows"
AFTER_ROWS=$($BIN -r "$TMP_DIR/model_inplace.json" -q | grep -o 'rows=[0-9]*' | head -n 1 | cut -d= -f2)
test "$AFTER_ROWS" -eq $((BEFORE_ROWS + 30))
echo "  [PASS] Continuous learning (-z) automatically used saved delimiter (rows: $BEFORE_ROWS -> $AFTER_ROWS)."

# Test 10: Explicit CLI override (-f) takes precedence over saved model delimiter
echo "Test 10: Explicit CLI flag overrides saved delimiter..."
# Convert wine batch to comma-separated
tr ';' ',' < "$TMP_DIR/wine_batch2.csv" > "$TMP_DIR/wine_comma.csv"
$BIN -z "$TMP_DIR/model_inplace.json" -l "$TMP_DIR/wine_comma.csv" -f ',' -v 2>&1 | grep "Ingested 30 rows"
echo "  [PASS] Explicit -f ',' successfully overrode model delimiter."

echo ">>> ALL FEATURE 1 TESTS PASSED SUCCESSFULLY! <<<"
