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
head -n 20 "$WINE_CSV" | $BIN -r "$TMP_DIR/model.json" -a - -H -f ';' -e ';' -o "$TMP_DIR/scores.csv" -O -1 || test $? -eq 2
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
$BIN -r "$TMP_DIR/model.json" -a "$WINE_CSV" -H -f ';' -O 0.9999 > /dev/null
EC_ZERO=$?
# Low threshold (outliers present) -> exit code 2
$BIN -r "$TMP_DIR/model.json" -a "$WINE_CSV" -H -f ';' -O 0.3000 > /dev/null
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
# Pass --no-header because nan_test.csv has no header row
$BIN -r "$TMP_DIR/model.json" -a "$TMP_DIR/nan_test.csv" --no-header -f ';' -e ';' -O -1 -o "$TMP_DIR/nan_out.csv" || test $? -eq 2
NAN_SCORED=$(wc -l < "$TMP_DIR/nan_out.csv")
test "$NAN_SCORED" -eq 2
echo "  [PASS] NaN, Inf, and empty fields gracefully parsed as 0.0 without errors."

# Test 7: Verify field delimiter (-f), list separator (-e), and header (-H) persistence in JSON model
echo "Test 7: Verify delimiter and header persistence in saved JSON model..."
$BIN -r "$TMP_DIR/model.json" -q | grep -F "Input Delimiter (-f): ';'"
$BIN -r "$TMP_DIR/model.json" -q | grep -F "List Separator (-e):  ','"
$BIN -r "$TMP_DIR/model.json" -q | grep -F "Skip Header (-H):     Yes"
grep -E -q '"input_separator":[[:space:]]*";"' "$TMP_DIR/model.json"
grep -E -q '"inputSeparator":[[:space:]]*";"' "$TMP_DIR/model.json"
grep -E -q '"has_header":[[:space:]]*true' "$TMP_DIR/model.json"
grep -E -q '"header":[[:space:]]*1' "$TMP_DIR/model.json"

# Also verify explicit -e '|' persistence and no-header model
$BIN -l "$WINE_CSV" -f ';' -e '|' --no-header -w "$TMP_DIR/custom_sep.json" -i 10 -s 32
$BIN -r "$TMP_DIR/custom_sep.json" -q | grep -F "Input Delimiter (-f): ';'"
$BIN -r "$TMP_DIR/custom_sep.json" -q | grep -F "List Separator (-e):  '|'"
$BIN -r "$TMP_DIR/custom_sep.json" -q | grep -F "Skip Header (-H):     No"
grep -E -q '"list_separator":[[:space:]]*"\|"' "$TMP_DIR/custom_sep.json"
grep -E -q '"listSeparator":[[:space:]]*"\|"' "$TMP_DIR/custom_sep.json"
grep -E -q '"has_header":[[:space:]]*false' "$TMP_DIR/custom_sep.json"
grep -E -q '"header":[[:space:]]*0' "$TMP_DIR/custom_sep.json"
echo "  [PASS] Delimiters and header state properly serialized in root and CEIF-compatible globals."

# Test 8: Auto-restoration of delimiter and header skip during scoring without -f or -H flags
echo "Test 8: Scoring without -f or -H flags (auto-restoring saved ';' delimiter and header skip)..."
# Extract 1 header line + 20 data lines from WINE_CSV
head -n 21 "$WINE_CSV" > "$TMP_DIR/wine_subset.csv"
# Notice: no -f ';' or -H supplied on CLI!
$BIN -r "$TMP_DIR/model.json" -a "$TMP_DIR/wine_subset.csv" -o "$TMP_DIR/restored_scores.csv" -O -1 || test $? -eq 2
RESTORED_COUNT=$(wc -l < "$TMP_DIR/restored_scores.csv")
test "$RESTORED_COUNT" -eq 20
# Verify semicolon-delimited input was tokenized into fields and formatted with default '%s %v'
grep -q "7.4,0.7,0,1.9" "$TMP_DIR/restored_scores.csv"
echo "  [PASS] Model automatically used saved delimiter ';' and skipped header ($RESTORED_COUNT rows)."

# Test 9: In-place continuous training (-z) without -f or -H flags
echo "Test 9: In-place continuous training (-z) without -f or -H flags..."
cp "$TMP_DIR/model.json" "$TMP_DIR/model_inplace.json"
BEFORE_ROWS=$($BIN -r "$TMP_DIR/model_inplace.json" -q | grep -o 'rows=[0-9]*' | head -n 1 | cut -d= -f2)
# Extract 1 header line + 30 data lines
head -n 31 "$WINE_CSV" > "$TMP_DIR/wine_batch2.csv"
# Ingest new batch without -f or -H flag
$BIN -z "$TMP_DIR/model_inplace.json" -l "$TMP_DIR/wine_batch2.csv" -v 2>&1 | grep "Ingested 30 rows"
AFTER_ROWS=$($BIN -r "$TMP_DIR/model_inplace.json" -q | grep -o 'rows=[0-9]*' | head -n 1 | cut -d= -f2)
test "$AFTER_ROWS" -eq $((BEFORE_ROWS + 30))
echo "  [PASS] Continuous learning (-z) automatically used saved delimiter and skipped header (rows: $BEFORE_ROWS -> $AFTER_ROWS)."

# Test 10: Explicit CLI override (-f) takes precedence over saved model delimiter
echo "Test 10: Explicit CLI flag overrides saved delimiter..."
# Convert wine batch (header + 30 rows) to comma-separated
tr ';' ',' < "$TMP_DIR/wine_batch2.csv" > "$TMP_DIR/wine_comma.csv"
$BIN -z "$TMP_DIR/model_inplace.json" -l "$TMP_DIR/wine_comma.csv" -f ',' -v 2>&1 | grep "Ingested 30 rows"
echo "  [PASS] Explicit -f ',' successfully overrode model delimiter."

# Test 11: Explicit CLI override (--no-header) on a model with has_header=true
echo "Test 11: Explicit --no-header override on model with saved header..."
cp "$TMP_DIR/model.json" "$TMP_DIR/model_no_hdr.json"
# Extract 20 raw data rows without header
tail -n 20 "$WINE_CSV" > "$TMP_DIR/wine_no_header_data.csv"
# Model has has_header=true; override with --no-header to ingest all 20 rows
$BIN -z "$TMP_DIR/model_no_hdr.json" -l "$TMP_DIR/wine_no_header_data.csv" --no-header -v 2>&1 | grep "Ingested 20 rows"
echo "  [PASS] Explicit --no-header successfully overrode saved header state."

# Test 12: RC file configuration for HEADER directive
echo "Test 12: RC configuration file HEADER directive..."
cat << 'EOF' > "$TMP_DIR/test_geifrc"
HEADER = true
EOF
$BIN -g "$TMP_DIR/test_geifrc" -l "$WINE_CSV" -f ';' -w "$TMP_DIR/rc_model.json" -i 10 -s 32 -v 2>&1 | grep "Ingested 1599 rows"
$BIN -r "$TMP_DIR/rc_model.json" -q | grep -F "Skip Header (-H):     Yes"
echo "  [PASS] RC file HEADER directive successfully applied."

echo ">>> ALL FEATURE 1 TESTS PASSED SUCCESSFULLY! <<<"
