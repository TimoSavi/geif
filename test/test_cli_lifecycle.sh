#!/bin/bash
set -euo pipefail

BIN="bin/geif"
TMP_DIR=$(mktemp -d)
trap 'rm -rf "$TMP_DIR"' EXIT

WINE_CSV="/home/timo_savinen_elisa_fi/docs/tools/geif/prod_plan/winequality-red.csv"

echo "================================================================="
echo "FEATURE 4 TEST: Category Lifecycle (-F, -R, -N, -M, -p, -S)"
echo "================================================================="

# Step 0: Train reference multi-category model on winequality-red.csv (-C 12)
echo "Training reference model with -C 12..."
$BIN -l "$WINE_CSV" -H -f ';' -C 12 -w "$TMP_DIR/model_full.json" -i 20 -s 64

# Test 1: Category Filter (-F) - Positive Regex Filter
echo "Test 1: Category Filter (-F '^(5|6)$') - Score only categories 5 and 6..."
$BIN -r "$TMP_DIR/model_full.json" -a "$WINE_CSV" -H -f ';' -F '^(5|6)$' -o "$TMP_DIR/filtered_pos.csv" || true
LINES_POS=$(wc -l < "$TMP_DIR/filtered_pos.csv")
# Category 5 has 681 rows, Category 6 has 638 rows. Total = 1319.
if [ "$LINES_POS" -ne 1319 ]; then
    echo "  [FAIL] Expected 1319 filtered rows (5 & 6), got $LINES_POS"
    exit 1
fi
echo "  [PASS] Positive filter (-F '^(5|6)$') correctly filtered to exactly 1319 rows."

# Test 2: Category Filter (-F) - Inverted Regex Filter (-F"-v ^5$")
echo "Test 2: Category Inverted Filter (-F'-v ^5$') - Skip category 5..."
$BIN -r "$TMP_DIR/model_full.json" -a "$WINE_CSV" -H -f ';' -F "-v ^5$" -o "$TMP_DIR/filtered_inv.csv" || true
LINES_INV=$(wc -l < "$TMP_DIR/filtered_inv.csv")
# 1599 total - 681 (quality 5) = 918 rows
if [ "$LINES_INV" -ne 918 ]; then
    echo "  [FAIL] Expected 918 rows when category 5 excluded, got $LINES_INV"
    exit 1
fi
echo "  [PASS] Inverted filter (-F'-v ^5$') correctly skipped category 5 (918 rows kept)."

# Test 3: Category Pruning by Minimum Row Count (-R <min_rows>)
echo "Test 3: Train with Category Minimum Row Threshold (-R 100)..."
# Quality counts: 3: 10, 4: 53, 5: 681, 6: 638, 7: 199, 8: 18.
# With -R 100, categories 3, 4, 8 must be pruned. Exactly 3 categories (5, 6, 7) must remain.
$BIN -l "$WINE_CSV" -H -f ';' -C 12 -R 100 -w "$TMP_DIR/model_r100.json" -i 15 -s 32 -v > "$TMP_DIR/r100.log" 2>&1
R100_SUMMARY=$($BIN -r "$TMP_DIR/model_r100.json" -q)
echo "$R100_SUMMARY" | grep "Sub-forests:         3"
echo "$R100_SUMMARY" | grep "Sub-forest '5'"
echo "$R100_SUMMARY" | grep "Sub-forest '6'"
echo "$R100_SUMMARY" | grep "Sub-forest '7'"
# Verify pruned categories do not exist
if echo "$R100_SUMMARY" | grep -E "Sub-forest '(3|4|8)'"; then
    echo "  [FAIL] Categories with <100 rows were not pruned!"
    exit 1
fi
echo "  [PASS] -R 100 correctly pruned categories below 100 rows, retaining exactly 3 sub-forests."

# Test 4: NEW / Unseen Category Template Formatting (-N <tmpl>)
echo "Test 4: Format NEW / Unseen category with custom template (-N 'NEW;%l;%c;%m')..."
UNSEEN_INPUT="7.4;0.7;0;1.9;0.076;11;34;0.9978;3.51;0.56;9.4;99"
NEW_OUT=$(echo "$UNSEEN_INPUT" | $BIN -r "$TMP_DIR/model_full.json" -a - -f ';' -N "NEW;cat=%c;metric=%m;outlier=%o" || true)
if [ "$NEW_OUT" != "NEW;cat=99;metric=0.000000;outlier=1" ]; then
    echo "  [FAIL] Unexpected NEW category formatted output: '$NEW_OUT'"
    exit 1
fi
echo "  [PASS] Unseen category formatted with -N template: '$NEW_OUT'."

# Test 5: MISSED Category Reporting (-M <tmpl>)
echo "Test 5: Report MISSED categories at end of analysis batch (-M 'MISSED;%c;%s')..."
# Create a test input containing only category 5 rows
awk -F';' '$12 == "5" { print; if (++n == 10) exit }' "$WINE_CSV" > "$TMP_DIR/test_cat5_only.csv"
set +e
$BIN -r "$TMP_DIR/model_full.json" -a "$TMP_DIR/test_cat5_only.csv" -f ';' -M "MISSED;%c;%s" -o "$TMP_DIR/missed_output.csv"
EXIT_CODE=$?
set -e
if [ "$EXIT_CODE" -ne 2 ]; then
    echo "  [FAIL] Expected exit code 2 when missed categories exist, got $EXIT_CODE"
    exit 1
fi
# Model has categories 3, 4, 5, 6, 7, 8. Test file has only 5.
# Missed categories should be: 3, 4, 6, 7, 8.
grep "MISSED;3;1.000000" "$TMP_DIR/missed_output.csv" > /dev/null
grep "MISSED;4;1.000000" "$TMP_DIR/missed_output.csv" > /dev/null
grep "MISSED;6;1.000000" "$TMP_DIR/missed_output.csv" > /dev/null
grep "MISSED;7;1.000000" "$TMP_DIR/missed_output.csv" > /dev/null
grep "MISSED;8;1.000000" "$TMP_DIR/missed_output.csv" > /dev/null
MISSED_COUNT=$(grep -c "^MISSED;" "$TMP_DIR/missed_output.csv")
if [ "$MISSED_COUNT" -ne 5 ]; then
    echo "  [FAIL] Expected 5 missed categories, got $MISSED_COUNT"
    exit 1
fi
echo "  [PASS] All 5 absent categories detected and emitted as MISSED notifications with exit code 2."

# Test 6: Custom Point Output Templating (-p <tmpl>)
echo "Test 6: Format scored observations with custom template (-p 'A;%s;%c;%m')..."
PT_OUT=$(head -n 2 "$TMP_DIR/test_cat5_only.csv" | $BIN -r "$TMP_DIR/model_full.json" -a - -f ';' -p "A;%s;%c;%m" || true)
echo "$PT_OUT" | grep "^A;[0-9.]*;5;[0-9.]*" > /dev/null
echo "  [PASS] Custom point output formatted accurately using -p."

# Test 7: Silent / Outliers Only (-S)
echo "Test 7: Outliers only filtering (-S)..."
# Score with a threshold where only some points trigger
$BIN -r "$TMP_DIR/model_full.json" -a "$WINE_CSV" -H -f ';' -O 0.55 -S -o "$TMP_DIR/outliers_only.csv" || true
OUTLIER_ROWS=$(wc -l < "$TMP_DIR/outliers_only.csv")
ALL_ROWS=$(wc -l < "$TMP_DIR/filtered_pos.csv")
if [ "$OUTLIER_ROWS" -ge 1599 ] || [ "$OUTLIER_ROWS" -le 0 ]; then
    echo "  [FAIL] Expected only outlier rows with -S, got $OUTLIER_ROWS"
    exit 1
fi
echo "  [PASS] -S correctly suppressed normal inliers, outputting only $OUTLIER_ROWS outliers."

echo ">>> ALL FEATURE 4 TESTS PASSED SUCCESSFULLY! <<<"
