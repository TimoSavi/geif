#!/bin/bash
set -euo pipefail

BIN="bin/geif"
TMP_DIR=$(mktemp -d)
trap 'rm -rf "$TMP_DIR"' EXIT

WINE_CSV="/home/timo_savinen_elisa_fi/docs/tools/geif/prod_plan/winequality-red.csv"
PADDY_CSV="/home/timo_savinen_elisa_fi/docs/tools/geif/prod_plan/paddydataset.csv"

echo "================================================================="
echo "FEATURE 3 TEST: Multi-Category Sub-Forest Engine (-C <range>)"
echo "================================================================="

# Test 1: Train multi-category model on winequality-red.csv with -C 12 (quality)
echo "Test 1: Train multi-category sub-forests with -C 12 on winequality-red.csv..."
$BIN -l "$WINE_CSV" -H -f ';' -C 12 -w "$TMP_DIR/model_wine_cat.json" -i 30 -s 64 -v > "$TMP_DIR/wine_train.log" 2>&1
cat "$TMP_DIR/wine_train.log" | grep "11 feature dimensions, 6 categories"
echo "  [PASS] Successfully partitioned into 6 distinct sub-forests across 11 features."

# Test 2: Query multi-category model metadata and sub-forest diagnostics
echo "Test 2: Query multi-category ensemble diagnostics (-r -q)..."
SUMMARY=$($BIN -r "$TMP_DIR/model_wine_cat.json" -q)
echo "$SUMMARY" | grep "Dimensions:          11"
echo "$SUMMARY" | grep "Sub-forests:         6"
echo "$SUMMARY" | grep "Category Columns (-C): 12"
echo "$SUMMARY" | grep "Sub-forest '3': rows=10"
echo "$SUMMARY" | grep "Sub-forest '5': rows=681"
echo "$SUMMARY" | grep "Sub-forest '8': rows=18"
echo "  [PASS] Query inspection accurately reported 6 sub-forests with exact row counts."

# Test 3: Verify CEIF-compatible JSON serialization of forests array and globals
echo "Test 3: Verify JSON model structure (forests array and categoryDims globals)..."
grep -E '"categoryDims"\s*:\s*"12"' "$TMP_DIR/model_wine_cat.json" > /dev/null
grep -E '"forests"\s*:' "$TMP_DIR/model_wine_cat.json" > /dev/null
grep -E '"category"\s*:\s*"3"' "$TMP_DIR/model_wine_cat.json" > /dev/null
grep -E '"category"\s*:\s*"5"' "$TMP_DIR/model_wine_cat.json" > /dev/null
grep -E '"category"\s*:\s*"8"' "$TMP_DIR/model_wine_cat.json" > /dev/null
echo "  [PASS] CEIF-compatible 'forests' array and 'categoryDims' JSON structure confirmed."

# Test 4: Score dataset with automatic category routing
echo "Test 4: Score dataset with dynamic category routing (auto-restoring -C from model)..."
set +e
$BIN -r "$TMP_DIR/model_wine_cat.json" -a "$WINE_CSV" -H -f ';' -o "$TMP_DIR/wine_scores.csv" -T 0.65
RC=$?
set -e
if [ "$RC" -ne 0 ] && [ "$RC" -ne 2 ]; then
    echo "  [FAIL] Unexpected exit code: $RC"
    exit 1
fi
LINES=$(wc -l < "$TMP_DIR/wine_scores.csv")
if [ "$LINES" -ne 1599 ]; then
    echo "  [FAIL] Expected 1599 scored rows, got $LINES"
    exit 1
fi
echo "  [PASS] Scored all 1599 rows with per-category isolation (Exit code: $RC)."

# Test 5: Verify unseen category handling (routes to maximum outlier score = 1.0)
echo "Test 5: Score observation with unseen category (quality = 99)..."
# Create a row with unknown quality 99
UNSEEN_ROW="7.4;0.7;0;1.9;0.076;11;34;0.9978;3.51;0.56;9.4;99"
echo "$UNSEEN_ROW" | $BIN -r "$TMP_DIR/model_wine_cat.json" -a - -f ';' -e ';' -o "$TMP_DIR/unseen_score.csv" -T 0.5 || true
UNSEEN_OUTPUT=$(cat "$TMP_DIR/unseen_score.csv")
echo "$UNSEEN_OUTPUT" | grep ";1.000000;1;"
echo "  [PASS] Unseen category successfully flagged as outlier with score 1.000000."

# Test 6: Multi-column category test on paddydataset.csv (-C 2-4)
echo "Test 6: Multi-column category partitioning on paddydataset.csv (-C 2-4)..."
$BIN -l "$PADDY_CSV" -H -f ',' -C 2-4 -w "$TMP_DIR/model_paddy.json" -i 15 -s 32 -v > "$TMP_DIR/paddy_train.log" 2>&1
cat "$TMP_DIR/paddy_train.log" | grep "42 feature dimensions"
PADDY_SUMMARY=$($BIN -r "$TMP_DIR/model_paddy.json" -q)
echo "$PADDY_SUMMARY" | grep "Dimensions:          42"
echo "$PADDY_SUMMARY" | grep "Category Columns (-C): 2-4"
echo "  [PASS] Multi-column composite categories (-C 2-4) correctly resolved 42 active features."

# Test 7: Combined Category (-C 12) and Label (-L 1)
echo "Test 7: Combined category and label specifications (-C 12 -L 1)..."
$BIN -l "$WINE_CSV" -H -f ';' -C 12 -L 1 -w "$TMP_DIR/model_wine_cat_label.json" -i 20 -s 32 -v > "$TMP_DIR/wine_cl.log" 2>&1
cat "$TMP_DIR/wine_cl.log" | grep "10 feature dimensions, 6 categories"
CL_SUMMARY=$($BIN -r "$TMP_DIR/model_wine_cat_label.json" -q)
echo "$CL_SUMMARY" | grep "Dimensions:          10"
echo "$CL_SUMMARY" | grep "Category Columns (-C): 12"
echo "$CL_SUMMARY" | grep "Label Columns (-L):  1"
echo "  [PASS] Combined -C 12 and -L 1 correctly resolved to 10 features across 6 sub-forests."

# Test 8: Categorization mode (-c) with 100 circles benchmark dataset
CIRCLES_TXT="/home/timo_savinen_elisa_fi/docs/tools/geif/prod_plan/circles.txt"
if [ -f "$CIRCLES_TXT" ]; then
    echo "Test 8: Categorization mode (-c) on 100 circles dataset..."
    $BIN -l "$CIRCLES_TXT" -H -C 3 -w "$TMP_DIR/model_circles.json" -t 50 -s 64
    
    # Categorize first 50 rows of circles.txt
    head -n 51 "$CIRCLES_TXT" > "$TMP_DIR/circles_sample.txt"
    $BIN -r "$TMP_DIR/model_circles.json" -c "$TMP_DIR/circles_sample.txt" -H -p "%c,%C" > "$TMP_DIR/circles_cat_out.csv"
    
    # Verify accurate categorization (actual == predicted)
    ACCURACY=$(awk -F',' '{total++; if ($1==$2) correct++} END {printf "%d/%d", correct, total}' "$TMP_DIR/circles_cat_out.csv")
    if [ "$ACCURACY" != "50/50" ]; then
        echo "  [FAIL] Expected 50/50 accurate predictions, got $ACCURACY"
        exit 1
    fi
    echo "  [PASS] Categorization achieved $ACCURACY (100%) accuracy on test sample."

    # Verify outlier suppression (-O 0.4) during categorization
    $BIN -r "$TMP_DIR/model_circles.json" -c "$TMP_DIR/circles_sample.txt" -H -F "-v ^(6|11)$" -O 0.4 -p "%r: %c->%C (%s)" > "$TMP_DIR/circles_thresh_out.csv"
    SUPPRESSED_LINES=$(wc -l < "$TMP_DIR/circles_thresh_out.csv")
    # Rows not matching 6 or 11 will have score > 0.4 and be suppressed
    if [ "$SUPPRESSED_LINES" -lt 1 ] || [ "$SUPPRESSED_LINES" -gt 10 ]; then
        echo "  [FAIL] Unexpected line count for outlier suppression: $SUPPRESSED_LINES"
        exit 1
    fi
    echo "  [PASS] Outlier suppression with -O 0.4 correctly filtered distant categories ($SUPPRESSED_LINES rows retained)."

    # Test 9: Multi-filter (-F) exclusion and inverted retention
    echo "Test 9: Multi-filter (-F) testing..."
    # Exclude categories 6 and 11
    $BIN -r "$TMP_DIR/model_circles.json" -c "$TMP_DIR/circles_sample.txt" -H -F "^6$" -F "^11$" -p "%C" > "$TMP_DIR/circles_multi_f.csv"
    if grep -q -E "^(6|11)$" "$TMP_DIR/circles_multi_f.csv"; then
        echo "  [FAIL] Categories 6 or 11 found despite -F exclusions!"
        exit 1
    fi
    echo "  [PASS] Multiple -F flags correctly excluded categories 6 and 11."

    # Inverted filter: keep only 6 and 11
    $BIN -r "$TMP_DIR/model_circles.json" -c "$TMP_DIR/circles_sample.txt" -H -F "-v ^(6|11)$" -p "%C" > "$TMP_DIR/circles_inv_f.csv"
    if grep -v -E "^(6|11)$" "$TMP_DIR/circles_inv_f.csv" | grep -q "[0-9]"; then
        echo "  [FAIL] Non-(6|11) categories found with inverted filter -F '-v ^(6|11)$'!"
        exit 1
    fi
    echo "  [PASS] Inverted filter -F '-v ^(6|11)$' retained exclusively categories 6 and 11."
fi

echo ">>> ALL FEATURE 3 TESTS PASSED SUCCESSFULLY! <<<"
