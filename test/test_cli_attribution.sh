#!/bin/bash
set -euo pipefail

GEIF_BIN="bin/geif"
TEST_DIR=$(mktemp -d /tmp/geif_test_attr_XXXXXX)
trap 'rm -rf "$TEST_DIR"' EXIT

echo "=== Running GEIF Attribution & Advanced Templating Integration Tests ==="

# Prepare synthetic dataset
# Columns: label,feat1,feat2,feat3,category
TRAIN_CSV="$TEST_DIR/train.csv"
cat << 'EOF' > "$TRAIN_CSV"
row1,10.0,20.0,30.0,catA
row2,10.2,19.8,30.1,catA
row3,9.8,20.1,29.9,catA
row4,10.1,20.2,30.2,catA
row5,9.9,19.9,29.8,catA
row6,10.0,20.0,30.0,catB
row7,10.1,20.1,30.1,catB
row8,9.9,19.9,29.9,catB
EOF

TEST_CSV="$TEST_DIR/test.csv"
cat << 'EOF' > "$TEST_CSV"
inlier1,10.0,20.0,30.0,catA
anomaly_f1,99.0,20.0,30.0,catA
anomaly_f2,10.0,99.0,30.0,catA
anomaly_f3,10.0,20.0,99.0,catA
inlier2,10.1,20.1,30.1,catB
EOF

MODEL_JSON="$TEST_DIR/model.json"

# Helper for scoring where exit code 2 (outliers present) or 0 (no outliers) are both valid success codes
run_geif_score() {
    set +e
    $GEIF_BIN "$@"
    local rc=$?
    set -e
    if [ "$rc" -ne 0 ] && [ "$rc" -ne 2 ]; then
        echo "ERROR: Command failed with exit code $rc: $GEIF_BIN $*"
        exit 1
    fi
}

# Test 1: Train model with -L 1 -C 5
echo "Test 1: Train multi-category model with -L 1 -C 5..."
$GEIF_BIN -l "$TRAIN_CSV" -w "$MODEL_JSON" -L 1 -C 5 -t 20 -s 64 -W -A

# Test 2: Custom templating with precision -d, RGB %x, score %s, label %l, category %c
echo "Test 2: Test -p templating with %x (RGB), -d precision, and %s/%l/%c..."
SCORES_OUT="$TEST_DIR/scores_basic.txt"
run_geif_score -r "$MODEL_JSON" -a "$TEST_CSV" -L 1 -C 5 -d 3 -p "%l;%c;%s;%x;%o" -o "$SCORES_OUT"

cat "$SCORES_OUT"
# Check row count
LINE_COUNT=$(wc -l < "$SCORES_OUT")
if [ "$LINE_COUNT" -ne 5 ]; then
    echo "ERROR: Expected 5 scored lines, got $LINE_COUNT"
    exit 1
fi

# Verify format of first line: inlier1;catA;<score_with_3_decimals>;<6_hex_digits>;<outlier_flag>
FIRST_LINE=$(head -n 1 "$SCORES_OUT")
echo "Checking first line: $FIRST_LINE"
if ! echo "$FIRST_LINE" | grep -qE '^inlier1;catA;[0-9]+\.[0-9]{3};[0-9A-Fa-f]{6};[01]$'; then
    echo "ERROR: First line does not match expected format"
    exit 1
fi
echo "  [PASS] -p template with %s, %l, %c, %x, %o and -d 3 precision verified."

# Test 3: Dimension attribution (%e) and feature averages (%a)
echo "Test 3: Single-dimension attribution impact (%e) and averages (%a)..."
ATTR_OUT="$TEST_DIR/scores_attr.txt"
run_geif_score -r "$MODEL_JSON" -a "$TEST_CSV" -L 1 -C 5 -e "," -d 2 -p "%l|%s|e=%e|a=%a|d=%d" -o "$ATTR_OUT"

cat "$ATTR_OUT"

# Verify that for anomaly_f1, attribution on feature 1 is significantly higher than feature 2 and 3
ANOMALY_F1_LINE=$(grep "^anomaly_f1|" "$ATTR_OUT")
echo "Anomaly F1 line: $ANOMALY_F1_LINE"
E_VALS=$(echo "$ANOMALY_F1_LINE" | awk -F'|' '{print $3}' | sed 's/e=//')
E1=$(echo "$E_VALS" | cut -d',' -f1)
E2=$(echo "$E_VALS" | cut -d',' -f2)
E3=$(echo "$E_VALS" | cut -d',' -f3)

echo "Attribution scores for anomaly_f1: E1=$E1, E2=$E2, E3=$E3"
AWK_CHECK=$(awk -v e1="$E1" -v e2="$E2" -v e3="$E3" 'BEGIN { if (e1 > e2 && e1 > e3 && e1 > 0.4) print "OK"; else print "FAIL"; }')
if [ "$AWK_CHECK" != "OK" ]; then
    echo "ERROR: Single-dimension attribution for anomaly_f1 did not isolate feature 1 properly: E1=$E1, E2=$E2, E3=$E3"
    exit 1
fi
echo "  [PASS] Single-dimension attribution (%e) accurately isolated the corrupted dimension."

# Test 4: Dimension expansion with -j and custom printf format with -m
echo "Test 4: Dimension expansion with -j and %m..."
EXPAND_OUT="$TEST_DIR/scores_expand.txt"
run_geif_score -r "$MODEL_JSON" -a "$TEST_CSV" -L 1 -C 5 -f "," -e "|" -m "%.1f" -j "%i:val=%d,imp=%e" -p "%l => %m" -o "$EXPAND_OUT"

cat "$EXPAND_OUT"
EXPAND_F2=$(grep "^anomaly_f2 =>" "$EXPAND_OUT")
echo "Expand F2: $EXPAND_F2"
if ! echo "$EXPAND_F2" | grep -q "1:val=10.0,imp="; then
    echo "ERROR: -j expansion failed on dimension 1"
    exit 1
fi
if ! echo "$EXPAND_F2" | grep -q "2:val=99.0,imp="; then
    echo "ERROR: -j expansion failed on dimension 2"
    exit 1
fi
echo "  [PASS] -j dimension template expansion (%m) verified."

# Test 5: Inlier output template (-v) vs Outlier output template (-p)
echo "Test 5: Dual output templates: -v for inliers and -p for outliers..."
DUAL_OUT="$TEST_DIR/scores_dual.txt"
run_geif_score -r "$MODEL_JSON" -a "$TEST_CSV" -L 1 -C 5 -T 0.6 -v "INLIER: %l (%c) score=%s" -p "OUTLIER: %l (%c) score=%s" -o "$DUAL_OUT"

cat "$DUAL_OUT"
if ! grep -q "^INLIER: inlier1" "$DUAL_OUT"; then
    echo "ERROR: inlier1 was not formatted with -v template"
    exit 1
fi
if ! grep -q "^OUTLIER: anomaly_f1" "$DUAL_OUT"; then
    echo "ERROR: anomaly_f1 was not formatted with -p template"
    exit 1
fi
echo "  [PASS] Inlier template (-v) and outlier template (-p) routed correctly."

# Test 6: Outlier threshold aliases (-O average and -O 80%)
echo "Test 6: Outlier threshold with -O average and -O 80%..."
run_geif_score -r "$MODEL_JSON" -a "$TEST_CSV" -L 1 -C 5 -O average -p "%l;%o" -o "$TEST_DIR/scores_avg.txt"
run_geif_score -r "$MODEL_JSON" -a "$TEST_CSV" -L 1 -C 5 -O 80% -p "%l;%o" -o "$TEST_DIR/scores_80pct.txt"

if [ ! -s "$TEST_DIR/scores_avg.txt" ] || [ ! -s "$TEST_DIR/scores_80pct.txt" ]; then
    echo "ERROR: -O average or -O 80% failed to produce output"
    exit 1
fi
echo "  [PASS] -O average and percentage thresholds accepted and evaluated."

echo "=== All GEIF Attribution & Advanced Templating Integration Tests Passed! ==="
