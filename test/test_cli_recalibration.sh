#!/bin/bash
set -euo pipefail

GEIF_BIN="bin/geif"
TEST_DIR=$(mktemp -d /tmp/geif_test_recal_XXXXXX)
trap 'rm -rf "$TEST_DIR"' EXIT

echo "=== Running GEIF Recalibration & Threshold Aliases Integration Tests ==="

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

# Prepare synthetic dataset with 1 extreme outlier in catA
TRAIN_CSV="$TEST_DIR/train.csv"
cat << 'EOF' > "$TRAIN_CSV"
in1,10.0,20.0,30.0,catA
in2,10.2,19.8,30.1,catA
in3,9.8,20.1,29.9,catA
in4,10.1,20.2,30.2,catA
in5,9.9,19.9,29.8,catA
in6,10.3,20.0,30.0,catA
bad1,999.0,999.0,999.0,catA
EOF

TEST_CSV="$TEST_DIR/test.csv"
cat << 'EOF' > "$TEST_CSV"
test_norm,10.0,20.0,30.0,catA
test_mild,15.0,25.0,35.0,catA
test_wild,999.0,999.0,999.0,catA
EOF

MODEL_RAW="$TEST_DIR/model_raw.json"
MODEL_CLEAN1="$TEST_DIR/model_clean1.json"
MODEL_CLEAN2="$TEST_DIR/model_clean2.json"

# Test 1: Train model with outlier and -O 80%
echo "Test 1: Train model containing outlier with -O 80%..."
$GEIF_BIN -l "$TRAIN_CSV" -w "$MODEL_RAW" -L 1 -C 5 -t 10 -s 64 -O "80%" -W -A

# Verify JSON contains outlierScore "80%"
if ! grep -qE '"outlierScore":\s*"80%"' "$MODEL_RAW"; then
    echo "ERROR: outlierScore '80%' was not persisted in model globals"
    exit 1
fi
echo "  [PASS] -O 80% persisted in model JSON."

# Check initial sample count in catA (should be 7)
SAMPLE_COUNT_RAW=$(grep -oE '"pool_count":\s*[0-9]+' "$MODEL_RAW" | head -n 1 | grep -oE '[0-9]+')
if [ "$SAMPLE_COUNT_RAW" -ne 7 ]; then
    echo "ERROR: Expected 7 samples in raw model, got $SAMPLE_COUNT_RAW"
    exit 1
fi
echo "  [PASS] Initial model has 7 reservoir samples."

# Test 2: Recalibrate model with -k (prunes single worst outlier)
echo "Test 2: Recalibrate with -r ... -k -w ... (prune 1 outlier)..."
$GEIF_BIN -r "$MODEL_RAW" -k -w "$MODEL_CLEAN1"

SAMPLE_COUNT_CLEAN1=$(grep -oE '"pool_count":\s*[0-9]+' "$MODEL_CLEAN1" | head -n 1 | grep -oE '[0-9]+')
if [ "$SAMPLE_COUNT_CLEAN1" -ne 6 ]; then
    echo "ERROR: Expected 6 samples in clean model after -k, got $SAMPLE_COUNT_CLEAN1"
    exit 1
fi

# Verify the outlier (999.0) was pruned from the sample pool
if grep -qw '999.0' "$MODEL_CLEAN1"; then
    echo "ERROR: Contaminant 999.0 is still present in model sample pool after -k"
    exit 1
fi
echo "  [PASS] -k successfully pruned the extreme outlier from the reservoir pool (pool_count 7 -> 6)."

# Test 3: Multiple -k flags (-k -k)
echo "Test 3: Recalibrate with multiple -k flags (-k -k)..."
$GEIF_BIN -r "$MODEL_RAW" -k -k -w "$MODEL_CLEAN2"

SAMPLE_COUNT_CLEAN2=$(grep -oE '"pool_count":\s*[0-9]+' "$MODEL_CLEAN2" | head -n 1 | grep -oE '[0-9]+')
if [ "$SAMPLE_COUNT_CLEAN2" -ne 5 ]; then
    echo "ERROR: Expected 5 samples after -k -k, got $SAMPLE_COUNT_CLEAN2"
    exit 1
fi
echo "  [PASS] Repeated -k (-k -k) successfully pruned 2 outliers (pool_count 7 -> 5)."

# Test 4: Implicit threshold restoration from model JSON
echo "Test 4: Verify implicit threshold restoration from loaded model..."
SCORES_IMPLICIT="$TEST_DIR/scores_implicit.txt"
run_geif_score -r "$MODEL_CLEAN1" -a "$TEST_CSV" -L 1 -C 5 -p "%l;%s;%o" -o "$SCORES_IMPLICIT"

cat "$SCORES_IMPLICIT"
# In test.csv: test_norm should be inlier (o=0), test_wild should be outlier (o=1)
if ! grep -q "^test_norm;.*;0$" "$SCORES_IMPLICIT"; then
    echo "ERROR: test_norm was flagged as outlier"
    exit 1
fi
if ! grep -q "^test_wild;.*;1$" "$SCORES_IMPLICIT"; then
    echo "ERROR: test_wild was flagged as inlier"
    exit 1
fi
echo "  [PASS] Implicit threshold from model JSON correctly evaluated."

# Test 5: Dynamic threshold alias (-O average)
echo "Test 5: Dynamic threshold alias (-O average)..."
SCORES_AVG="$TEST_DIR/scores_avg.txt"
run_geif_score -r "$MODEL_CLEAN1" -a "$TEST_CSV" -L 1 -C 5 -O "average" -p "%l;score=%s;o=%o" -o "$SCORES_AVG"

cat "$SCORES_AVG"
if ! grep -q "^test_wild;score=.*;o=1$" "$SCORES_AVG"; then
    echo "ERROR: test_wild was not flagged as outlier under -O average"
    exit 1
fi
echo "  [PASS] -O average dynamic threshold evaluated successfully."

# Test 6: Scaled threshold alias (-O 0.65s)
echo "Test 6: Scaled threshold alias (-O 0.65s)..."
SCORES_SCALED="$TEST_DIR/scores_scaled.txt"
run_geif_score -r "$MODEL_CLEAN1" -a "$TEST_CSV" -L 1 -C 5 -O "0.65s" -p "%l;score=%s;o=%o" -o "$SCORES_SCALED"

cat "$SCORES_SCALED"
if ! grep -q "^test_wild;score=.*;o=1$" "$SCORES_SCALED"; then
    echo "ERROR: test_wild was not flagged as outlier under -O 0.65s"
    exit 1
fi
echo "  [PASS] -O 0.65s threshold evaluated successfully."

# Test 7: Prune during analysis and save updated model (-r ... -a ... -k -w ...)
echo "Test 7: In-line outlier pruning during analysis (-r ... -a ... -k -w ...)..."
MODEL_INLINE="$TEST_DIR/model_inline.json"
run_geif_score -r "$MODEL_RAW" -a "$TEST_CSV" -L 1 -C 5 -k -w "$MODEL_INLINE"

SAMPLE_COUNT_INLINE=$(grep -oE '"pool_count":\s*[0-9]+' "$MODEL_INLINE" | head -n 1 | grep -oE '[0-9]+')
if [ "$SAMPLE_COUNT_INLINE" -ne 6 ]; then
    echo "ERROR: Expected 6 samples in inline-pruned model, got $SAMPLE_COUNT_INLINE"
    exit 1
fi
echo "  [PASS] In-line pruning during analysis produced clean model with pool_count 6."

# Test 8: Explicit Percentile Threshold (-O 95%)
echo "Test 8: Percentage-based percentile threshold (-O 95%)..."
SCORES_PCT="$TEST_DIR/scores_pct.txt"
run_geif_score -r "$MODEL_CLEAN1" -a "$TEST_CSV" -L 1 -C 5 -O "95%" -v -p "%l;score=%s;o=%o" -o "$SCORES_PCT" 2>&1 | tee "$TEST_DIR/pct_log.txt"

if ! grep -q "Percentage score for 'catA':" "$TEST_DIR/pct_log.txt"; then
    echo "ERROR: -O 95% did not compute or log percentage score"
    exit 1
fi
echo "  [PASS] -O 95% computed dynamic percentile score successfully."

echo "=== All GEIF Recalibration & Threshold Integration Tests Passed! ==="
